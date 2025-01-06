from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from typing import Any, Dict, Generator

from .task_group import TaskGroup
from .worker_context import AsyncioSingleTask, WorkerContext
from ..config import Config
from ..events import Closed, Event, RawData, Updated
from ..extensions.tls import TLS_CIPHER_SUITES, TLS_VERSIONS
from ..protocol import ProtocolWrapper
from ..typing import AppWrapper, ConnectionState, LifespanState
from ..utils import parse_socket_addr

MAX_RECV = 2**16


class TCPServer:
    def __init__(
        self,
        app: AppWrapper,
        loop: asyncio.AbstractEventLoop,
        config: Config,
        context: WorkerContext,
        state: LifespanState,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.app = app
        self.config = config
        self.context = context
        self.loop = loop
        self.protocol: ProtocolWrapper
        self.reader = reader
        self.writer = writer
        self.send_lock = asyncio.Lock()
        self.state = state
        self.idle_task = AsyncioSingleTask()

    def __await__(self) -> Generator[Any, None, None]:
        return self.run().__await__()

    async def run(self) -> None:
        socket = self.writer.get_extra_info("socket")
        try:
            client = parse_socket_addr(socket.family, socket.getpeername())
            server = parse_socket_addr(socket.family, socket.getsockname())
            ssl_object = self.writer.get_extra_info("ssl_object")
            tls: Dict[str, Any] = {
                "server_cert": None,
                "client_cert_chain": [],
                "client_cert_name": None,
                "client_cert_error": None,
                "tls_version": None,
                "cipher_suite": None,
            }
            if ssl_object is not None:
                _ssl = True
                alpn_protocol = ssl_object.selected_alpn_protocol()
                if client_cert_chain := ssl_object.getpeercert(binary_form=True):
                    tls["client_cert_chain"].append(ssl.DER_cert_to_PEM_cert(client_cert_chain))
                if cipher := ssl_object.cipher():
                    (cipher_name, ssl_version, _cipher_nbits) = cipher
                    tls["cipher_suite"] = TLS_CIPHER_SUITES.get(cipher_name)
                    tls["tls_version"] = TLS_VERSIONS.get(ssl_version)
                server_cert = Path(self.config.certfile).resolve()
                if server_cert.is_file():
                    tls["server_cert"] = Path(server_cert).read_text()
            else:
                _ssl = False
                alpn_protocol = "http/1.1"

            async with TaskGroup(self.loop) as task_group:
                self._task_group = task_group
                self.protocol = ProtocolWrapper(
                    self.app,
                    self.config,
                    self.context,
                    task_group,
                    ConnectionState(self.state.copy()),
                    _ssl,
                    tls,
                    client,
                    server,
                    self.protocol_send,
                    alpn_protocol,
                )
                await self.protocol.initiate()
                await self.idle_task.restart(task_group, self._idle_timeout)
                await self._read_data()
        except OSError:
            pass
        finally:
            await self._close()

    async def protocol_send(self, event: Event) -> None:
        if isinstance(event, RawData):
            async with self.send_lock:
                try:
                    self.writer.write(event.data)
                    await self.writer.drain()
                except (ConnectionError, RuntimeError):
                    await self.protocol.handle(Closed())
        elif isinstance(event, Closed):
            await self._close()
        elif isinstance(event, Updated):
            if event.idle:
                await self.idle_task.restart(self._task_group, self._idle_timeout)
            else:
                await self.idle_task.stop()

    async def _read_data(self) -> None:
        while not self.reader.at_eof():
            try:
                data = await asyncio.wait_for(self.reader.read(MAX_RECV), self.config.read_timeout)
            except (
                ConnectionError,
                OSError,
                asyncio.TimeoutError,
                TimeoutError,
                ssl.SSLError,
            ):
                break
            else:
                await self.protocol.handle(RawData(data))

        await self.protocol.handle(Closed())

    async def _close(self) -> None:
        try:
            self.writer.write_eof()
        except (NotImplementedError, OSError, RuntimeError):
            pass  # Likely SSL connection

        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (
            BrokenPipeError,
            ConnectionAbortedError,
            ConnectionResetError,
            RuntimeError,
            asyncio.CancelledError,
        ):
            pass  # Already closed
        finally:
            await self.idle_task.stop()

    async def _initiate_server_close(self) -> None:
        await self.protocol.handle(Closed())
        self.writer.close()

    async def _idle_timeout(self) -> None:
        try:
            await asyncio.wait_for(self.context.terminated.wait(), self.config.keep_alive_timeout)
        except asyncio.TimeoutError:
            pass
        await asyncio.shield(self._initiate_server_close())
