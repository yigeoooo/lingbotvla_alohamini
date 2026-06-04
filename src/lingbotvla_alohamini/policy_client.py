from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LingBotPolicyClient:
    host: str
    port: int
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        try:
            from websockets.sync.client import connect  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "websockets is missing. Run `pip install -e .` in this adapter."
            ) from exc

        self._socket = None
        self.server_metadata: dict[str, Any] | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def connect(self) -> None:
        from websockets.sync.client import connect

        self._socket = connect(
            self.url,
            compression=None,
            max_size=None,
            open_timeout=self.timeout_s,
            close_timeout=self.timeout_s,
        )
        self.server_metadata = _unpack(self._socket.recv())

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def reset(self, robo_name: str) -> Any:
        return self.request({"reset": True, "robo_name": robo_name})

    def infer(self, observation: dict[str, Any]) -> Any:
        return self.request(observation)

    def request(self, payload: dict[str, Any]) -> Any:
        if self._socket is None:
            self.connect()
        assert self._socket is not None
        self._socket.send(_pack(payload))
        response = self._socket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot server returned text error: {response}")
        return _unpack(response)


def _pack(payload: dict[str, Any]) -> bytes:
    try:
        from deploy.msgpack_numpy import Packer

        return Packer().pack(payload)
    except ImportError:
        import msgpack
        import msgpack_numpy as m

        return msgpack.packb(payload, default=m.encode, use_bin_type=True)


def _unpack(payload: bytes | str) -> Any:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    try:
        from deploy.msgpack_numpy import unpackb

        return unpackb(payload)
    except ImportError:
        import msgpack
        import msgpack_numpy as m

        return msgpack.unpackb(payload, object_hook=m.decode, raw=False)
