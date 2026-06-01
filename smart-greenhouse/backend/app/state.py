"""Paylaşılan çalışma-zamanı durumu: en son telemetri, mod, eşikler ve WS yayını."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket

from .models import Command, Telemetry, Thresholds

log = logging.getLogger("greenhouse.state")


class AppState:
    def __init__(self) -> None:
        self.latest: Telemetry | None = None
        self.latest_ts: datetime | None = None
        self.device_online: bool = False
        self.mode: str = "auto"                       # "auto" | "manual"
        self.thresholds: Thresholds = Thresholds()
        self.last_command: Command = Command()        # en son yollanan komut
        self.manual_command: Command = Command()      # manuel modda hedeflenen durum

        # aiomqtt client referansı (publish için); bağlantı yokken None
        self.mqtt_client = None

        self._ws_clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ---- WebSocket istemci yönetimi ----
    async def ws_connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._ws_clients.add(ws)
        # bağlanır bağlanmaz mevcut durumu yolla
        await self._safe_send(ws, self.snapshot())

    def ws_disconnect(self, ws: WebSocket) -> None:
        self._ws_clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self._ws_clients):
            ok = await self._safe_send(ws, payload)
            if not ok:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    @staticmethod
    async def _safe_send(ws: WebSocket, payload: dict) -> bool:
        try:
            await ws.send_text(json.dumps(payload, default=str))
            return True
        except Exception:
            return False

    # ---- Durum güncelleme ----
    def update_telemetry(self, t: Telemetry) -> None:
        self.latest = t
        self.latest_ts = datetime.now(timezone.utc)
        self.device_online = True

    def snapshot(self) -> dict:
        """React'e yollanacak tam durum görüntüsü."""
        return {
            "type": "snapshot",
            "online": self.device_online,
            "mode": self.mode,
            "ts": self.latest_ts.isoformat() if self.latest_ts else None,
            "telemetry": self.latest.model_dump() if self.latest else None,
            "thresholds": self.thresholds.model_dump(),
            "command": self.last_command.model_dump(),
        }


state = AppState()
