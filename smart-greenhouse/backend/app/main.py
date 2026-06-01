"""FastAPI uygulaması: MQTT köprüsü, kontrol döngüsü, WebSocket + REST API."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .control import compute_command
from .influx import influx
from .models import Command, ManualCommand, ModeUpdate, Thresholds
from .mqtt_client import mqtt_loop, publish_command
from .state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("greenhouse.main")


async def control_loop() -> None:
    """Auto modda eşiklere göre komut üretir; manuel modda hedef komutu tazeler."""
    last_publish = 0.0
    while True:
        await asyncio.sleep(settings.control_interval_s)
        try:
            now = time.monotonic()
            if state.mode == "auto":
                if state.latest is None:
                    continue
                cmd = compute_command(state.latest, state.thresholds, state.last_command)
            else:
                cmd = state.manual_command

            changed = cmd.model_dump() != state.last_command.model_dump()
            stale = (now - last_publish) >= settings.command_resend_s
            if changed or stale:
                if await publish_command(cmd):
                    last_publish = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Kontrol döngüsü hatası: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await influx.start()
    tasks = [
        asyncio.create_task(mqtt_loop(), name="mqtt"),
        asyncio.create_task(control_loop(), name="control"),
    ]
    log.info("Sera backend başladı (mod=%s)", state.mode)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await influx.stop()


app = FastAPI(title="Smart Greenhouse Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- REST ----------------
@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "mqtt_connected": state.mqtt_client is not None, "device_online": state.device_online}


@app.get("/api/state")
async def get_state() -> dict:
    return state.snapshot()


@app.get("/api/thresholds")
async def get_thresholds() -> Thresholds:
    return state.thresholds


@app.put("/api/thresholds")
async def set_thresholds(th: Thresholds) -> Thresholds:
    state.thresholds = th
    await state.broadcast({"type": "thresholds", "thresholds": th.model_dump()})
    return th


@app.put("/api/mode")
async def set_mode(body: ModeUpdate) -> dict:
    state.mode = body.mode
    # manuel moda geçerken hedefi mevcut komutla başlat (ani sıçrama olmasın)
    if body.mode == "manual":
        state.manual_command = Command(**state.last_command.model_dump())
    await state.broadcast({"type": "mode", "mode": state.mode})
    log.info("Mod değişti: %s", state.mode)
    return {"mode": state.mode}


@app.post("/api/command")
async def manual_command(body: ManualCommand) -> dict:
    """Manuel kontrol. Yalnızca manuel modda etkili; gönderilen alanları günceller."""
    if state.mode != "manual":
        return {"applied": False, "reason": "manuel mod kapalı"}
    merged = state.manual_command.model_dump()
    for k, v in body.model_dump(exclude_none=True).items():
        merged[k] = v
    state.manual_command = Command(**merged)
    ok = await publish_command(state.manual_command)
    return {"applied": ok, "command": state.manual_command.model_dump()}


# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await state.ws_connect(ws)
    try:
        while True:
            # istemciden mesaj beklemiyoruz; bağlantıyı canlı tutmak için dinle
            await ws.receive_text()
    except WebSocketDisconnect:
        state.ws_disconnect(ws)
    except Exception:
        state.ws_disconnect(ws)
