"""MQTT bağlantısı: telemetri/durum dinleme + komut yayınlama (otomatik yeniden bağlanma)."""
from __future__ import annotations

import asyncio
import json
import logging

import aiomqtt
from pydantic import ValidationError

from .config import settings
from .influx import influx
from .models import Command, Telemetry
from .state import state

log = logging.getLogger("greenhouse.mqtt")


async def _handle_message(topic: str, payload: bytes) -> None:
    if topic == settings.topic_status:
        text = payload.decode(errors="ignore").strip().lower()
        state.device_online = text == "online"
        log.info("Cihaz durumu: %s", text)
        await state.broadcast({"type": "status", "online": state.device_online})
        return

    if topic == settings.topic_telemetry:
        try:
            t = Telemetry.model_validate_json(payload)
        except ValidationError as exc:
            log.warning("Geçersiz telemetri yoksayıldı: %s", exc)
            return
        state.update_telemetry(t)
        await influx.write_telemetry(t)
        await state.broadcast(state.snapshot())
        return

    # greenhouse/state — aktüatör onayı (telemetri içine de gömülü olabilir)
    if topic == settings.topic_state:
        log.debug("state mesajı: %s", payload[:200])


async def publish_command(cmd: Command) -> bool:
    """Komutu cmd topic'ine yayınla. Bağlantı yoksa False döner."""
    client = state.mqtt_client
    if client is None:
        log.warning("MQTT bağlı değil — komut yayınlanamadı")
        return False
    try:
        await client.publish(settings.topic_cmd, json.dumps(cmd.model_dump()), qos=1)
        state.last_command = cmd
        await state.broadcast({"type": "command", "command": cmd.model_dump()})
        return True
    except Exception as exc:
        log.warning("Komut yayını başarısız: %s", exc)
        return False


async def mqtt_loop() -> None:
    """Sonsuz dinleme döngüsü; kopunca 5 sn sonra yeniden bağlanır."""
    while True:
        try:
            async with aiomqtt.Client(
                hostname=settings.mqtt_host,
                port=settings.mqtt_port,
                username=settings.mqtt_username,
                password=settings.mqtt_password,
                keepalive=settings.mqtt_keepalive,
            ) as client:
                state.mqtt_client = client
                log.info("MQTT bağlandı: %s:%s", settings.mqtt_host, settings.mqtt_port)
                await client.subscribe(settings.topic_telemetry, qos=1)
                await client.subscribe(settings.topic_state, qos=1)
                await client.subscribe(settings.topic_status, qos=1)
                async for msg in client.messages:
                    await _handle_message(str(msg.topic), bytes(msg.payload))
        except aiomqtt.MqttError as exc:
            log.warning("MQTT bağlantısı koptu (%s) — 5 sn sonra yeniden denenecek", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("MQTT döngüsü beklenmeyen hata: %s", exc)
        finally:
            state.mqtt_client = None
            state.device_online = False
        await asyncio.sleep(5)
