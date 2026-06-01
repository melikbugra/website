"""Sahte ESP32 — donanım olmadan uçtan uca akışı test eder.

Telemetri yayınlar, `greenhouse/cmd` komutlarını dinler ve basit bir fizik
simülasyonuyla tepki verir (fan açılınca sıcaklık düşer, pompa açılınca toprak
nemi artar, vb.). Böylece auto kontrol döngüsünü gözle doğrulayabilirsin.

Kullanım:
    pip install aiomqtt
    python scripts/mock_publisher.py            # mosquitto localhost:1883
    MQTT_HOST=1.2.3.4 python scripts/mock_publisher.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
from datetime import datetime

import aiomqtt

HOST = os.environ.get("MQTT_HOST", "localhost")
PORT = int(os.environ.get("MQTT_PORT", "1883"))
PERIOD = float(os.environ.get("PERIOD", "3"))

T_TELEMETRY = "greenhouse/telemetry"
T_CMD = "greenhouse/cmd"
T_STATUS = "greenhouse/status"

# --- Simülasyon durumu ---
sim = {
    "temp": 24.0,
    "humidity": 55.0,
    "soil": [40.0, 38.0, 42.0],
    "lux": 5000.0,
    "water_ok": True,
}
actuators = {"fan_pwm": 0, "pump": False, "humidifier": False, "light": False}


def step() -> None:
    """Bir zaman adımı: aktüatörlerin ortama etkisini uygula."""
    # Sıcaklık: gündüz ısınır; fan soğutur
    hour = datetime.now().hour + datetime.now().minute / 60
    ambient = 22 + 8 * math.sin((hour - 6) / 24 * 2 * math.pi)  # gün içi salınım
    sim["temp"] += (ambient - sim["temp"]) * 0.05
    sim["temp"] -= (actuators["fan_pwm"] / 255) * 1.2          # fan etkisi
    sim["temp"] += random.uniform(-0.1, 0.1)

    # Nem: yavaş düşer; nemlendirici yükseltir; fan biraz düşürür
    sim["humidity"] -= 0.4
    if actuators["humidifier"]:
        sim["humidity"] += 3.0
    sim["humidity"] -= (actuators["fan_pwm"] / 255) * 0.5
    sim["humidity"] = max(20, min(95, sim["humidity"] + random.uniform(-0.3, 0.3)))

    # Toprak nemi: buharlaşmayla düşer; pompa açıkken artar
    for i in range(len(sim["soil"])):
        sim["soil"][i] -= 0.3
        if actuators["pump"] and sim["water_ok"]:
            sim["soil"][i] += 4.0
        sim["soil"][i] = max(0, min(100, sim["soil"][i] + random.uniform(-0.2, 0.2)))

    # Işık: gün ışığı + grow-light
    daylight = max(0, 30000 * math.sin((hour - 6) / 12 * math.pi))
    sim["lux"] = daylight + (6000 if actuators["light"] else 0) + random.uniform(-200, 200)


def telemetry_payload() -> str:
    return json.dumps({
        "temp": round(sim["temp"], 2),
        "humidity": round(sim["humidity"], 1),
        "pressure": round(1013 + random.uniform(-2, 2), 1),
        "soil": [round(s, 1) for s in sim["soil"]],
        "lux": round(sim["lux"], 0),
        "water_ok": sim["water_ok"],
        "actuators": dict(actuators),
        "rssi": random.randint(-70, -45),
        "uptime": int(asyncio.get_event_loop().time()),
    })


async def listen_commands(client: aiomqtt.Client) -> None:
    async for msg in client.messages:
        try:
            cmd = json.loads(msg.payload)
            for k in actuators:
                if k in cmd:
                    actuators[k] = cmd[k]
            print(f"  ← cmd: {cmd}")
        except Exception as exc:
            print(f"  ! geçersiz cmd: {exc}")


async def main() -> None:
    print(f"Mock ESP32 → {HOST}:{PORT} (her {PERIOD}s)")
    async with aiomqtt.Client(HOST, PORT, will=aiomqtt.Will(T_STATUS, "offline", retain=True)) as client:
        await client.publish(T_STATUS, "online", retain=True)
        await client.subscribe(T_CMD, qos=1)
        listener = asyncio.create_task(listen_commands(client))
        try:
            while True:
                step()
                payload = telemetry_payload()
                await client.publish(T_TELEMETRY, payload, qos=1)
                print(f"→ T={sim['temp']:.1f}°C H={sim['humidity']:.0f}% "
                      f"soil={[round(s) for s in sim['soil']]} lux={sim['lux']:.0f} "
                      f"| fan={actuators['fan_pwm']} pump={actuators['pump']} "
                      f"hum={actuators['humidifier']} light={actuators['light']}")
                await asyncio.sleep(PERIOD)
        finally:
            listener.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDurduruldu.")
