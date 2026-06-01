"""InfluxDB 2.x'e telemetri yazımı (RL/ML için zaman serisi birikimi)."""
from __future__ import annotations

import logging

from influxdb_client import Point
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.write_api import SYNCHRONOUS  # noqa: F401  (tip referansı)

from .config import settings
from .models import Telemetry

log = logging.getLogger("greenhouse.influx")


class InfluxWriter:
    def __init__(self) -> None:
        self._client: InfluxDBClientAsync | None = None
        self._write_api = None

    async def start(self) -> None:
        if not settings.influx_enabled:
            log.warning("InfluxDB devre dışı (INFLUX_ENABLED=false) — yazım atlanacak")
            return
        try:
            self._client = InfluxDBClientAsync(
                url=settings.influx_url,
                token=settings.influx_token,
                org=settings.influx_org,
            )
            self._write_api = self._client.write_api()
            log.info("InfluxDB bağlantısı hazır: %s", settings.influx_url)
        except Exception as exc:  # bağlantı kurulamasa bile servis ayakta kalsın
            log.error("InfluxDB başlatılamadı: %s", exc)
            self._client = None

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def write_telemetry(self, t: Telemetry) -> None:
        if self._write_api is None:
            return
        try:
            point = (
                Point("environment")
                .tag("device", "greenhouse-esp32")
            )
            for field in ("temp", "humidity", "pressure", "lux", "soil_temp", "rssi", "uptime"):
                val = getattr(t, field)
                if val is not None:
                    point.field(field, float(val))
            if t.water_ok is not None:
                point.field("water_ok", 1.0 if t.water_ok else 0.0)
            for i, s in enumerate(t.soil):
                point.field(f"soil_{i}", float(s))
            # aktüatör durumlarını da kaydet (RL için aksiyon geçmişi)
            point.field("act_fan_pwm", float(t.actuators.fan_pwm))
            point.field("act_pump", 1.0 if t.actuators.pump else 0.0)
            point.field("act_humidifier", 1.0 if t.actuators.humidifier else 0.0)
            point.field("act_light", 1.0 if t.actuators.light else 0.0)

            await self._write_api.write(bucket=settings.influx_bucket, record=point)
        except Exception as exc:
            log.warning("InfluxDB yazımı başarısız (atlanıyor): %s", exc)


influx = InfluxWriter()
