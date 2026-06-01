"""Uygulama ayarları — tümü ortam değişkenlerinden okunur (12-factor)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- MQTT ---
    mqtt_host: str = "mosquitto"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_keepalive: int = 30

    topic_telemetry: str = "greenhouse/telemetry"
    topic_cmd: str = "greenhouse/cmd"
    topic_state: str = "greenhouse/state"
    topic_status: str = "greenhouse/status"

    # --- InfluxDB 2.x ---
    influx_url: str = "http://influxdb:8086"
    influx_token: str = "dev-token-change-me"
    influx_org: str = "greenhouse"
    influx_bucket: str = "greenhouse"
    influx_enabled: bool = True

    # --- Kontrol döngüsü ---
    control_interval_s: float = 5.0     # eşik değerlendirme periyodu
    # ESP32 bu süre içinde komut almazsa güvenli moda geçer (firmware watchdog ile uyumlu olmalı)
    command_resend_s: float = 30.0      # durum değişmese bile komutu tazeleme periyodu

    # --- Sunucu ---
    cors_origins: str = "*"             # virgülle ayrılmış; React dev sunucusu için


settings = Settings()
