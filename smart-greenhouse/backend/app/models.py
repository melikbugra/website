"""Pydantic veri modelleri — telemetri, komut, eşikler."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Actuators(BaseModel):
    """Aktüatörlerin o anki durumu (fan hızı + aç/kapa yükler)."""
    fan_pwm: int = Field(0, ge=0, le=255)
    pump: bool = False
    humidifier: bool = False
    light: bool = False


class Telemetry(BaseModel):
    """ESP32'nin `greenhouse/telemetry` topic'ine yolladığı paket."""
    temp: float | None = None          # °C, hava
    humidity: float | None = None      # %RH
    pressure: float | None = None      # hPa
    soil: list[float] = Field(default_factory=list)  # her bölge için %nem (0-100)
    lux: float | None = None           # ışık
    water_ok: bool | None = None       # su deposu seviye şalteri (True = su var)
    soil_temp: float | None = None     # opsiyonel DS18B20
    actuators: Actuators = Field(default_factory=Actuators)
    rssi: int | None = None
    uptime: int | None = None          # saniye


class Command(BaseModel):
    """Backend'in `greenhouse/cmd` topic'ine yolladığı komut."""
    fan_pwm: int = Field(0, ge=0, le=255)
    pump: bool = False
    humidifier: bool = False
    light: bool = False


class Thresholds(BaseModel):
    """Otomatik kontrol eşikleri — REST ile güncellenebilir."""
    temp_fan_on: float = 26.0          # bu sıcaklıktan itibaren fan dönmeye başlar
    temp_fan_full: float = 32.0        # bu sıcaklıkta fan tam hız (255)
    fan_min_pwm: int = Field(80, ge=0, le=255)   # dönmeye başladığında minimum hız
    soil_dry: float = 30.0             # bu %nemin altında sulama tetiklenir
    soil_wet: float = 55.0             # bu %neme ulaşınca pompa durur
    humidity_low: float = 50.0         # bu %RH altında nemlendirici açılır
    humidity_high: float = 70.0        # bu %RH üstünde nemlendirici kapanır
    lux_low: float = 8000.0            # bu lux altında (gündüz saatinde) ışık açılır
    light_on_hour: int = Field(7, ge=0, le=23)
    light_off_hour: int = Field(21, ge=0, le=23)


class ModeUpdate(BaseModel):
    mode: Literal["auto", "manual"]


class ManualCommand(BaseModel):
    """React'ten manuel kontrol — yalnızca gönderilen alanlar değişir."""
    fan_pwm: int | None = Field(None, ge=0, le=255)
    pump: bool | None = None
    humidifier: bool | None = None
    light: bool | None = None
