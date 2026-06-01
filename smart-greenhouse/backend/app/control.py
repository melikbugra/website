"""Sunucu taraflı kontrol mantığı.

Auto modda telemetriyi + eşikleri okuyup hedef komutu hesaplar.
İleride bu fonksiyonun yerini mlflow'dan yüklenen bir RL politikası alabilir
(compute_command imzası aynı kalacak şekilde tasarlandı).
"""
from __future__ import annotations

from datetime import datetime

from .models import Command, Telemetry, Thresholds


def _fan_pwm(temp: float | None, th: Thresholds) -> int:
    if temp is None:
        return 0
    if temp <= th.temp_fan_on:
        return 0
    if temp >= th.temp_fan_full:
        return 255
    # temp_fan_on..temp_fan_full aralığında fan_min_pwm..255 doğrusal
    span = th.temp_fan_full - th.temp_fan_on
    frac = (temp - th.temp_fan_on) / span if span > 0 else 1.0
    return int(th.fan_min_pwm + frac * (255 - th.fan_min_pwm))


def _pump(soil: list[float], pump_was_on: bool, th: Thresholds) -> bool:
    if not soil:
        return False
    avg = sum(soil) / len(soil)
    # histerezis: kuruyken aç, ıslanınca kapat
    if pump_was_on:
        return avg < th.soil_wet
    return avg < th.soil_dry


def _humidifier(humidity: float | None, was_on: bool, th: Thresholds) -> bool:
    if humidity is None:
        return False
    if was_on:
        return humidity < th.humidity_high
    return humidity < th.humidity_low


def _light(lux: float | None, now: datetime, th: Thresholds) -> bool:
    hour = now.hour
    in_window = th.light_on_hour <= hour < th.light_off_hour
    if not in_window:
        return False
    if lux is None:
        return True  # gündüz penceresinde ışık verisi yoksa güvenli taraf: aç
    return lux < th.lux_low


def compute_command(
    t: Telemetry,
    th: Thresholds,
    prev: Command,
    now: datetime | None = None,
) -> Command:
    """Telemetri + eşiklerden hedef aktüatör komutunu üret."""
    now = now or datetime.now()
    pump_on = _pump(t.soil, prev.pump, th)
    # su deposu boşsa pompayı asla çalıştırma (kuru çalışma koruması)
    if t.water_ok is False:
        pump_on = False
    return Command(
        fan_pwm=_fan_pwm(t.temp, th),
        pump=pump_on,
        humidifier=_humidifier(t.humidity, prev.humidifier, th),
        light=_light(t.lux, now, th),
    )
