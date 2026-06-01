# Sera ESP32 Firmware (PlatformIO)

ESP32 NodeMCU: sensörleri okur → MQTT telemetri yayınlar; backend komutlarını
uygular; bağlantı koparsa güvenli moda geçer. **Kontrol kararları sunucuda**
verilir; ESP32 yalnızca okuma + uygulama yapar.

## Kurulum

1. [PlatformIO](https://platformio.org/install) kur (VS Code eklentisi veya `pip install platformio`).
2. `src/config.example.h`'ı `src/config.h` olarak kopyala, WiFi/MQTT/pin değerlerini gir.
3. İlk yükleme USB ile:
   ```bash
   pio run -t upload && pio device monitor
   ```
4. Sonraki yüklemeler kablosuz (OTA): `platformio.ini`'de `upload_protocol = espota`
   ve `upload_port = greenhouse.local` satırlarını aç.

## Pin haritası (config.h)

| İşlev | GPIO | Not |
|-------|------|-----|
| I2C SDA/SCL | 21 / 22 | BME280 + BH1750 |
| Toprak nemi ×3 | 36, 39, 34 | **ADC1 zorunlu** (WiFi açıkken ADC2 okunmaz) |
| Şamandıra | 33 | INPUT_PULLUP, su varken LOW |
| Fan PWM | 25 | MOSFET gate / 4-pin fan; 25 kHz |
| Röle pompa / nem / ışık | 26 / 27 / 13 | strapping pinlerinden (0,2,12,15) kaçınıldı |
| DS18B20 (ops.) | 4 | `USE_DS18B20 1` ile aç, 4.7k pull-up |

> Röle modülün active-LOW ise `RELAY_ACTIVE_LOW 1` (varsayılan) kalsın.

## Toprak nemi kalibrasyonu

Sensörü kuru havadayken ve suya batırıp ham ADC değerlerini seri porttan oku,
`SOIL_RAW_DRY` / `SOIL_RAW_WET` değerlerini güncelle.

## Failsafe (güvenli mod)

`FAILSAFE_TIMEOUT_MS` (varsayılan 60 sn) boyunca komut gelmezse: pompa/nem/ışık
KAPANIR, fan son sıcaklığa göre ayarlanır (≥`FAILSAFE_FAN_TEMP` ise tam hız).
Bu, sunucu/WiFi çökse bile bitkileri korur. Backend `command_resend_s`
(30 sn) periyodunda komut tazelediği için normal işleyişte tetiklenmez.

## MQTT sözleşmesi

- Yayınlar: `greenhouse/telemetry` (JSON), `greenhouse/status` (`online`/`offline`, LWT + retained).
- Dinler: `greenhouse/cmd` → `{fan_pwm:0-255, pump, humidifier, light}`.

Telemetri ve komut formatı backend `app/models.py` ile birebir uyumludur.
