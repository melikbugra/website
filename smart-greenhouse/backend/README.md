# Sera Backend (MQTT köprüsü + kontrol + API)

ESP32 ↔ MQTT ↔ **bu servis** ↔ InfluxDB + React.
Telemetriyi dinler, InfluxDB'ye yazar, sunucu taraflı kontrol döngüsünü çalıştırır,
React'e WebSocket (canlı) + REST (komut/ayar) sunar.

## Mimari

```
ESP32 ──MQTT──> Mosquitto ──> backend ──> InfluxDB (RL/ML verisi)
                  ^              │  └─WS/REST─> React (pano + kontrol)
                  └────cmd───────┘
```

## Çalıştırma

```bash
cp .env.example .env          # token/parolaları değiştir
docker compose up -d --build
docker compose logs -f backend
```

Servisler: Mosquitto `1883` (+ WS `9001`), InfluxDB `8086`, backend `8000`.
Opsiyonel Grafana: `docker compose --profile grafana up -d` → `3001`.

> Üretim: `mosquitto/mosquitto.conf` içinde `allow_anonymous false` yapıp parola
> dosyası tanımla; servisleri nginx proxy manager arkasına al.

## Donanımsız test (mock ESP32)

```bash
pip install aiomqtt        # ya da backend image'ı içinden çalıştır
python scripts/mock_publisher.py
```

Mock; fizik simülasyonuyla telemetri yayınlar ve komutlara tepki verir
(fan açılınca sıcaklık düşer, pompa açılınca toprak nemi artar). Auto kontrol
döngüsünü gözle doğrulamak için idealdir.

## API

| Yöntem | Yol | Açıklama |
|--------|-----|----------|
| GET  | `/api/health` | MQTT/cihaz bağlantı durumu |
| GET  | `/api/state` | Tam durum görüntüsü (telemetri + mod + eşik + komut) |
| GET/PUT | `/api/thresholds` | Otomatik kontrol eşiklerini oku/güncelle |
| PUT  | `/api/mode` | `{"mode":"auto"\|"manual"}` |
| POST | `/api/command` | Manuel komut (yalnızca manuel modda) |
| WS   | `/ws` | Canlı telemetri/komut/durum yayını |

## MQTT topic'leri

| Topic | Yön | İçerik |
|-------|-----|--------|
| `greenhouse/telemetry` | ESP32→ | sensör + aktüatör JSON |
| `greenhouse/cmd` | →ESP32 | `{fan_pwm, pump, humidifier, light}` |
| `greenhouse/status` | ESP32→ | `online`/`offline` (LWT) |

## InfluxDB sorgusu

```bash
docker compose exec influxdb influx query \
  'from(bucket:"greenhouse")|>range(start:-1h)|>filter(fn:(r)=>r._measurement=="environment")' \
  --org greenhouse --token <INFLUX_TOKEN>
```

Measurement `environment`, alanlar: `temp, humidity, pressure, lux, soil_0..n,
water_ok, act_fan_pwm, act_pump, act_humidifier, act_light` — aktüatör aksiyonları
da kayıtlı olduğu için RL eğitimi için (durum, aksiyon) çiftleri hazırdır.

## Kontrol mantığı

`app/control.py::compute_command` eşik + histerezisle hedef komutu üretir.
İleride bu fonksiyonun yerini mlflow'dan yüklenen RL politikası alabilir
(imza sabit tutuldu). Eşikler `/api/thresholds` ile çalışırken değiştirilebilir.
