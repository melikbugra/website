# Akıllı Balkon Serası

Kapalı balkon içinde 120×80×100 cm izole polikarbonat sera + ESP32 tabanlı iklim/sulama otomasyonu. Tam plan ve donanım listesi: [`plan.md`](./plan.md).

## Mimari

```
ESP32 (sensör+aktüatör) ──MQTT──> Mosquitto ──> backend (FastAPI)
                                                  ├─> InfluxDB (RL/ML verisi)
                                                  └─> WebSocket/REST ─> React 3D (pano + kontrol)
```
Kontrol kararları **sunucuda** verilir; ESP32 okur + uygular (komut kesilirse failsafe).

## Bileşenler

| Dizin | Ne | Çalıştırma |
|-------|-----|-----------|
| [`firmware/`](./firmware) | ESP32 NodeMCU firmware (PlatformIO/C++) | `pio run -t upload` |
| [`backend/`](./backend) | MQTT köprüsü + kontrol + InfluxDB + API (Docker) | `docker compose up -d --build` |
| [`greenhouse-3d/`](./greenhouse-3d) | 3D dijital ikiz + canlı veri panosu (React) | `npm install && npm run dev` |

## Hızlı başlangıç (donanımsız test)

```bash
# 1) sunucu tarafı
cd backend && cp .env.example .env && docker compose up -d --build
# 2) sahte ESP32 (fizik simülasyonu, komutlara tepki verir)
pip install aiomqtt && python scripts/mock_publisher.py
# 3) arayüz
cd ../greenhouse-3d && cp .env.example .env && npm install && npm run dev
```
React'te canlı değerleri gör, Manuel moda geçip fan/pompa/ışığı kontrol et.

## Durum

Faz 1 (inşaat) ✅ tamamlandı. Faz 2/3 (elektronik + yazılım) — yazılım hattı
mock ile uçtan uca doğrulandı, gerçek donanımla devreye alma bekliyor.
