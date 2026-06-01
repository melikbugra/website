# Akıllı Sera — Alışveriş Listesi & Yapılacaklar

Faz 2/3 (elektronik + otomasyon) için pratik kontrol listesi. Teknik detaylar:
[`plan.md`](./plan.md) Bölüm 2.3 + [`firmware/README.md`](./firmware/README.md).

Elinde **ESP32 NodeMCU** zaten var. ✅

---

## 1. Alışveriş Listesi

### Sensörler
- [ ] **BME280** (I2C — sıcaklık/nem/basınç) × 1 — *0x76 adresli modül, 3.3V*
- [ ] **Kapasitif toprak nemi sensörü v2.0** × 3 — *kapasitif olduğundan emin ol, rezistif ALMA*
- [ ] **BH1750** (I2C ışık/lux sensörü) × 1
- [ ] **Şamandıra seviye şalteri** × 1 — *su deposu için, dikey veya yatay*
- [ ] *(opsiyonel)* **DS18B20 waterproof** + 4.7kΩ direnç × 1 — *toprak sıcaklığı*

### Aktüatörler
- [ ] **120mm 12V PC fanı** × 3 — *2 egzoz + 1 intake. Mümkünse **4-pinli PWM** model*
- [ ] **4 kanal opto-izoleli röle modülü** × 1 — *3.3V tetikli olan; pompa+nem+ışık+yedek*
- [ ] **12V mini dalgıç su pompası** × 1
- [ ] **Damla sulama kiti** (hortum + dripper/kazık + manifold) × 1 set
- [ ] **Ultrasonik sis/atomizer modülü** × 1 — *nemlendirici, 5V/24V disk + sürücü*
- [ ] **Full-spectrum LED grow-light** (12V şerit veya panel) × 1

### Sürücü / elektronik
- [ ] **Logic-level N-MOSFET IRLZ44N** × 2-3 — *2-pinli fan ve/veya pompa PWM için*
- [ ] **Direnç paketi** — *gate için 100-220Ω, pulldown için 10kΩ; DS18B20 için 4.7kΩ*
- [ ] **Flyback diyot 1N4007 (veya Schottky)** × 3-4 — *pompa/indüktif yükler için*

### Güç
- [ ] **12V ~5A güç adaptörü** × 1
- [ ] **Buck konvertör 12V→5V** (MP1584 veya LM2596) × 1 — *ESP32 + röle + sensör beslemesi*
- [ ] **İnline sigorta (5A) + yuva** × 1-2

### Kablolama & kasa
- [ ] **IP65 sızdırmaz kutu** + **kablo rakorları (gland)** — *elektroniği nemden korur*
- [ ] **Konformal kaplama spreyi** — *kart yüzeyine, korozyona karşı*
- [ ] **Vidalı klemensler**, JST/Dupont kablolar, perfboard veya küçük PCB
- [ ] Lehim, makaron (heatshrink), kablo bağı

> **Not:** 3'ten fazla toprak sensörü istersen ek olarak **CD74HC4067** (16 kanal analog mux) al — ESP32'nin ADC1 pinleri sınırlı.

---

## 2. Sırasıyla Yapılacaklar

### Aşama A — Sipariş & hazırlık
1. [ ] Yukarıdaki listeyi sipariş et.
2. [ ] Beklerken yazılımı tanı: donanımsız demoyu çalıştır (aşağıda Aşama F'deki mock).

### Aşama B — Masa üstü prototip (sera DIŞINDA, breadboard'da)
> Her şeyi seraya monte etmeden önce masada test et — hata ayıklaması çok daha kolay.
3. [ ] ESP32'yi breadboard'a tak, USB ile besle.
4. [ ] **I2C sensörleri bağla:** BME280 + BH1750 → SDA=GPIO21, SCL=GPIO22, 3.3V + GND.
5. [ ] **Toprak sensörlerini bağla:** sinyal uçları GPIO 36, 39, 34 (ADC1!), besleme 3.3V.
6. [ ] **Şamandırayı bağla:** GPIO33 ↔ GND (firmware INPUT_PULLUP kullanır).
7. [ ] **Güç hattını kur:** 12V adaptör → buck → 5V. **Tüm GND'leri birleştir** (12V, buck, ESP32, röle).
8. [ ] **Röle modülünü bağla:** IN'ler GPIO 26 (pompa), 27 (nem), 13 (ışık); VCC/GND.
9. [ ] **Fanı bağla:** MOSFET gate → GPIO25 (gate'e 150Ω, gate-GND arası 10kΩ); fan 12V'tan, MOSFET drain'den. *(4-pinli PWM fan kullanıyorsan PWM ucu doğrudan GPIO25.)*
10. [ ] Pompa/ışık/nem yüklerini röle çıkışlarına tak; pompa hattına flyback diyot.

### Aşama C — Sunucuyu ayağa kaldır
11. [ ] `cd backend && cp .env.example .env` → token/parolaları değiştir.
12. [ ] `docker compose up -d --build` → Mosquitto + InfluxDB + backend çalışsın.
13. [ ] ESP32'nin bağlanacağı **sunucu IP'sini** not et (`MQTT_HOST` olacak).

### Aşama D — Firmware
14. [ ] PlatformIO kur (VS Code eklentisi veya `pip install platformio`).
15. [ ] `firmware/src/config.example.h` → `config.h` kopyala; WiFi, `MQTT_HOST`, pinleri gir.
16. [ ] USB ile yükle: `pio run -t upload && pio device monitor`.
17. [ ] Seri monitörde WiFi + MQTT bağlantısını doğrula.
18. [ ] Backend loglarında telemetrinin geldiğini gör (`docker compose logs -f backend`).

### Aşama E — Kalibrasyon & test
19. [ ] **Toprak sensörü kalibrasyonu:** sensörü kuru havada ve suda oku, ham ADC değerlerini `config.h`'taki `SOIL_RAW_DRY` / `SOIL_RAW_WET`'e gir, tekrar yükle.
20. [ ] BME280 değerini bilinen bir termometreyle kıyasla.
21. [ ] **Manuel test:** React'i aç (Aşama F), Manuel moda geç, fan/pompa/ışık/nem'i tek tek çalıştır — fiziksel tepkiyi doğrula.
22. [ ] **Failsafe testi:** broker'ı durdur (`docker compose stop mosquitto`), 60 sn sonra ESP32 güvenli moda geçsin (pompa kapalı).

### Aşama F — Arayüz
23. [ ] `cd greenhouse-3d && cp .env.example .env` → `VITE_API_BASE`'i sunucu adresine ayarla.
24. [ ] `npm install && npm run dev` → tarayıcıda canlı değerler + kontrol paneli.
25. [ ] Eşikleri ayarla (sıcaklık/nem/toprak) ve Otomatik moda al.

### Aşama G — Fiziksel montaj (sera İÇİNE)
26. [ ] Elektroniği IP65 kutuya al, kartı konformal kapla, kabloları rakordan geçir.
27. [ ] Fanları tavana (2 egzoz) + alt arkaya (1 intake) monte et.
28. [ ] BME280'i üst-orta, toprak sensörlerini saksılara, şamandırayı su deposuna yerleştir.
29. [ ] Pompa + damla sulama hortumlarını çek, grow-light'ı as.
30. [ ] Kutuyu mümkünse nemli bölgenin dışına/üstüne sabitle.

### Aşama H — Devreye alma
31. [ ] Bitkileri ek, sistemi otomatik modda birkaç gün izle, eşikleri ince ayar yap.
32. [ ] *(opsiyonel, ileride)* InfluxDB'de biriken veriyle mlflow'da RL politikası eğit.

---

## Kritik hatırlatmalar ⚠️
- **Toprak sensörleri ADC1'de** (GPIO 32-39) olmalı — WiFi açıkken ADC2 okunmaz.
- **Strapping pinlerini** (GPIO 0, 2, 12, 15) röle/aktüatöre bağlama — ESP32 açılmaz.
- **Ortak GND** olmadan hiçbir şey düzgün çalışmaz.
- Röle modülün **active-LOW** ise `config.h`'ta `RELAY_ACTIVE_LOW 1` kalsın.
- Polikarbonat değil, elektronik nemden korunmalı — kutu + konformal kaplama atlanmaz.
