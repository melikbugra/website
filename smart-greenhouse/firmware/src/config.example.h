// config.h örneği — bu dosyayı `config.h` olarak kopyalayıp kendi değerlerini gir.
// config.h .gitignore'dadır (WiFi parolası repo'ya girmesin).
#pragma once

// ---------- WiFi ----------
#define WIFI_SSID      "WIFI_ADIN"
#define WIFI_PASSWORD  "WIFI_PAROLAN"

// ---------- MQTT ----------
#define MQTT_HOST      "192.168.1.50"   // backend/mosquitto host IP'si
#define MQTT_PORT      1883
#define MQTT_USER      ""               // anonim ise boş bırak
#define MQTT_PASS      ""
#define MQTT_CLIENT_ID "greenhouse-esp32"
#define OTA_HOSTNAME   "greenhouse"
#define OTA_PASSWORD   "ota-parola"

// ---------- Topic'ler (backend ile aynı olmalı) ----------
#define TOPIC_TELEMETRY "greenhouse/telemetry"
#define TOPIC_CMD       "greenhouse/cmd"
#define TOPIC_STATUS    "greenhouse/status"

// ---------- Zamanlama ----------
#define TELEMETRY_INTERVAL_MS 5000UL    // telemetri yayın periyodu
#define FAILSAFE_TIMEOUT_MS   60000UL   // bu süre komut gelmezse güvenli moda geç
                                        // (backend command_resend_s=30s'nin ~2 katı)

// ---------- Pin haritası ----------
// Kurallar: kapasitif toprak sensörleri ADC1'de (GPIO 32-39); strapping pinleri
// (0,2,12,15) aktüatör için kullanma; 34-39 yalnızca giriştir.
#define PIN_I2C_SDA      21
#define PIN_I2C_SCL      22

#define PIN_SOIL_0       36   // ADC1
#define PIN_SOIL_1       39   // ADC1
#define PIN_SOIL_2       34   // ADC1
#define SOIL_COUNT       3

#define PIN_WATER_FLOAT  33   // şamandıra (dahili pull-up destekli pin), kapanınca su var

#define PIN_FAN_PWM      25   // MOSFET gate veya 4-pin fan PWM
#define PIN_RELAY_PUMP   26
#define PIN_RELAY_HUMID  27
#define PIN_RELAY_LIGHT  13

#define USE_DS18B20      0    // 1 yaparsan toprak sıcaklığı okunur
#define PIN_DS18B20      4

// Röle modülü active-LOW ise 1 yap (çoğu ucuz modül böyledir)
#define RELAY_ACTIVE_LOW 1

// ---------- Toprak nemi kalibrasyonu ----------
// Sensörü kuru havadayken ve suya batırıp ölç, bu iki değeri gir.
#define SOIL_RAW_DRY     3000   // kuru (yüksek ADC)
#define SOIL_RAW_WET     1200   // ıslak (düşük ADC)

// ---------- Failsafe güvenli durum ----------
// Komut kesilince fanı açacak sıcaklık eşiği (°C)
#define FAILSAFE_FAN_TEMP 28.0
