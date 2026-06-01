/*
 * Akıllı Sera — ESP32 NodeMCU firmware
 *
 * Görev: sensörleri oku → JSON telemetri yayınla; backend'den gelen komutları
 * uygula; bağlantı koparsa güvenli moda geç (failsafe watchdog).
 * Kontrol kararları SUNUCUDA verilir — burada yalnızca okuma + uygulama var.
 */
#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_BME280.h>
#include <BH1750.h>

#include "config.h"

#if USE_DS18B20
#include <OneWire.h>
#include <DallasTemperature.h>
OneWire oneWire(PIN_DS18B20);
DallasTemperature ds18b20(&oneWire);
#endif

// ---------- LEDC (fan PWM) ----------
static const int FAN_PWM_CH = 0;
static const int FAN_PWM_FREQ = 25000;   // 25 kHz — duyulmaz
static const int FAN_PWM_RES = 8;        // 8-bit (0-255)

WiFiClient net;
PubSubClient mqtt(net);
Adafruit_BME280 bme;
BH1750 lightMeter;

bool bmeOk = false;
bool bhOk = false;

// Aktüatörlerin hedef durumu
struct Actuators {
  int fan_pwm = 0;
  bool pump = false;
  bool humidifier = false;
  bool light = false;
} act;

float lastTemp = NAN;
unsigned long lastTelemetry = 0;
unsigned long lastCmdMs = 0;
bool failsafeActive = false;

// ---------- Aktüatör uygulama ----------
inline void writeRelay(int pin, bool on) {
  bool level = RELAY_ACTIVE_LOW ? !on : on;
  digitalWrite(pin, level ? HIGH : LOW);
}

void applyActuators() {
  ledcWrite(FAN_PWM_CH, constrain(act.fan_pwm, 0, 255));
  writeRelay(PIN_RELAY_PUMP, act.pump);
  writeRelay(PIN_RELAY_HUMID, act.humidifier);
  writeRelay(PIN_RELAY_LIGHT, act.light);
}

// ---------- Sensör okuma ----------
int readSoilPct(int pin) {
  int raw = analogRead(pin);
  // kuru (yüksek raw) → 0%, ıslak (düşük raw) → 100%
  long pct = map(raw, SOIL_RAW_DRY, SOIL_RAW_WET, 0, 100);
  return constrain((int)pct, 0, 100);
}

// ---------- Telemetri yayını ----------
void publishTelemetry() {
  JsonDocument doc;

  if (bmeOk) {
    float t = bme.readTemperature();
    lastTemp = t;
    doc["temp"] = round(t * 10) / 10.0;
    doc["humidity"] = round(bme.readHumidity() * 10) / 10.0;
    doc["pressure"] = round(bme.readPressure() / 10.0) / 10.0; // hPa
  }
  if (bhOk) {
    doc["lux"] = round(lightMeter.readLightLevel());
  }
#if USE_DS18B20
  ds18b20.requestTemperatures();
  float st = ds18b20.getTempCByIndex(0);
  if (st > -100) doc["soil_temp"] = round(st * 10) / 10.0;
#endif

  JsonArray soil = doc["soil"].to<JsonArray>();
  int soilPins[SOIL_COUNT] = {PIN_SOIL_0, PIN_SOIL_1, PIN_SOIL_2};
  for (int i = 0; i < SOIL_COUNT; i++) soil.add(readSoilPct(soilPins[i]));

  // şamandıra: pull-up'lı, su varken kapalı (LOW) varsayımı
  doc["water_ok"] = (digitalRead(PIN_WATER_FLOAT) == LOW);

  JsonObject a = doc["actuators"].to<JsonObject>();
  a["fan_pwm"] = act.fan_pwm;
  a["pump"] = act.pump;
  a["humidifier"] = act.humidifier;
  a["light"] = act.light;

  doc["rssi"] = WiFi.RSSI();
  doc["uptime"] = (uint32_t)(millis() / 1000);

  char buf[384];
  size_t n = serializeJson(doc, buf);
  mqtt.publish(TOPIC_TELEMETRY, (const uint8_t*)buf, n, false);
}

// ---------- Komut işleme ----------
void onMessage(char* topic, byte* payload, unsigned int len) {
  if (strcmp(topic, TOPIC_CMD) != 0) return;
  JsonDocument doc;
  if (deserializeJson(doc, payload, len)) return;

  if (doc["fan_pwm"].is<int>())     act.fan_pwm    = doc["fan_pwm"].as<int>();
  if (doc["pump"].is<bool>())       act.pump       = doc["pump"].as<bool>();
  if (doc["humidifier"].is<bool>()) act.humidifier = doc["humidifier"].as<bool>();
  if (doc["light"].is<bool>())      act.light      = doc["light"].as<bool>();

  lastCmdMs = millis();
  failsafeActive = false;
  applyActuators();
}

// ---------- Güvenli mod ----------
void enterFailsafe() {
  // Komut akışı kesildi: sel/taşma riskli yükleri kapat, ışık/nem kapat,
  // fanı son sıcaklığa göre ayarla.
  act.pump = false;
  act.humidifier = false;
  act.light = false;
  if (!isnan(lastTemp) && lastTemp >= FAILSAFE_FAN_TEMP) {
    act.fan_pwm = 255;   // sıcaksa havalandır
  } else {
    act.fan_pwm = 0;
  }
  applyActuators();
}

// ---------- Bağlantı ----------
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(250);
  }
}

void ensureMqtt() {
  if (mqtt.connected()) return;
  while (!mqtt.connected() && WiFi.status() == WL_CONNECTED) {
    // LWT: bağlantı koparsa broker "offline" yayınlar (retained)
    bool ok = mqtt.connect(MQTT_CLIENT_ID, MQTT_USER, MQTT_PASS,
                           TOPIC_STATUS, 1, true, "offline");
    if (ok) {
      mqtt.publish(TOPIC_STATUS, "online", true);
      mqtt.subscribe(TOPIC_CMD, 1);
      lastCmdMs = millis();  // bağlanır bağlanmaz failsafe sayacını sıfırla
    } else {
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  // Aktüatör pinleri
  pinMode(PIN_RELAY_PUMP, OUTPUT);
  pinMode(PIN_RELAY_HUMID, OUTPUT);
  pinMode(PIN_RELAY_LIGHT, OUTPUT);
  pinMode(PIN_WATER_FLOAT, INPUT_PULLUP);

  ledcSetup(FAN_PWM_CH, FAN_PWM_FREQ, FAN_PWM_RES);
  ledcAttachPin(PIN_FAN_PWM, FAN_PWM_CH);
  applyActuators();  // başlangıçta her şey kapalı

  // I2C sensörler
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  bmeOk = bme.begin(0x76) || bme.begin(0x77);
  bhOk = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
#if USE_DS18B20
  ds18b20.begin();
#endif

  ensureWifi();

  // OTA
  ArduinoOTA.setHostname(OTA_HOSTNAME);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.begin();

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(512);
  mqtt.setCallback(onMessage);

  lastCmdMs = millis();
}

void loop() {
  ensureWifi();
  ensureMqtt();
  ArduinoOTA.handle();
  mqtt.loop();

  unsigned long now = millis();

  // Failsafe: belirli süre komut gelmezse güvenli duruma geç
  if (now - lastCmdMs > FAILSAFE_TIMEOUT_MS && !failsafeActive) {
    failsafeActive = true;
    enterFailsafe();
  }

  if (now - lastTelemetry >= TELEMETRY_INTERVAL_MS) {
    lastTelemetry = now;
    if (mqtt.connected()) publishTelemetry();
  }
}
