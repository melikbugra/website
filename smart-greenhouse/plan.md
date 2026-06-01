# Akıllı Balkon Serası ve Tarım Otomasyonu Projesi

## 1. Proje Özeti
Bu proje, kapalı cam balkon içerisinde 120x80x100 cm boyutlarında izole bir polikarbonat sera inşa edilmesini ve bu seranın iklim/sulama koşullarının ESP32 (edge) & Docker sunucu mimarisiyle otonom olarak yönetilmesini amaçlamaktadır.

* **Boyutlar:** 120 cm (En) x 80 cm (Boy) x 100 cm (Yükseklik)
* **Hacim:** ~0.96 Metreküp
* **Odak:** Meyve/Sebze yetiştiriciliği (Çilek, Çeri Domates, Biber, Salatalık)

---

## 2. Mekanik ve Donanım Mimarisi

### 2.1. İskelet ve Kasa
* **Karkas:** 45×45 mm Emprenye Çam Çıta. (Uygun maliyet, kolay işlenebilirlik. Sera içi neme karşı emprenye veya 2 kat dış cephe verniği ile korunmalıdır).
* **Kaplama:** 4 mm Şeffaf Oluklu Polikarbonat Levha. (Isı yalıtımı ve ışık geçirgenliği sağlar).
* **Bağlantı:** Metal L-köşe braketleri + 4×40 mm çinko kaplama ahşap vidalar. Her köşede 2 adet L-braketi.
* **Erişim:** Ön cephede (120×100) menteşeli çift kapı sistemi. Hava kaçaklarını önlemek için yalıtım fitili kullanılmalıdır.

#### 2.1.1. Malzeme Listesi — İskelet

| # | Malzeme | Adet / Miktar | Açıklama | Durum |
|---|---------|---------------|----------|-------|
| 1 | 45×45 mm emprenye çam çıta | 7 adet × 2m (14 m) | 4×120cm (A), 4×71cm (B, iç ölçü), 5×100cm (C+D). Kesim planı Adım 1'de | Alındı |
| 2 | 4 mm şeffaf oluklu polikarbonat levha | 1 adet (600×210 cm) | ~5 m² ihtiyaç, tek levhadan tüm paneller çıkar. Kaynak: olukmarket.com | Alındı |
| 3 | Metal L-köşe braketi (50×50 mm) | 16+ adet | Her köşe için 2 adet (8 köşe × 2) | Bekliyor |
| 4 | Çinko kaplama ahşap vida 4×40 mm | ~80 adet | İskelet braketleri için | Bekliyor |
| 5 | Çinko kaplama ahşap vida 4×30 mm + geniş pul (rondela) | ~60 adet | Polikarbonat panel montajı, 20-25cm aralıkla. Delik vidadan 1-2mm geniş açılmalı (genleşme payı) | Bekliyor |
| 6 | Dış cephe ahşap koruyucu / vernik | 1 kutu (~750 ml) | Tüm yüzeylere montaj öncesi 2 kat | Bekliyor |
| 6 | Yalıtım fitili (kapı contası) | ~4 m | Kapı çerçevesi çevresi | Bekliyor |
| 7 | 20×30mm çam çıta (kapı çerçevesi) | ~6m (3×2m veya hazır kesim) | Her kanat: 2×91cm dikey + 2×52cm yatay. Toplam 4 dikey + 4 yatay = 8 parça | Bekliyor |
| 8 | Menteşe (orta boy) | 4 adet | Çift kapı, her kanatta 2 adet | Bekliyor |
| 9 | Kapı mandalı / mıknatıslı kilit | 2 adet | İki kapı kanadı için | Bekliyor |

#### 2.1.2. İnşaat Adımları — Faz 1

> **Gereken Aletler:** Akülü matkap/vidalama, ahşap testere (el testeresi veya dekupaj yeterli), metre, gönye (90° kontrolü), kurşun kalem, zımpara (120-180 kum), matkap uçları (3mm ahşap için, 6mm polikarbonat delikleri için), maket bıçağı.

---

**ADIM 1 — Çıta Kesimi ve Yüzey Hazırlığı**

Çıtaları aşağıdaki ölçülere kes. Kesim öncesi metreyle ölç, kurşun kalemle çevreden işaretle:

**45×45mm iskelet çıtaları:**

| Parça | Uzunluk | Adet | Kullanım |
|-------|---------|------|----------|
| A | 120 cm | 4 | Alt ön, alt arka, üst ön, üst arka |
| B | 71 cm | 4 | Alt sol, alt sağ, üst sol, üst sağ (iç ölçü: 80 - 4.5 - 4.5) |
| C | 100 cm | 4 | Köşe dikmeleri (dört köşe) |
| D | 100 cm | 1 | Kapı orta bölücü dikme |

**20×30mm kapı çerçevesi çıtaları:**

| Parça | Uzunluk | Adet | Kullanım |
|-------|---------|------|----------|
| E | 91 cm | 4 | Kapı çerçevesi dikey (2 kanat × 2) |
| F | 52 cm | 4 | Kapı çerçevesi yatay (2 kanat × 2) |

Kesim sonrası (her iki çıta tipi için):
1. Tüm kesim yüzeylerini 120 kum zımpara ile düzelt (çapak kaldırma).
2. Tüm çıtalara **2 kat dış cephe verniği / ahşap koruyucu** sür.
3. İlk kat kuruyunca (etiketindeki süreye bak, genellikle 2-4 saat) ikinci katı sür.
4. İkinci kat tamamen kuruyana kadar bekle (genellikle 12-24 saat). **Montaja kurumadan başlama.**

```
45×45mm kesim planı (7 adet × 2 metre çıtadan):

Çıta 1: [====120 cm====][=71 cm=]     → 1×A + 1×B (+9cm fire)
Çıta 2: [====120 cm====][=71 cm=]     → 1×A + 1×B
Çıta 3: [====120 cm====][=71 cm=]     → 1×A + 1×B
Çıta 4: [====120 cm====][=71 cm=]     → 1×A + 1×B
Çıta 5: [===100 cm===][===100 cm===]  → 2×C
Çıta 6: [===100 cm===][===100 cm===]  → 1×C + 1×D
Çıta 7: [===100 cm===][fire ~100cm]   → son C

20×30mm kesim planı (~6m, 3 adet × 2m çıtadan):

Çıta 1: [===91 cm===][===91 cm===]    → 2×E (+18cm fire)
Çıta 2: [===91 cm===][===91 cm===]    → 2×E
Çıta 3: [==52 cm==][==52 cm==][==52 cm==][==52cm==] → 4×F (tam 208cm, sıkışık!)
```

> **Not:** 20×30mm çıtalardan 3. çıta çok sıkışık (4×52=208cm). Testere payı düşünülürse 4 adet × 2m almak daha güvenli.

---

**ADIM 2 — Alt Çerçeveyi Kur (Zemin Karesi)**

Bu adımda 4 çıtayı (2×A + 2×B) yere düz şekilde dikdörtgen çerçeve olarak birleştiriyorsun.

```
  ←───── 120 cm (A) ─────→
  ┌──────────────────────────┐
  │                          │  ↑
  │                          │  80 cm (B)
  │                          │  ↓
  └──────────────────────────┘
```

Birleştirme sırası:
1. İki A çıtasını (120 cm) paralel olarak yere koy, aralarında 80 cm mesafe bırak.
2. İki B çıtasını (80 cm) uçlara yerleştir. B çıtaları A çıtalarının **iç tarafına** gelecek (yani toplam dış ölçü 120 × 80+4.5+4.5 = 120 × 89 cm değil, 120 × 80 cm olmalı). Bunu sağlamak için B çıtalarının uzunluğunu **80 - 4.5 - 4.5 = 71 cm** olarak kesmek gerekir.

> **DİKKAT — Ölçü Düzeltmesi:** Dış ölçü tam 120×80 cm olsun istiyorsan, B çıtaları iki A arasına girdiği için B boyutu = 80 - 4.5 - 4.5 = **71 cm** olmalı. Alternatif olarak A çıtalarını 111 cm yapıp B'yi 80 cm bırakabilirsin. **Hangisini seçersen tüm adımlarda tutarlı ol.** Bu kılavuzda A=120 cm dıştan dışa, B=71 cm iç ölçü olarak devam ediyoruz.

Montaj:
1. A ve B çıtalarını köşede hizala. Gönye ile 90° olduğunu kontrol et.
2. L-braketini köşenin iç tarafına yerleştir. Braketi iki çıtanın iç yüzeyine daya.
3. Matkap ile 3 mm ön delikleri aç (ahşabın çatlamasını önler).
4. 4×40 mm vidalarla braketi sabitle (her kola 2 vida).
5. Aynı işlemi 4 köşede tekrarla. Her köşeye 1 L-braketi yeterli (alt çerçevede üstten de braketi gelecek).

```
Köşe detayı (üstten görünüm):

     B (71 cm)
     ┌────────
     │ ┌─┐
     │ │L│ ← L-braketi iç köşeye
     │ └─┘
─────┘
A (120 cm)
```

Bitince çerçeveyi yere koy ve köşeden köşeye çapraz mesafeleri ölç. İki çapraz eşitse çerçeve kare (90°). Eşit değilse braketleri hafif gevşetip ayarla, sonra tekrar sık.

---

**ADIM 3 — Dikey Dikmeleri Dik (4 Köşe + 1 Orta)**

4 adet C çıtasını (100 cm) alt çerçevenin 4 köşesine dikey olarak monte et.

```
      C                C
      │                │
      │   (üstten)     │
  ────┼────────────────┼────  A (ön)
      │                │
      │                │
  ────┼────────────────┼────  A (arka)
      │                │
      C                C
```

Her dikmede:
1. C çıtasını köşeye dik olarak tut. Varsa birisi tutmasına yardım etsin, yoksa köşeye geçici olarak işkence/mengene ile sabitle.
2. L-braketini dikme ile alt yatay çıta arasına yerleştir (dikmenin iki açık yüzüne birer braketi).
3. Her braket için 3 mm ön delik aç, 4×40 mm vidala.
4. Su terazisi veya telefon terazisi uygulamasıyla dikmenin düşey olduğunu kontrol et.

```
Köşe dikme detayı (yandan görünüm):

         │ C (dikme)
    ┌────┤
    │ L  │  ← L-braketi
    └────┤
─────────┘  A veya B (alt yatay)
```

4 köşe dikmesi bittikten sonra **kapı orta bölücü** (D çıtası, 100 cm) dikmeyi monte et:
- Ön A çıtasının tam ortasına (60 cm noktasına) yerleştir.
- Alt A çıtasına L-braketi ile bağla.
- Bu dikme iki kapı kanadının buluştuğu noktadır.

---

**ADIM 4 — Üst Çerçeveyi Kur (Tavan Karesi)**

4 köşe dikmenin üst ucuna 2×A + 2×B çıtası bağla. Alt çerçevenin aynısı, bu sefer havada:

1. Bir A çıtasını (120 cm, arka) iki arka dikmenin üstüne koy, uçları hizala.
2. L-braketi ile her iki köşeyi sabitle.
3. Bir A çıtasını (120 cm, ön) iki ön dikmenin üstüne koy ve aynı şekilde monte et.
4. İki B çıtasını (71 cm) sol ve sağ tarafta ön-arka dikmelerin üst uçları arasına yerleştir.
5. Orta dikmenin (D) üstünü ön üst A çıtasına L-braketi ile bağla.

> **İPUCU:** Üst çerçeveyi monte ederken yapı sallanabilir. Her köşeyi sabitledikten sonra yapıyı hafifçe salla — sağlamsa devam et, değilse eksik braketleri ekle.

Tüm iskelet tamamlandığında yapı böyle görünmeli:

```
     ┌─────────────────────────┐
    /│                        /│
   / │                       / │
  ┌──┼──────────────────────┐  │
  │  │     (tavan)          │  │
  │  │                      │  │  100 cm
  │  │         │←orta dikme │  │
  │  └─────────┼────────────┤  │
  │ /          │           │  /
  │/           │           │ /
  └────────────┼───────────┘
       ←─── 120 cm ───→
              80 cm
```

---

**ADIM 5 — Polikarbonat Levha Kesimi**

600×210 cm levhadan aşağıdaki panelleri kes. Kesim için maket bıçağı + düz cetvel yeterli — polikarbonat kolay kesilir.

| Panel | Boyut (cm) | Adet | Not |
|-------|-----------|------|-----|
| Arka panel | 120 × 100 | 1 | Çıta dış yüzeyine oturacak |
| Sol yan panel | 71 × 100 | 1 | Derinlik iç ölçü (B çıtası boyu) |
| Sağ yan panel | 71 × 100 | 1 | Aynı |
| Tavan paneli | 120 × 71 | 1 | Üst çerçeveye oturacak |
| Sol kapı kanadı | ~56 × 95 | 1 | İskelet iç ölçüsüne göre ayarla |
| Sağ kapı kanadı | ~56 × 95 | 1 | Aynı |

Kesim öncesi:
1. Levhayı düz bir zemine yayar.
2. Metreyle ölç, kurşun kalemle **levhanın koruyucu filmi üzerine** işaretle.
3. Metal cetvel/düz tahta kenarını çizgiye daya.
4. Maket bıçağıyla cetvel boyunca 2-3 kez çiz. Derin kesmene gerek yok, çizdikten sonra kenardan kır.
5. Koruyucu filmi montajdan sonra sök (çizilmeyi önler).

```
Kesim planı (600 × 210 cm levha):

┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  [Arka 120×100]  [Sol yan 71×100]  [Sağ yan 71×100]  [Tavan 120×71] │  210 cm
│                                                                      │
│  [Sol kapı 56×95] [Sağ kapı 56×95]  [     fire      ]               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              600 cm
```

---

**ADIM 6 — Panel Montajı (Arka, Yanlar, Tavan)**

Montaj sırası: **Arka → Sol yan → Sağ yan → Tavan**. Kapı en son.

Her panel için aynı prosedür:
1. Paneli çıtanın dış yüzeyine yerleştir, kenarları hizala.
2. Paneli geçici olarak sabitle (bant veya birisi tutsun).
3. Matkap ile **6 mm uçla** polikarbonatta ön delikleri aç (vida 4 mm, delik 6 mm — bu 2 mm fark ısı genleşme payı).
4. Delikleri 20-25 cm aralıklarla aç. Panel kenarlarından en az 2 cm içeride kal (kırılma riski).
5. Her deliğe **geniş pul (rondela) + 4×30 mm vida** ile sabitle.
6. Vidayı **tam sıkma** — pul levhaya hafifçe bassın ama levhayı eğmesin. Çok sıkınca polikarbonat çatlar veya genleşince kırılır.

```
Panel montaj detayı (kesit görünüm):

   vida başı
   ┌─┐
   │●│
 ──┘ └──  ← geniş pul/rondela
═════════  ← polikarbonat levha
█████████  ← ahşap çıta
           ↑
     2mm boşluk (genleşme payı)
```

**Arka panel (120×100 cm):**
- 4 arka çıtaya (alt A, üst A, sol arka dikme, sağ arka dikme) vidalanır.
- Üst ve alt kenarlarda ~6 vida, yanlarda ~5 vida ≈ toplam ~22 vida.

**Yan paneller (71×100 cm):**
- Her biri 4 çıtaya (alt B, üst B, ön dikme, arka dikme) vidalanır.
- Her panelde ~16 vida.

**Tavan paneli (120×71 cm):**
- Üst çerçevedeki 4 çıtaya vidalanır.
- ~18 vida.

> **İPUCU:** Tavan panelini monte ederken fan delikleri için yer bırakmayı unutma. Faz 2'de 2 adet 120mm fan deliği açılacak. Şimdilik monte et, fan deliğini sonra açarsın.

---

**ADIM 7 — Kapı Yapımı ve Montajı**

Sera ön yüzü (120×100 cm) iki kapı kanadından oluşur. Her kanat 20×30mm çıtadan yapılmış bir çerçeve + üzerine vidalanmış polikarbonat panelden oluşur.

**7a — Kapı çerçevesi yapımı (20×30mm çıta):**

4mm polikarbonat tek başına kapı olamaz — eğilir, menteşe delikleri yırtar. Her kanat için ince çıtadan dikdörtgen çerçeve yapılmalı.

Kesim ölçüleri (her kanat için):
| Parça | Uzunluk | Adet | Not |
|-------|---------|------|-----|
| Dikey | 91 cm | 2 | Kapı boşluğu yüksekliği (~95cm) - üst/alt çıta payı |
| Yatay | 52 cm | 2 | Kapı boşluğu genişliği (~56cm) - iki dikey çıta payı (56 - 2×2 = 52cm) |

İki kanat = toplam 4 dikey (91cm) + 4 yatay (52cm) = ~5.7m çıta.

Çerçeve montajı:
1. Her kanat için 2 dikey + 2 yatay çıtayı dikdörtgen olarak birleştir.
2. Köşelerde 3mm ön delik aç, 3×30mm vida ile sabitle (veya her köşeye küçük L-braketi).
3. Gönye ile 90° kontrolü yap.
4. Çerçevenin dış ölçüsü ~56×95 cm olmalı — kapı boşluğundan her yandan 1-2mm küçük (sürtmemesi için).

**7b — Polikarbonat montajı (çerçeveye):**

1. Kesilen kapı panelini (~56×95 cm) çerçevenin bir yüzüne koy.
2. 6mm matkap ucuyla polikarbonatta delik aç (genleşme payı).
3. Geniş pul + 4×30mm vida ile 20cm aralıkla sabitle.
4. Aynı kural: vidayı aşırı sıkma, pul levhaya hafifçe bassın yeter.
5. Her kanat için ~10-12 vida yeterli.

**7c — Menteşe montajı:**

Her kapı kanadı için 2 menteşe (biri üste, biri alta):
1. Menteşeyi **önce iskelet dikme çıtasına** vida ile sabitle. Sol kanat → sol dikme, sağ kanat → sağ dikme.
2. Üst menteşeyi dikmede yukarıdan ~15 cm, alt menteşeyi aşağıdan ~15 cm noktaya koy.
3. Menteşenin diğer kanadını **kapı çerçevesinin dikey çıtasına** vidale (polikarbonata değil!).
4. Kapıyı aç-kapa test et, sürtüyorsa menteşe pozisyonunu ayarla.

```
Sol dikme (45×45)     Kapı çerçevesi (20×30)
      │                    │
      │──[menteşe]─────────│  ← üst menteşe (~15 cm yukarıdan)
      │                    │
      │             [PC levha bu çerçevenin
      │              üzerine vidalı]
      │                    │
      │──[menteşe]─────────│  ← alt menteşe (~15 cm aşağıdan)
      │                    │
```

**7d — Yalıtım fitili (conta):**

1. D-profil yapışkanlı contayı al.
2. Kapının kapandığında temas ettiği yüzeylere yapıştır:
   - Sol dikme çıtanın iç yüzü (sol kapı temas noktası)
   - Sağ dikme çıtanın iç yüzü
   - Orta bölücü dikmenin (D) her iki yüzü (iki kapı kanadının buluştuğu yer)
   - Alt ve üst çıtanın kapıya bakan yüzü
3. Contayı yapıştırmadan önce ahşap yüzeyi temiz ve kuru yap.

```
Conta yerleşimi (üstten görünüm):

  sol dikme        orta dikme        sağ dikme
     │                 │                 │
     │▓  sol kapı   ▓│▓  sağ kapı   ▓│
     │                 │                 │

  ▓ = yapışkanlı conta şeridi
```

**7e — Kapı mandalı / mıknatıslı kilit:**

Her kapı kanadının serbest kenarına (orta dikmeye yakın tarafına) mıknatıslı dolap mandalı veya basit sürgü monte et. Kapı kapandığında orta dikmeye tutunur.

---

**ADIM 8 — Son Kontroller**

1. Tüm vidaları tek tek kontrol et — gevşeyen varsa sık (ama polikarbonat vidalarını aşırı sıkma).
2. İki kapıyı aç-kapa test et. Conta kapı kapandığında hafifçe sıkışmalı.
3. Yapıyı hafifçe salla — sallanıyorsa zayıf köşelere ek L-braketi ekle.
4. Polikarbonat üzerindeki koruyucu filmi **şimdi** sök (montaj sonrası).
5. İskeletin balkon zemininde kaymadığından emin ol. Gerekirse alt çıtaların altına kaydırmaz keçe veya silikon ayak yapıştır.

> **Faz 1 tamamlandı.** Sera artık kapalı bir kutu. Faz 2'de fanlar, Raspberry Pi ve sensörler monte edilecek.

---

### 2.2. İklimlendirme ve Havalandırma
* **Egzoz (Sıcak Hava Çıkışı):** Tavan kısmına yerleştirilmiş 2 adet 120mm 12V PC Fanı. Sıcaklık 28°C'yi aştığında içerideki havayı tahliye eder.
* **Temiz Hava Girişi:** Alt seviyelerde pasif hava kanalları veya içeriye taze hava basacak 1 adet 120mm 12V PC fanı (Intake).
* **Nemlendirme:** Röle kontrollü ultrasonik buhar modülü veya mini nemlendirici.

### 2.3. Elektronik Bileşenler (Edge Layer)
* **Kontrolcü:** ESP32 NodeMCU (WiFi dahili mikrodenetleyici — Raspberry Pi'nin yerini alır, daha ucuz ve bu iş için yeterli). Firmware: Arduino / PlatformIO (C++).
* **Sensörler:**
    * BME280 (ortam sıcaklık + nem + basınç, I2C) — seranın üst orta kısmına. DHT22'ye göre nemde daha dayanıklı ve doğru.
    * Kapasitif Toprak Nemi Sensörleri ×3 (korozyon direnci için kesinlikle kapasitif). **ESP32'de ADC1 pinlerine (GPIO 32-39) bağlanmalı** — WiFi açıkken ADC2 okunamaz.
    * BH1750 (ışık/lux, I2C) — izleme + grow-light kontrolü.
    * Şamandıra seviye şalteri (su deposu kuru çalışma koruması).
* **Eylemciler (Aktüatörler):**
    * Opto-izoleli çoklu röle kartı (pompa, nemlendirici, grow-light). ESP32 3.3V mantık — 3.3V tetikli modül veya JD-VCC ayrımı.
    * Fanlar: logic-level MOSFET (IRLZ44N) ile PWM hız kontrolü (veya 4-pin PWM fan).
    * 12V Mini Dalgıç Su Pompası (sulama rezervuarı için, flyback diyot ile).
    * Ultrasonik sis modülü (nemlendirme), full-spectrum LED grow-light (aydınlatma).
* **Güç Yönetimi:** 12V ~5A adaptör (fanlar, pompa, ışık) ve buck konvertör 12V→5V (ESP32 + röle + sensör beslemesi). **Ortak GND şart.** 12V hattına inline sigorta.
* **Montaj:** Elektronik IP65 sızdırmaz kutuda + kablo rakorları; kart konformal kaplı (sera nemi korozyon yapar).

---

## 3. Yazılım ve Veri Mimarisi

Sistem, edge (cihaz) ve cloud (sunucu) olmak üzere iki katmanlı çalışır.

### 3.1. ESP32 (Edge) — `firmware/`
* **Dil:** Arduino / PlatformIO (C++).
* **Görev:** Sensörleri periyodik okumak (~5 sn) ve telemetriyi JSON olarak yayınlamak; sunucudan gelen komutları aktüatörlere uygulamak. **Karar vermez** — kontrol mantığı sunucudadır.
* **Haberleşme:** MQTT (PubSubClient). Yayınlar: `greenhouse/telemetry`, `greenhouse/status` (online/offline, LWT). Dinler: `greenhouse/cmd`.
* **Failsafe:** 60 sn komut gelmezse güvenli moda geçer (pompa/nem/ışık kapanır, fan son sıcaklığa göre) — sunucu/WiFi çökse bile bitkiler korunur.
* **OTA:** ArduinoOTA ile kablosuz güncelleme.

### 3.2. Sunucu (Docker host — mevcut altyapı) — `backend/`
ESP32 verisini toplayan, kontrol kararını veren ve React'e sunan katman. `/home/melik/website/` altındaki mevcut Docker + nginx proxy manager ortamına eklenir (ayrı VPS gerekmez).
* **Veri Alımı:** Eclipse Mosquitto (MQTT Broker).
* **Köprü + Kontrol:** Python / FastAPI servisi — telemetriyi InfluxDB'ye yazar, sunucu taraflı kontrol döngüsünü çalıştırır (eşik + histerezis), komutları yayınlar; React'e WebSocket (canlı) + REST (komut/eşik/mod) sunar.
* **Veritabanı:** InfluxDB 2.x (zaman serisi). Aktüatör aksiyonları da kaydedilir → (durum, aksiyon) çiftleri RL için hazır.
* **Görselleştirme + Kontrol:** Mevcut React 3D uygulaması (`greenhouse-3d/`) canlı dijital ikiz + kontrol paneli olur (oto/manuel mod, fan/pompa/ışık/nem). Grafana opsiyonel profil olarak mevcut.
* **Gelişmiş Kontrol (Opsiyonel):** `backend/app/control.py::compute_command` imzası sabit tutuldu; ilerleyen aşamada eşik mantığının yerini mlflow'dan yüklenen bir pekiştirmeli öğrenme (RL) politikası alabilir. Bu model, enerji tüketimini ve sıcaklık dalgalanmalarını minimize ederek en optimal politikayı öğrenebilir.

---

## 4. Botanik Planlama ve Yerleşim

Meyve veren sebzeler yüksek su, derin toprak ve yoğun besin gerektirir. 100 cm yükseklik kısıtlaması nedeniyle bitki seçimi ve yerleşimi kritiktir.

### 4.1. Yetiştirilecek Türler
1.  **Çeri Domates:** Mutlaka "bodur" veya "oturak" cins (determinate) seçilmelidir. Sırık domatesler 2 metreye ulaşabileceği için 100 cm'lik kabine sığmaz.
2.  **Biber (Sivri/Jalapeno):** Güneşi ve sıcağı sever. 20-25 cm derinliğinde toprak ister.
3.  **Salatalık:** Sarılıcı bir bitkidir. Kabinin arka profillerine bir ağ (trellis) gerilerek salatalığın yukarı doğru tırmanması sağlanmalıdır. Çok su tüketir.
4.  **Çilek:** Sığ köklüdür, diğer bitkilerin diplerinde veya kabinin ön kısımlarına asılacak ufak saksılarda (dikey tarım mantığıyla) yetiştirilebilir.

### 4.2. Sulama ve Saksı Düzeni
* **Saksılar:** Dikdörtgen balkon saksıları (en az 20 cm derinlik). Seranın zeminine yan yana dizilecektir.
* **Sulama Sistemi:** Su deposuna atılan 12V dalgıç pompadan çıkan ana hortum, saksıların üzerinden geçer. Ana hortuma takılan "damla sulama kazıkları" ile her bitkinin köküne eşit su verilir.
* **Toprak:** Suyu iyi süzdüren, perlit ve torf karışımlı sebze toprağı kullanılmalıdır. Ayrıca bu türler ürün verirken yoğun efor harcadığı için sulama suyuna periyodik olarak sıvı solucan gübresi veya NPK gübresi eklenmelidir.

---

## 5. Proje Fazları ve Yol Haritası

* **Faz 1:** Ahşap iskeletin kurulması ve polikarbonat kaplamanın yapılması. ✅ **Tamamlandı.**
* **Faz 2:** Fanların, ESP32'nin ve sensörlerin kabine fiziksel montajı (bkz. Bölüm 2.3 donanım listesi + `firmware/README.md` pin haritası).
* **Faz 3:** ESP32 firmware (`firmware/`) + sunucu veri/kontrol hattı (`backend/`: MQTT → InfluxDB → FastAPI → React). Mock simülatörle uçtan uca akış doğrulandı; gerçek donanımla devreye alınacak.
* **Faz 4:** Bitkilerin ekilmesi, damla sulama hortumlarının çekilmesi ve sistemin canlıya alınması.
* **Faz 5 (opsiyonel):** InfluxDB'de biriken (durum, aksiyon) verisiyle mlflow üzerinde RL politikası eğitimi ve eşik mantığının yerine konması.