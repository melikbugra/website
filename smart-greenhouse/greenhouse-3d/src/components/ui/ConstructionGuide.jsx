import { useState } from 'react'
import {
  CuttingDiagram,
  BaseFrameDiagram,
  PostsDiagram,
  TopFrameDiagram,
  PCCuttingDiagram,
  PanelMountDiagram,
  DoorDiagram,
  ChecklistDiagram,
} from './diagrams'

const STEPS = [
  {
    title: 'Çıta Kesimi ve Yüzey Hazırlığı',
    icon: '1',
    Diagram: CuttingDiagram,
    materials: [
      '45×45mm emprenye çam çıta (7 × 2m) — iskelet',
      '20×30mm çam çıta (3-4 × 2m) — kapı çerçevesi',
      'Dış cephe verniği / ahşap koruyucu',
      'Zımpara (120 kum)',
    ],
    tools: ['Testere', 'Metre', 'Kurşun kalem', 'Gönye', 'Fırça'],
    instructions: [
      '45×45mm İSKELET ÇITALARI:',
      '  A parçası: 4 adet × 120cm (ön/arka yatay)',
      '  B parçası: 4 adet × 71cm (sol/sağ yatay, iç ölçü!)',
      '  C parçası: 4 adet × 100cm (köşe dikmeleri)',
      '  D parçası: 1 adet × 100cm (kapı orta dikme)',
      '20×30mm KAPI ÇERÇEVESİ ÇITALARI:',
      '  E parçası: 4 adet × 91cm (dikey, 2 kanat × 2)',
      '  F parçası: 4 adet × 52cm (yatay, 2 kanat × 2)',
      'Her çıtayı metreyle ölç, kurşun kalemle dört yüzünden işaretle.',
      'Testereyle düzgün kes. Kesim yerini sabit bir yüzeye daya.',
      'Tüm kesim yüzeylerini 120 kum zımpara ile düzelt.',
      '1. kat vernik/ahşap koruyucu sür (her iki çıta tipi için).',
      '2-4 saat kurumasını bekle, 2. katı sür.',
      '2. kat tamamen kuruyana kadar bekle (12-24 saat).',
    ],
    warnings: [
      'B çıtaları 71 cm olmalı, 80 cm değil! (120 - 4.5 - 4.5 = 71)',
      'Vernikleme atlanmamalı — sera nemi korumasız ahşabı 1-2 sezonda çürütür.',
      '20×30mm çıtalardan 3. çıta sıkışık (4×52=208cm). 4 adet almak daha güvenli.',
    ],
  },
  {
    title: 'Alt Çerçeve (Zemin Karesi)',
    icon: '2',
    Diagram: BaseFrameDiagram,
    materials: ['2x A çıtası (120cm)', '2x B çıtası (71cm)', '4x L-braketi', '~16x vida 4x40mm'],
    tools: ['Akülü matkap', 'Matkap ucu 3mm', 'Gönye', 'Metre'],
    instructions: [
      'İki A çıtasını (120cm) yere paralel koy, aralarında 71cm mesafe bırak.',
      'İki B çıtasını (71cm) A çıtalarının arasına yerleştir.',
      'Gönye ile köşenin 90 derece olduğunu kontrol et.',
      'L-braketini köşenin iç tarafına daya.',
      '3mm matkap ucuyla ön delikleri aç (çatlama önlenir).',
      '4x40mm vidalarla braketi sabitle (her kola 2 vida).',
      '4 köşeyi de aynı şekilde yap.',
      'Çapraz kontrol: köşeden köşeye çapraz mesafeleri ölç. İki çapraz eşitse 90 derece tamam.',
    ],
    warnings: [
      'Ön delik açmayı atlama! 45mm ahşapta ön deliksiz vida çakınca çıta çatlar.',
      'Çaprazlar eşit değilse braketleri hafifçe gevşetip ayarla, sonra tekrar sık.',
    ],
  },
  {
    title: 'Dikey Dikmeleri Dik',
    icon: '3',
    Diagram: PostsDiagram,
    materials: ['4x C çıtası (100cm)', '1x D çıtası (100cm)', '10x L-braketi', '~40x vida 4x40mm'],
    tools: ['Akülü matkap', 'Su terazisi / telefon uygulaması', 'İşkence/mengene (varsa)'],
    instructions: [
      'İlk C çıtasını (100cm) alt çerçevenin bir köşesine dik olarak tut.',
      'Varsa birisi tutsun veya işkence ile geçici sabitle.',
      'L-braketini dikme ile alt yatay çıta arasına yerleştir.',
      'Her dikmeye 2 L-braketi koy (dikmenin iki açık yüzüne birer tane).',
      '3mm ön delik + 4x40mm vida ile sabitle.',
      'Telefon terazisi ile dikmenin düşey olduğunu kontrol et.',
      '4 köşe dikmesini aynı şekilde monte et.',
      'Son olarak D çıtasını (100cm) ön A çıtasının tam ortasına (60cm noktası) dik olarak monte et.',
    ],
    warnings: [
      'Her dikmeyi monte ettikten sonra terazile! Eğik dikme sonraki adımlarda sorun yaratır.',
      'Orta dikme (D) tam ortada olmalı — iki kapı kanadı eşit genişlikte olacak.',
    ],
  },
  {
    title: 'Üst Çerçeve (Tavan Karesi)',
    icon: '4',
    Diagram: TopFrameDiagram,
    materials: ['2x A çıtası (120cm)', '2x B çıtası (71cm)', '8x L-braketi', '~32x vida 4x40mm'],
    tools: ['Akülü matkap', 'Merdiven/tabure'],
    instructions: [
      'Arka A çıtasını (120cm) iki arka dikmenin üst ucuna koy, uçları hizala.',
      'L-braketi ile her iki köşeyi sabitle.',
      'Ön A çıtasını (120cm) iki ön dikmenin üstüne koy, aynı şekilde monte et.',
      'Sol B çıtasını (71cm) sol ön-arka dikmeler arasına yerleştir, braketlerle sabitle.',
      'Sağ B çıtasını aynı şekilde monte et.',
      'Orta dikmenin (D) üst ucunu ön üst A çıtasına L-braketi ile bağla.',
      'Yapıyı hafifçe salla — sağlamsa devam, sallanıyorsa zayıf köşelere ek braketi ekle.',
    ],
    warnings: [
      'Üst çerçeveyi monte ederken yapı sallanabilir — dikkatli ol, birisi tutsun.',
    ],
  },
  {
    title: 'Polikarbonat Levha Kesimi',
    icon: '5',
    Diagram: PCCuttingDiagram,
    materials: ['Polikarbonat levha (600x210cm)'],
    tools: ['Maket bıçağı', 'Metal cetvel (min 120cm)', 'Metre', 'Kurşun kalem'],
    instructions: [
      'Levhayı düz zemine yay.',
      'Koruyucu filmi SÖKME — montaj bitene kadar çizilmeyi önler.',
      'Metreyle ölç, film üzerine kurşun kalemle işaretle.',
      'Büyük panelden başla: Arka panel 120x100 cm.',
      'Cetvel boyunca maket bıçağıyla 2-3 kez çiz, sonra kenardan kır.',
      'Sırasıyla: Sol yan (71x100), Sağ yan (71x100), Tavan (120x71).',
      'En son kapı panelleri: 2 adet ~56x95 cm.',
    ],
    warnings: [
      'Kapı panellerini iskeletten 1-2 cm dar kes — açılırken çerçeveye sürtmemeli.',
      'Olukların yönüne dikkat et — dikey panellerde oluklar dikey olsun.',
    ],
  },
  {
    title: 'Panel Montajı (Arka, Yan, Tavan)',
    icon: '6',
    Diagram: PanelMountDiagram,
    materials: ['Kesilen paneller (4 adet)', '~60x vida 4x30mm', '~60x geniş pul (rondela)'],
    tools: ['Akülü matkap', 'Matkap ucu 6mm (PC delikleri)', 'Matkap ucu 3mm (ahşap ön delikleri)'],
    instructions: [
      'Montaj sırası: Arka (1) → Sol yan (2) → Sağ yan (3) → Tavan (4).',
      'Paneli çıtanın dış yüzeyine daya, kenarları hizala.',
      'Paneli geçici olarak bant ile sabitle veya birisi tutsun.',
      '6mm matkap ucuyla polikarbonatta delik aç (vida 4mm, delik 6mm = 2mm genleşme payı!).',
      'Delikleri 20-25 cm aralıkla aç. Panel kenarından en az 2cm içeride kal.',
      'Her deliğe geniş pul + 4x30mm vida ile sabitle.',
      'Vidayı TAM SIKMA — pul levhaya hafifçe bassın ama levhayı eğmesin.',
    ],
    warnings: [
      'PC deliği mutlaka vidadan büyük olmalı (6mm delik, 4mm vida). Yoksa ısıda çatlar!',
      'Vidayı aşırı sıkma — polikarbonat kırılgan.',
      'Tavan panelinde ileride fan deliği açılacak. Şimdilik tam monte et.',
    ],
  },
  {
    title: 'Kapı Yapımı ve Montajı',
    icon: '7',
    Diagram: DoorDiagram,
    materials: [
      '20×30mm çam çıta (~6m — kapı çerçevesi)',
      '2x kapı paneli (~56x95 cm polikarbonat)',
      '4x menteşe (orta boy)',
      '2x mıknatıslı mandal veya sürgü',
      '~4 m yalıtım fitili (D-profil yapışkanlı conta)',
      '~24x vida 4x30mm + geniş pul (PC montajı)',
      '~16x vida 3x30mm (çerçeve köşeleri)',
      'Vida (menteşe + mandal için)',
    ],
    tools: ['Akülü matkap', 'Testere', 'Gönye', 'Makas (conta için)', 'Matkap ucu 6mm', 'Matkap ucu 3mm'],
    instructions: [
      'ÇERÇEVE KESİMİ (20×30mm çıta):',
      '  Her kanat için: 2x dikey 91cm + 2x yatay 52cm.',
      '  Toplam 8 parça kes (2 kanat).',
      '  Kesim yüzeylerini zımparalayıp vernikle.',
      'ÇERÇEVE MONTAJI:',
      '  2 dikey + 2 yatay çıtayı dikdörtgen olarak birleştir.',
      '  Köşelerde 3mm ön delik aç, 3×30mm vida ile sabitle.',
      '  Gönye ile 90° kontrol et. Dış ölçü ~56×95cm olmalı.',
      '  Çerçeve kapı boşluğundan her yandan 1-2mm küçük olsun (sürtmesin).',
      'PC MONTAJI (çerçeveye):',
      '  Polikarbonat paneli çerçevenin bir yüzüne koy.',
      '  6mm delik aç, geniş pul + 4×30mm vida ile 20cm aralıkla sabitle.',
      '  Genleşme kuralı geçerli: 6mm delik, 4mm vida, aşırı sıkma.',
      'MENTEŞE MONTAJI:',
      '  Menteşeyi önce iskelet dikme çıtasına (45×45) vidale.',
      '  Üst menteşe: yukarıdan ~15cm. Alt menteşe: aşağıdan ~15cm.',
      '  Menteşenin diğer kanadını kapı çerçevesinin dikey çıtasına vidale.',
      '  Kapıyı aç-kapa test et. Sürtüyorsa ayarla.',
      '  Sağ kapıyı aynı şekilde sağ dikmeye monte et.',
      'YALITIM FİTİLİ:',
      '  D-profil yapışkanlı contayı şu yüzeylere yapıştır:',
      '  Sol dikme + Sağ dikme iç yüzleri',
      '  Orta dikmenin her iki yüzü',
      '  Alt + Üst çıtanın kapıya bakan yüzleri',
      '  Yapıştırmadan önce yüzeyi kuru bezle temizle.',
      'MANDAL — Her kapının serbest kenarına mıknatıslı mandal monte et.',
    ],
    warnings: [
      'Menteşeyi polikarbonata değil, ahşap çerçeveye bağla!',
      'Çerçeve ölçüsü kapı boşluğundan 2-3mm küçük olmalı — tam eşit yaparsan sürtür.',
      'Conta kalınlığını kontrol et — çok kalın conta kapıyı tam kapattırmaz.',
    ],
  },
  {
    title: 'Son Kontroller',
    icon: '8',
    Diagram: ChecklistDiagram,
    materials: ['Kaydırmaz keçe ayak / silikon (opsiyonel)'],
    tools: [],
    instructions: [
      'Tüm vidaları tek tek kontrol et — gevşeyeni sık (PC vidalarını aşırı sıkma).',
      'İki kapıyı aç-kapa test et. Sürtme varsa menteşeyi ayarla.',
      'Contanın kapı kapandığında hafifçe sıkıştığını kontrol et.',
      'Yapıyı hafifçe salla — sallanıyorsa zayıf köşelere ek L-braketi ekle.',
      'Polikarbonat koruyucu filmini ŞİMDİ sök.',
      'Sera zeminde kayıyorsa alt çıtaların altına kaydırmaz keçe yapıştır.',
      'El ile panel kenarlarını yokla — hava sızıntısı varsa silikon veya ek conta sür.',
    ],
    warnings: [],
  },
]

function StepProgress({ current, total, completed, onJump }) {
  return (
    <div style={s.progress}>
      {Array.from({ length: total }).map((_, i) => (
        <button
          key={i}
          onClick={() => onJump(i)}
          style={{
            ...s.dot,
            background: completed[i] ? '#4caf50' : i === current ? '#fff' : '#1a1a2a',
            color: completed[i] ? '#fff' : i === current ? '#000' : '#556677',
            border: i === current ? '2px solid #4caf50' : completed[i] ? '2px solid #4caf50' : '2px solid #2a2a3a',
          }}
        >
          {completed[i] ? '\u2713' : i + 1}
        </button>
      ))}
    </div>
  )
}

export default function ConstructionGuide() {
  const [step, setStep] = useState(0)
  const [completed, setCompleted] = useState(Array(STEPS.length).fill(false))

  const current = STEPS[step]
  const isLast = step === STEPS.length - 1
  const isFirst = step === 0

  function markComplete() {
    const next = [...completed]
    next[step] = true
    setCompleted(next)
    if (!isLast) setStep(step + 1)
  }

  function goBack() {
    if (!isFirst) setStep(step - 1)
  }

  return (
    <div style={s.container}>
      <div style={s.header}>
        <div style={s.headerTitle}>Sera Yapım Kılavuzu</div>
        <div style={s.headerSub}>Faz 1 — İskelet ve Kaplama ({completed.filter(Boolean).length}/{STEPS.length} tamamlandı)</div>
      </div>

      <StepProgress current={step} total={STEPS.length} completed={completed} onJump={setStep} />

      <div style={s.content}>
        <div style={s.stepHeader}>
          <div style={s.stepBadge}>{step + 1}</div>
          <div>
            <div style={s.stepNum}>Adım {step + 1} / {STEPS.length}</div>
            <div style={s.stepTitle}>{current.title}</div>
          </div>
          {completed[step] && <span style={s.checkBadge} onClick={() => {
            const next = [...completed]; next[step] = false; setCompleted(next)
          }}>Tamamlandı ✕</span>}
        </div>

        {/* Malzeme + Alet */}
        <div style={s.tagsSection}>
          {current.materials.length > 0 && (
            <div style={s.tagGroup}>
              <div style={s.tagLabel}>MALZEME</div>
              <div style={s.tagList}>
                {current.materials.map((m, i) => <span key={i} style={s.tag}>{m}</span>)}
              </div>
            </div>
          )}
          {current.tools.length > 0 && (
            <div style={s.tagGroup}>
              <div style={s.tagLabel}>ALET</div>
              <div style={s.tagList}>
                {current.tools.map((t, i) => <span key={i} style={s.toolTag}>{t}</span>)}
              </div>
            </div>
          )}
        </div>

        {/* SVG Diagram */}
        <div style={s.diagramBox}>
          <current.Diagram />
        </div>

        {/* Talimatlar */}
        <div style={s.instructionSection}>
          <div style={s.sectionLabel}>ADIMLAR</div>
          <div style={s.instructionList}>
            {(() => {
              let stepNum = 0
              return current.instructions.map((inst, i) => {
                if (inst.endsWith(':')) {
                  // Başlık satırı — numarasız
                  return <div key={i} style={{ ...s.instructionItem, ...s.headerItem }}>{inst}</div>
                }
                if (inst.startsWith('  ')) {
                  // Alt madde — bullet
                  return <div key={i} style={{ ...s.instructionItem, ...s.subItem }}>• {inst.trimStart()}</div>
                }
                // Normal adım — numaralı
                stepNum++
                return <div key={i} style={s.instructionItem}><span style={s.stepNumber}>{stepNum}.</span> {inst}</div>
              })
            })()}
          </div>
        </div>

        {/* Uyarilar */}
        {current.warnings.length > 0 && (
          <div style={s.warningSection}>
            <div style={s.warningLabel}>DİKKAT</div>
            {current.warnings.map((w, i) => (
              <div key={i} style={s.warningItem}>
                <span style={s.warningIcon}>!</span>
                {w}
              </div>
            ))}
          </div>
        )}

        <div style={{ height: 20 }} />
      </div>

      <div style={s.nav}>
        <button
          onClick={goBack}
          disabled={isFirst}
          style={{ ...s.navBtn, ...s.navBtnBack, opacity: isFirst ? 0.3 : 1 }}
        >
          Önceki
        </button>
        <button onClick={markComplete} style={{
          ...s.navBtn,
          ...s.navBtnNext,
          background: completed[step] && !isLast ? '#2a6aaa' : completed[step] && isLast ? '#4caf50' : '#2a6aaa',
        }}>
          {completed[step]
            ? (isLast ? 'Faz 1 Tamam!' : 'Sonraki Adım')
            : 'Tamamlandı \u2192 Sonraki'}
        </button>
      </div>
    </div>
  )
}

const s = {
  container: {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    background: '#0a0a14',
    color: '#e0e0e0',
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    overflow: 'hidden',
  },
  header: {
    padding: '14px 20px 6px',
    borderBottom: '1px solid #1e2a3a',
    flexShrink: 0,
  },
  headerTitle: { fontSize: 18, fontWeight: 700, color: '#a0e0a0' },
  headerSub: { fontSize: 12, color: '#556677', marginTop: 2 },
  progress: {
    display: 'flex',
    gap: 8,
    padding: '12px 20px',
    overflowX: 'auto',
    flexShrink: 0,
  },
  dot: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
    flexShrink: 0,
    fontFamily: 'inherit',
    transition: 'all 0.2s',
  },
  content: {
    flex: 1,
    overflow: 'auto',
    padding: '0 20px 20px',
    WebkitOverflowScrolling: 'touch',
  },
  stepHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
    padding: '14px 0 10px',
  },
  stepBadge: {
    width: 44,
    height: 44,
    borderRadius: 12,
    background: '#1a2a3a',
    border: '2px solid #2a4a6a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 20,
    fontWeight: 800,
    color: '#88aacc',
    flexShrink: 0,
  },
  stepNum: { fontSize: 11, color: '#556677', letterSpacing: '0.05em' },
  stepTitle: { fontSize: 20, fontWeight: 700, color: '#fff' },
  checkBadge: {
    marginLeft: 'auto',
    background: '#1a3a1a',
    color: '#4caf50',
    border: '1px solid #2a5a2a',
    borderRadius: 6,
    padding: '4px 10px',
    fontSize: 11,
    fontWeight: 700,
    flexShrink: 0,
    cursor: 'pointer',
  },
  tagsSection: { marginBottom: 14 },
  tagGroup: { marginBottom: 8 },
  tagLabel: {
    fontSize: 10,
    color: '#445566',
    fontWeight: 700,
    letterSpacing: '0.1em',
    marginBottom: 6,
  },
  tagList: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  tag: {
    background: '#1a2a1a',
    color: '#8ac88a',
    border: '1px solid #2a4a2a',
    borderRadius: 6,
    padding: '5px 10px',
    fontSize: 13,
  },
  toolTag: {
    background: '#1a1a2a',
    color: '#8a8ac8',
    border: '1px solid #2a2a4a',
    borderRadius: 6,
    padding: '5px 10px',
    fontSize: 13,
  },
  diagramBox: {
    background: '#0e1a28',
    border: '1px solid #1e2a3a',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    overflow: 'auto',
    WebkitOverflowScrolling: 'touch',
  },
  sectionLabel: {
    fontSize: 10,
    color: '#445566',
    fontWeight: 700,
    letterSpacing: '0.1em',
    marginBottom: 10,
    paddingBottom: 4,
    borderBottom: '1px solid #1e2a3a',
  },
  instructionSection: { marginBottom: 16 },
  instructionList: { paddingLeft: 8, margin: 0 },
  instructionItem: { fontSize: 15, lineHeight: 1.75, color: '#d0d0d0', marginBottom: 4 },
  stepNumber: { color: '#88aacc', fontWeight: 700, marginRight: 4 },
  subItem: { color: '#a0b0c0', fontSize: 14, paddingLeft: 20 },
  headerItem: { fontWeight: 700, color: '#88aacc', marginTop: 12, marginBottom: 2, fontSize: 14, letterSpacing: '0.02em' },
  warningSection: {
    background: '#2a1a0a',
    border: '1px solid #4a3a1a',
    borderRadius: 10,
    padding: 14,
    marginBottom: 16,
  },
  warningLabel: { fontSize: 11, color: '#ffaa00', fontWeight: 700, letterSpacing: '0.1em', marginBottom: 8 },
  warningItem: {
    fontSize: 14,
    color: '#e0c080',
    lineHeight: 1.6,
    marginBottom: 6,
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
  },
  warningIcon: {
    background: '#ff9800',
    color: '#000',
    width: 20,
    height: 20,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    fontWeight: 900,
    flexShrink: 0,
    marginTop: 2,
  },
  nav: {
    display: 'flex',
    gap: 10,
    padding: '12px 20px',
    borderTop: '1px solid #1e2a3a',
    background: '#0a0a14',
    flexShrink: 0,
  },
  navBtn: {
    padding: '14px 20px',
    borderRadius: 10,
    border: 'none',
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
    fontFamily: 'inherit',
    transition: 'all 0.2s',
  },
  navBtnBack: { background: '#1a1a2a', color: '#8888aa', flex: '0 0 110px' },
  navBtnNext: { background: '#2a6aaa', color: '#fff', flex: 1 },
}
