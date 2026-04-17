// Her adım için SVG şema bileşenleri

const C = {
  wood: '#a0703c',
  woodDark: '#7a5028',
  woodLight: '#c89050',
  pc: '#88ccee',
  pcLight: '#aaddff',
  metal: '#8a8a8a',
  screw: '#666',
  bg: '#0e1a28',
  text: '#88aacc',
  textDim: '#556677',
  accent: '#4caf50',
  warn: '#ff9800',
  seal: '#333',
  grid: '#1a2a3a',
}

function Label({ x, y, children, size = 11, color = C.text, anchor = 'middle' }) {
  return <text x={x} y={y} fill={color} fontSize={size} textAnchor={anchor} fontFamily="system-ui">{children}</text>
}

// ── ADIM 1: Kesim Planı ──
export function CuttingDiagram() {
  const doorFrame = '#6aa06a'
  const doorFrameDark = '#4a804a'
  const bars45 = [
    { label: 'Çıta 1', a: { w: 220, c: C.wood, t: '120cm (A)' }, b: { w: 130, c: C.woodLight, t: '71cm (B)' } },
    { label: 'Çıta 2', a: { w: 220, c: C.wood, t: '120cm (A)' }, b: { w: 130, c: C.woodLight, t: '71cm (B)' } },
    { label: 'Çıta 3', a: { w: 220, c: C.wood, t: '120cm (A)' }, b: { w: 130, c: C.woodLight, t: '71cm (B)' } },
    { label: 'Çıta 4', a: { w: 220, c: C.wood, t: '120cm (A)' }, b: { w: 130, c: C.woodLight, t: '71cm (B)' } },
    { label: 'Çıta 5', a: { w: 183, c: C.woodDark, t: '100cm (C)' }, b: { w: 183, c: C.woodDark, t: '100cm (C)' } },
    { label: 'Çıta 6', a: { w: 183, c: C.woodDark, t: '100cm (C)' }, b: { w: 183, c: '#8a6030', t: '100cm (D)' } },
    { label: 'Çıta 7', a: { w: 183, c: C.woodDark, t: '100cm (C)' }, b: { w: 183, c: C.grid, t: 'fire' } },
  ]
  const bars20 = [
    { label: 'İnce 1', a: { w: 167, c: doorFrame, t: '91cm (E)' }, b: { w: 167, c: doorFrame, t: '91cm (E)' } },
    { label: 'İnce 2', a: { w: 167, c: doorFrame, t: '91cm (E)' }, b: { w: 167, c: doorFrame, t: '91cm (E)' } },
    { label: 'İnce 3', a: { w: 95, c: doorFrameDark, t: '52cm (F)' }, b: { w: 95, c: doorFrameDark, t: '52cm (F)' }, c: { w: 95, c: doorFrameDark, t: '52cm (F)' }, d: { w: 95, c: doorFrameDark, t: '52cm (F)' } },
  ]
  return (
    <svg viewBox="0 0 460 500" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="500" rx="8" fill={C.bg} />
      <Label x={230} y={20} size={13} color="#fff">Kesim Planı</Label>

      {/* 45×45mm bölümü */}
      <Label x={12} y={44} size={10} color={C.warn} anchor="start">45×45mm iskelet (7 × 2m)</Label>
      {bars45.map((bar, i) => {
        const y = 54 + i * 34
        return (
          <g key={i}>
            <Label x={12} y={y + 16} size={9} color={C.textDim} anchor="start">{bar.label}</Label>
            <rect x={70} y={y} width={bar.a.w} height={22} rx={3} fill={bar.a.c} stroke={C.woodDark} strokeWidth={1} />
            <Label x={70 + bar.a.w / 2} y={y + 15} size={9} color="#fff">{bar.a.t}</Label>
            <rect x={70 + bar.a.w + 4} y={y} width={bar.b.w} height={22} rx={3} fill={bar.b.c} stroke={C.woodDark} strokeWidth={1} opacity={bar.b.t === 'fire' ? 0.3 : 1} />
            <Label x={70 + bar.a.w + 4 + bar.b.w / 2} y={y + 15} size={9} color={bar.b.t === 'fire' ? '#666' : '#fff'}>{bar.b.t}</Label>
          </g>
        )
      })}

      {/* 20×30mm bölümü */}
      <Label x={12} y={306} size={10} color={doorFrame} anchor="start">20×30mm kapı çerçevesi (3-4 × 2m)</Label>
      {bars20.map((bar, i) => {
        const y = 316 + i * 34
        if (bar.c) {
          // 4 parçalı satır (F)
          return (
            <g key={`d${i}`}>
              <Label x={12} y={y + 16} size={9} color={C.textDim} anchor="start">{bar.label}</Label>
              {[bar.a, bar.b, bar.c, bar.d].map((p, j) => {
                const px = 70 + j * (p.w + 4)
                return (
                  <g key={j}>
                    <rect x={px} y={y} width={p.w} height={22} rx={3} fill={p.c} stroke={doorFrameDark} strokeWidth={1} />
                    <Label x={px + p.w / 2} y={y + 15} size={8} color="#fff">{p.t}</Label>
                  </g>
                )
              })}
            </g>
          )
        }
        return (
          <g key={`d${i}`}>
            <Label x={12} y={y + 16} size={9} color={C.textDim} anchor="start">{bar.label}</Label>
            <rect x={70} y={y} width={bar.a.w} height={22} rx={3} fill={bar.a.c} stroke={doorFrameDark} strokeWidth={1} />
            <Label x={70 + bar.a.w / 2} y={y + 15} size={9} color="#fff">{bar.a.t}</Label>
            <rect x={70 + bar.a.w + 4} y={y} width={bar.b.w} height={22} rx={3} fill={bar.b.c} stroke={doorFrameDark} strokeWidth={1} />
            <Label x={70 + bar.a.w + 4 + bar.b.w / 2} y={y + 15} size={9} color="#fff">{bar.b.t}</Label>
          </g>
        )
      })}

      {/* Lejant */}
      <g transform="translate(12, 430)">
        <rect x={0} y={0} width={436} height={60} rx={6} fill="#0a1018" stroke={C.grid} />
        <rect x={12} y={10} width={10} height={10} rx={2} fill={C.wood} />
        <Label x={28} y={19} size={8} color={C.textDim} anchor="start">A — Ön/Arka yatay (120cm)</Label>
        <rect x={180} y={10} width={10} height={10} rx={2} fill={C.woodLight} />
        <Label x={196} y={19} size={8} color={C.textDim} anchor="start">B — Sol/Sağ yatay (71cm)</Label>
        <rect x={340} y={10} width={10} height={10} rx={2} fill={C.woodDark} />
        <Label x={356} y={19} size={8} color={C.textDim} anchor="start">C/D — Dikmeler</Label>
        <rect x={12} y={32} width={10} height={10} rx={2} fill={doorFrame} />
        <Label x={28} y={41} size={8} color={C.textDim} anchor="start">E — Kapı çerçeve dikey (91cm)</Label>
        <rect x={220} y={32} width={10} height={10} rx={2} fill={doorFrameDark} />
        <Label x={236} y={41} size={8} color={C.textDim} anchor="start">F — Kapı çerçeve yatay (52cm)</Label>
      </g>
    </svg>
  )
}

// ── ADIM 2: Alt Çerçeve ──
export function BaseFrameDiagram() {
  const ox = 80, oy = 50, w = 280, h = 170, t = 12
  return (
    <svg viewBox="0 0 460 300" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="300" rx="8" fill={C.bg} />
      <Label x={230} y={24} size={13} color="#fff">Alt Çerçeve — Üstten Görünüm</Label>
      {/* Dış ölçü çizgileri */}
      <line x1={ox} y1={oy - 18} x2={ox + w} y2={oy - 18} stroke={C.textDim} strokeWidth={1} markerEnd="url(#arr)" markerStart="url(#arr)" />
      <Label x={ox + w / 2} y={oy - 22} size={10}>120 cm</Label>
      <line x1={ox - 18} y1={oy} x2={ox - 18} y2={oy + h} stroke={C.textDim} strokeWidth={1} />
      <Label x={ox - 30} y={oy + h / 2} size={10}>71 cm</Label>
      {/* A çıtaları (yatay, 120cm) */}
      <rect x={ox} y={oy} width={w} height={t} rx={2} fill={C.wood} stroke={C.woodDark} />
      <rect x={ox} y={oy + h - t} width={w} height={t} rx={2} fill={C.wood} stroke={C.woodDark} />
      {/* B çıtaları (dikey, 71cm, iç) */}
      <rect x={ox} y={oy + t} width={t} height={h - t * 2} rx={2} fill={C.woodLight} stroke={C.woodDark} />
      <rect x={ox + w - t} y={oy + t} width={t} height={h - t * 2} rx={2} fill={C.woodLight} stroke={C.woodDark} />
      {/* L braketleri */}
      {[[ox + t, oy + t], [ox + w - t - 16, oy + t], [ox + t, oy + h - t - 16], [ox + w - t - 16, oy + h - t - 16]].map(([x, y], i) => (
        <g key={i}>
          <rect x={x} y={y} width={16} height={4} rx={1} fill={C.metal} />
          <rect x={x} y={y} width={4} height={16} rx={1} fill={C.metal} />
          <circle cx={x + 10} cy={y + 2} r={2} fill={C.screw} />
          <circle cx={x + 2} cy={y + 10} r={2} fill={C.screw} />
        </g>
      ))}
      {/* Etiketler */}
      <Label x={ox + w / 2} y={oy + 8} size={9} color="#fff">A (120cm)</Label>
      <Label x={ox + 6} y={oy + h / 2 + 3} size={9} color="#fff" anchor="start">B</Label>
      {/* Çapraz kontrol */}
      <line x1={ox + t} y1={oy + t} x2={ox + w - t} y2={oy + h - t} stroke={C.accent} strokeWidth={1} strokeDasharray="4 4" opacity={0.5} />
      <line x1={ox + w - t} y1={oy + t} x2={ox + t} y2={oy + h - t} stroke={C.accent} strokeWidth={1} strokeDasharray="4 4" opacity={0.5} />
      <Label x={ox + w / 2} y={oy + h / 2 + 4} size={9} color={C.accent}>Çaprazlar eşit = 90°</Label>
      {/* Lejant */}
      <rect x={ox} y={260} width={10} height={10} rx={2} fill={C.metal} />
      <Label x={ox + 16} y={268} size={9} color={C.textDim} anchor="start">L-braketi + vida</Label>
    </svg>
  )
}

// ── ADIM 3: Dikmeler ──
export function PostsDiagram() {
  // İzometrik görünüm — 5 dikmenin tamamı görünsün
  return (
    <svg viewBox="0 0 460 340" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="340" rx="8" fill={C.bg} />
      <Label x={230} y={22} size={13} color="#fff">Dikmeler — İzometrik Görünüm (5 adet)</Label>

      <g transform="translate(80, 36)">
        {/* Alt çerçeve (2. adımdan) — soluk */}
        <polygon points="20,190 220,190 280,160 80,160" fill="none" stroke={C.wood} strokeWidth={2.5} opacity={0.4} />
        <Label x={120} y={200} size={8} color={C.textDim}>Alt çerçeve (Adım 2)</Label>

        {/* 4 köşe dikmesi (C) */}
        {[
          { x1: 20, y1: 190, x2: 20, y2: 40, label: 'C1', lx: 6, ly: 34 },    // sol ön
          { x1: 220, y1: 190, x2: 220, y2: 40, label: 'C2', lx: 228, ly: 34 }, // sağ ön
          { x1: 80, y1: 160, x2: 80, y2: 10, label: 'C3', lx: 68, ly: 6 },     // sol arka
          { x1: 280, y1: 160, x2: 280, y2: 10, label: 'C4', lx: 288, ly: 6 },  // sağ arka
        ].map((p, i) => (
          <g key={i}>
            <line x1={p.x1} y1={p.y1} x2={p.x2} y2={p.y2} stroke={C.woodDark} strokeWidth={10} strokeLinecap="round" />
            <line x1={p.x1} y1={p.y1} x2={p.x2} y2={p.y2} stroke={C.woodLight} strokeWidth={6} strokeLinecap="round" opacity={0.3} />
            {/* L-braketi */}
            <rect x={p.x1 + 5} y={p.y1 - 18} width={4} height={16} rx={1} fill={C.metal} />
            <rect x={p.x1 + 5} y={p.y1 - 4} width={16} height={4} rx={1} fill={C.metal} />
            <Label x={p.lx} y={p.ly} size={9} color={C.textDim} anchor="start">{p.label}</Label>
          </g>
        ))}

        {/* Orta dikme (D) — kapı bölücü, vurgulu */}
        <line x1={120} y1={190} x2={120} y2={40} stroke="#8a6030" strokeWidth={10} strokeLinecap="round" />
        <line x1={120} y1={190} x2={120} y2={40} stroke={C.warn} strokeWidth={6} strokeLinecap="round" opacity={0.2} />
        <rect x={125} y={172} width={4} height={16} rx={1} fill={C.metal} />
        <rect x={125} y={186} width={16} height={4} rx={1} fill={C.metal} />
        <Label x={120} y={34} size={10} color={C.warn}>D (orta)</Label>

        {/* Ölçü: yükseklik */}
        <line x1={-10} y1={40} x2={-10} y2={190} stroke={C.textDim} strokeWidth={1} />
        <line x1={-14} y1={40} x2={-6} y2={40} stroke={C.textDim} strokeWidth={1} />
        <line x1={-14} y1={190} x2={-6} y2={190} stroke={C.textDim} strokeWidth={1} />
        <Label x={-18} y={118} size={10} color={C.text} anchor="end">100cm</Label>

        {/* Ölçü: orta dikme konumu */}
        <line x1={20} y1={206} x2={120} y2={206} stroke={C.textDim} strokeWidth={1} />
        <Label x={70} y={220} size={9}>60 cm</Label>
        <line x1={120} y1={206} x2={220} y2={206} stroke={C.textDim} strokeWidth={1} />
        <Label x={170} y={220} size={9}>60 cm</Label>
      </g>

      {/* Terazi notu */}
      <g transform="translate(340, 220)">
        <rect x={0} y={0} width={90} height={56} rx={8} fill="#1a2a1a" stroke={C.accent} strokeWidth={1} />
        <line x1={20} y1={28} x2={70} y2={28} stroke={C.accent} strokeWidth={2} />
        <line x1={45} y1={12} x2={45} y2={44} stroke={C.accent} strokeWidth={2} />
        <Label x={45} y={64} size={8} color={C.accent}>Terazile!</Label>
      </g>

      {/* Lejant */}
      <g transform="translate(20, 300)">
        <line x1={0} y1={6} x2={14} y2={6} stroke={C.woodDark} strokeWidth={6} />
        <Label x={20} y={10} size={9} color={C.textDim} anchor="start">C — Köşe dikmesi (×4)</Label>
        <line x1={160} y1={6} x2={174} y2={6} stroke="#8a6030" strokeWidth={6} />
        <Label x={180} y={10} size={9} color={C.textDim} anchor="start">D — Orta dikme (×1)</Label>
        <rect x={320} y={0} width={10} height={10} rx={2} fill={C.metal} />
        <Label x={336} y={10} size={9} color={C.textDim} anchor="start">L-braketi</Label>
      </g>
    </svg>
  )
}

// ── ADIM 4: Üst Çerçeve ──
export function TopFrameDiagram() {
  // Basit 3D isometrik görünüm
  const s = 0.9
  return (
    <svg viewBox="0 0 460 310" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="310" rx="8" fill={C.bg} />
      <Label x={230} y={22} size={13} color="#fff">Tamamlanmış İskelet</Label>
      <g transform="translate(100, 40) scale(0.85)">
        {/* Alt çerçeve */}
        <polygon points="30,200 230,200 290,170 90,170" fill="none" stroke={C.wood} strokeWidth={3} />
        {/* Üst çerçeve */}
        <polygon points="30,60 230,60 290,30 90,30" fill="none" stroke={C.woodLight} strokeWidth={3} />
        {/* Dikey dikmeler */}
        {[[30, 200, 30, 60], [230, 200, 230, 60], [290, 170, 290, 30], [90, 170, 90, 30]].map(([x1, y1, x2, y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.woodDark} strokeWidth={3} />
        ))}
        {/* Orta dikme (kapı bölücü) */}
        <line x1={130} y1={200} x2={130} y2={60} stroke="#8a6030" strokeWidth={3} strokeDasharray="6 3" />
        {/* Ölçüler */}
        <Label x={130} y={218} size={10}>120 cm</Label>
        <Label x={268} y={104} size={10} anchor="start">100 cm</Label>
        <Label x={190} y={18} size={10}>80 cm</Label>
        {/* Paneller (saydam) */}
        <polygon points="30,200 230,200 230,60 30,60" fill={C.pc} opacity={0.08} />
        <polygon points="290,170 290,30 230,60 230,200" fill={C.pc} opacity={0.06} />
        <polygon points="30,60 230,60 290,30 90,30" fill={C.pc} opacity={0.06} />
        {/* "Ön yüz = kapı" label */}
        <Label x={130} y={140} size={10} color={C.warn}>Ön yüz = Kapı</Label>
      </g>
      {/* Montaj sırası */}
      <g transform="translate(20, 260)">
        {['1. Arka üst A', '2. Ön üst A', '3. Sol B', '4. Sağ B'].map((t, i) => (
          <g key={i}>
            <circle cx={i * 110 + 14} cy={8} r={8} fill={C.accent} opacity={0.8} />
            <Label x={i * 110 + 14} y={12} size={8} color="#fff">{i + 1}</Label>
            <Label x={i * 110 + 28} y={12} size={9} color={C.textDim} anchor="start">{t.slice(3)}</Label>
          </g>
        ))}
      </g>
    </svg>
  )
}

// ── ADIM 5: PC Kesim ──
export function PCCuttingDiagram() {
  return (
    <svg viewBox="0 0 460 260" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="260" rx="8" fill={C.bg} />
      <Label x={230} y={22} size={13} color="#fff">Polikarbonat Kesim Planı (600×210cm)</Label>
      <g transform="translate(15, 36)">
        {/* Ana levha */}
        <rect x={0} y={0} width={430} height={190} rx={4} fill="none" stroke={C.pc} strokeWidth={1} strokeDasharray="4 2" />
        {/* Arka */}
        <rect x={4} y={4} width={130} height={115} rx={3} fill={C.pc} opacity={0.25} stroke={C.pc} strokeWidth={1} />
        <Label x={69} y={55} size={10} color="#fff">Arka</Label>
        <Label x={69} y={70} size={9} color={C.pcLight}>120×100</Label>
        {/* Sol yan */}
        <rect x={138} y={4} width={80} height={115} rx={3} fill={C.pc} opacity={0.2} stroke={C.pc} strokeWidth={1} />
        <Label x={178} y={55} size={10} color="#fff">Sol</Label>
        <Label x={178} y={70} size={9} color={C.pcLight}>71×100</Label>
        {/* Sağ yan */}
        <rect x={222} y={4} width={80} height={115} rx={3} fill={C.pc} opacity={0.2} stroke={C.pc} strokeWidth={1} />
        <Label x={262} y={55} size={10} color="#fff">Sağ</Label>
        <Label x={262} y={70} size={9} color={C.pcLight}>71×100</Label>
        {/* Tavan */}
        <rect x={306} y={4} width={120} height={82} rx={3} fill={C.pc} opacity={0.15} stroke={C.pc} strokeWidth={1} />
        <Label x={366} y={40} size={10} color="#fff">Tavan</Label>
        <Label x={366} y={55} size={9} color={C.pcLight}>120×71</Label>
        {/* Sol kapı */}
        <rect x={4} y={123} width={70} height={63} rx={3} fill={C.warn} opacity={0.2} stroke={C.warn} strokeWidth={1} />
        <Label x={39} y={152} size={9} color={C.warn}>Sol kapı</Label>
        <Label x={39} y={165} size={8} color={C.warn}>56×95</Label>
        {/* Sağ kapı */}
        <rect x={78} y={123} width={70} height={63} rx={3} fill={C.warn} opacity={0.2} stroke={C.warn} strokeWidth={1} />
        <Label x={113} y={152} size={9} color={C.warn}>Sağ kapı</Label>
        <Label x={113} y={165} size={8} color={C.warn}>56×95</Label>
        {/* Fire */}
        <rect x={152} y={123} width={274} height={63} rx={3} fill={C.grid} opacity={0.3} stroke={C.grid} strokeWidth={1} />
        <Label x={289} y={158} size={11} color="#444">fire (yedek parça)</Label>
      </g>
    </svg>
  )
}

// ── ADIM 6: Panel Montajı ──
export function PanelMountDiagram() {
  return (
    <svg viewBox="0 0 460 320" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="320" rx="8" fill={C.bg} />
      <Label x={230} y={22} size={13} color="#fff">Panel Montajı — Sıra ve Vida Detayı</Label>
      {/* Montaj sırası - 3D view */}
      <g transform="translate(20, 40)">
        <g transform="scale(0.65) translate(40, 0)">
          <polygon points="30,180 230,180 290,150 90,150" fill="none" stroke={C.woodDark} strokeWidth={2} />
          <polygon points="30,30 230,30 290,0 90,0" fill="none" stroke={C.woodDark} strokeWidth={2} />
          {[[30, 180, 30, 30], [230, 180, 230, 30], [290, 150, 290, 0], [90, 150, 90, 0]].map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.woodDark} strokeWidth={2} />
          ))}
          {/* Arka panel - 1 */}
          <polygon points="90,150 290,150 290,0 90,0" fill={C.pc} opacity={0.3} stroke={C.accent} strokeWidth={2} />
          <circle cx={190} cy={75} r={16} fill={C.accent} opacity={0.8} />
          <Label x={190} y={79} size={14} color="#fff">1</Label>
          {/* Sol panel - 2 */}
          <polygon points="30,180 90,150 90,0 30,30" fill={C.pc} opacity={0.2} stroke={C.pc} strokeWidth={1.5} />
          <circle cx={60} cy={90} r={12} fill={C.pc} opacity={0.8} />
          <Label x={60} y={94} size={12} color="#fff">2</Label>
          {/* Sağ panel - 3 */}
          <polygon points="230,180 290,150 290,0 230,30" fill={C.pc} opacity={0.2} stroke={C.pc} strokeWidth={1.5} />
          <circle cx={260} cy={90} r={12} fill={C.pc} opacity={0.8} />
          <Label x={260} y={94} size={12} color="#fff">3</Label>
          {/* Tavan - 4 */}
          <polygon points="30,30 230,30 290,0 90,0" fill={C.pc} opacity={0.15} stroke={C.pcLight} strokeWidth={1.5} />
          <circle cx={160} cy={15} r={12} fill={C.pcLight} opacity={0.8} />
          <Label x={160} y={19} size={12} color="#000">4</Label>
        </g>
      </g>
      {/* Vida detayı (kesit) */}
      <g transform="translate(280, 50)">
        <rect x={0} y={0} width={160} height={210} rx={8} fill="#0a1018" stroke={C.grid} />
        <Label x={80} y={20} size={10} color="#fff">Vida Kesit Detayı</Label>
        {/* Vida başı */}
        <rect x={58} y={35} width={44} height={10} rx={3} fill={C.screw} />
        <line x1={72} y1={40} x2={88} y2={40} stroke="#888" strokeWidth={1} />
        <line x1={80} y1={35} x2={80} y2={45} stroke="#888" strokeWidth={1} />
        <Label x={120} y={43} size={8} color={C.textDim} anchor="start">vida</Label>
        {/* Pul */}
        <rect x={48} y={48} width={64} height={6} rx={2} fill={C.metal} />
        <Label x={120} y={54} size={8} color={C.textDim} anchor="start">pul</Label>
        {/* Boşluk göstergesi */}
        <rect x={68} y={57} width={24} height={4} rx={1} fill="none" stroke={C.warn} strokeWidth={1} strokeDasharray="2 1" />
        <Label x={120} y={63} size={8} color={C.warn} anchor="start">2mm boşluk</Label>
        {/* Polikarbonat */}
        <rect x={36} y={63} width={88} height={14} rx={2} fill={C.pc} opacity={0.4} stroke={C.pc} />
        <Label x={120} y={74} size={8} color={C.pcLight} anchor="start">polikarbonat</Label>
        {/* 6mm delik göster */}
        <circle cx={80} cy={70} r={8} fill="none" stroke={C.warn} strokeWidth={1} strokeDasharray="2 1" />
        <Label x={80} y={95} size={8} color={C.warn}>6mm delik</Label>
        {/* Ahşap */}
        <rect x={28} y={80} width={104} height={50} rx={3} fill={C.wood} stroke={C.woodDark} />
        <Label x={80} y={110} size={10} color="#fff">ahşap çıta</Label>
        {/* 3mm ön delik */}
        <circle cx={80} cy={90} r={4} fill="none" stroke="#ccc" strokeWidth={1} />
        <Label x={80} y={145} size={8} color={C.textDim}>3mm ön delik</Label>
        {/* Aralık notu */}
        <Label x={80} y={175} size={9} color={C.accent}>20-25 cm aralıkla</Label>
        <Label x={80} y={190} size={8} color={C.textDim}>Kenarda min 2cm</Label>
      </g>
    </svg>
  )
}

// ── ADIM 7: Kapı ──
export function DoorDiagram() {
  return (
    <svg viewBox="0 0 460 440" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="440" rx="8" fill={C.bg} />
      <Label x={230} y={22} size={13} color="#fff">Kapı Montajı — Önden Görünüm</Label>
      <g transform="translate(40, 35)">
        {/* İskelet ön yüz */}
        {/* Sol dikme (45×45) */}
        <rect x={0} y={0} width={14} height={230} rx={2} fill={C.woodDark} />
        <Label x={7} y={-6} size={8} color={C.textDim}>45×45 dikme</Label>
        {/* Sağ dikme (45×45) */}
        <rect x={290} y={0} width={14} height={230} rx={2} fill={C.woodDark} />
        <Label x={297} y={-6} size={8} color={C.textDim}>45×45 dikme</Label>
        {/* Orta dikme */}
        <rect x={146} y={0} width={14} height={230} rx={2} fill="#8a6030" />
        <Label x={153} y={-6} size={8} color={C.warn}>orta dikme</Label>
        {/* Üst/alt yatay */}
        <rect x={0} y={0} width={304} height={10} rx={2} fill={C.wood} />
        <rect x={0} y={222} width={304} height={10} rx={2} fill={C.wood} />

        {/* Sol kapı — 20×30 çerçeve */}
        <rect x={18} y={14} width={124} height={204} rx={2} fill="none" stroke={C.woodLight} strokeWidth={5} />
        {/* PC levha çerçevenin üzerinde */}
        <rect x={22} y={18} width={116} height={196} rx={2} fill={C.pc} opacity={0.12} stroke={C.pc} strokeWidth={1} strokeDasharray="4 2" />
        {/* PC vida noktaları */}
        {[35, 65, 95, 125, 155, 185].map((vy, i) => (
          <g key={`lv${i}`}>
            <circle cx={26} cy={vy} r={2.5} fill={C.screw} />
            <circle cx={134} cy={vy} r={2.5} fill={C.screw} />
          </g>
        ))}
        {[50, 100].map((vx, i) => (
          <g key={`lh${i}`}>
            <circle cx={vx} cy={20} r={2.5} fill={C.screw} />
            <circle cx={vx} cy={212} r={2.5} fill={C.screw} />
          </g>
        ))}
        <Label x={80} y={100} size={10} color={C.pcLight}>Sol kapı</Label>
        <Label x={80} y={115} size={8} color={C.textDim}>20×30 çerçeve</Label>
        <Label x={80} y={128} size={8} color={C.textDim}>+ PC levha</Label>

        {/* Sağ kapı — 20×30 çerçeve */}
        <rect x={164} y={14} width={124} height={204} rx={2} fill="none" stroke={C.woodLight} strokeWidth={5} />
        <rect x={168} y={18} width={116} height={196} rx={2} fill={C.pc} opacity={0.12} stroke={C.pc} strokeWidth={1} strokeDasharray="4 2" />
        {[35, 65, 95, 125, 155, 185].map((vy, i) => (
          <g key={`rv${i}`}>
            <circle cx={172} cy={vy} r={2.5} fill={C.screw} />
            <circle cx={280} cy={vy} r={2.5} fill={C.screw} />
          </g>
        ))}
        {[200, 250].map((vx, i) => (
          <g key={`rh${i}`}>
            <circle cx={vx} cy={20} r={2.5} fill={C.screw} />
            <circle cx={vx} cy={212} r={2.5} fill={C.screw} />
          </g>
        ))}
        <Label x={226} y={100} size={10} color={C.pcLight}>Sağ kapı</Label>
        <Label x={226} y={115} size={8} color={C.textDim}>20×30 çerçeve</Label>
        <Label x={226} y={128} size={8} color={C.textDim}>+ PC levha</Label>

        {/* Menteşeler — sol (dikme → çerçeve) */}
        {[40, 190].map((my, i) => (
          <g key={`lh${i}`}>
            <rect x={10} y={my} width={18} height={16} rx={2} fill={C.metal} stroke="#999" strokeWidth={1} />
            <circle cx={14} cy={my + 8} r={2.5} fill={C.screw} />
            <circle cx={24} cy={my + 8} r={2.5} fill={C.screw} />
          </g>
        ))}
        {/* Menteşeler — sağ */}
        {[40, 190].map((my, i) => (
          <g key={`rh${i}`}>
            <rect x={278} y={my} width={18} height={16} rx={2} fill={C.metal} stroke="#999" strokeWidth={1} />
            <circle cx={282} cy={my + 8} r={2.5} fill={C.screw} />
            <circle cx={292} cy={my + 8} r={2.5} fill={C.screw} />
          </g>
        ))}

        {/* Mandallar */}
        <rect x={134} y={108} width={10} height={18} rx={2} fill={C.warn} opacity={0.6} />
        <rect x={162} y={108} width={10} height={18} rx={2} fill={C.warn} opacity={0.6} />
        <Label x={153} y={140} size={7} color={C.warn}>mandal</Label>

        {/* Conta şeritleri */}
        <line x1={15} y1={12} x2={15} y2={220} stroke={C.accent} strokeWidth={3} opacity={0.6} />
        <line x1={145} y1={12} x2={145} y2={220} stroke={C.accent} strokeWidth={3} opacity={0.6} />
        <line x1={161} y1={12} x2={161} y2={220} stroke={C.accent} strokeWidth={3} opacity={0.6} />
        <line x1={291} y1={12} x2={291} y2={220} stroke={C.accent} strokeWidth={3} opacity={0.6} />
      </g>

      {/* Kapı kanadı kesit detayı */}
      <g transform="translate(20, 280)">
        <rect x={0} y={0} width={420} height={80} rx={8} fill="#0a1018" stroke={C.grid} />
        <Label x={210} y={18} size={10} color="#fff">Kapı Kanadı Kesiti (yandan)</Label>
        {/* Çerçeve çıtası */}
        <rect x={160} y={32} width={30} height={36} rx={2} fill={C.woodLight} stroke={C.woodDark} />
        <Label x={175} y={54} size={7} color="#fff">20×30</Label>
        {/* PC levha */}
        <rect x={192} y={30} width={4} height={40} rx={1} fill={C.pc} opacity={0.5} stroke={C.pc} />
        <Label x={208} y={46} size={8} color={C.pcLight} anchor="start">PC 4mm</Label>
        {/* Vida + pul */}
        <rect x={190} y={42} width={10} height={3} rx={1} fill={C.metal} />
        <rect x={193} y={38} width={4} height={8} rx={1} fill={C.screw} />
        <Label x={208} y={56} size={7} color={C.textDim} anchor="start">vida+pul</Label>
        {/* Menteşe tarafı */}
        <rect x={130} y={32} width={28} height={36} rx={2} fill={C.metal} opacity={0.6} />
        <Label x={144} y={54} size={7} color={C.textDim}>menteşe</Label>
        {/* Dikme */}
        <rect x={98} y={26} width={30} height={45} rx={2} fill={C.woodDark} stroke={C.woodDark} />
        <Label x={113} y={52} size={7} color="#fff">45×45</Label>
        {/* Ok */}
        <Label x={50} y={50} size={8} color={C.textDim}>dikme → menteşe → çerçeve → PC</Label>
      </g>

      {/* Lejant */}
      <g transform="translate(20, 370)">
        <rect x={0} y={0} width={10} height={10} rx={2} fill={C.woodLight} />
        <Label x={16} y={9} size={9} color={C.textDim} anchor="start">20×30 çerçeve</Label>
        <rect x={110} y={0} width={10} height={10} rx={2} fill={C.metal} />
        <Label x={126} y={9} size={9} color={C.textDim} anchor="start">Menteşe</Label>
        <rect x={190} y={0} width={10} height={10} rx={2} fill={C.warn} opacity={0.6} />
        <Label x={206} y={9} size={9} color={C.textDim} anchor="start">Mandal</Label>
        <line x1={260} y1={5} x2={275} y2={5} stroke={C.accent} strokeWidth={3} opacity={0.6} />
        <Label x={280} y={9} size={9} color={C.textDim} anchor="start">Yalıtım fitili</Label>
        <circle cx={370} cy={5} r={3} fill={C.screw} />
        <Label x={378} y={9} size={9} color={C.textDim} anchor="start">PC vidası</Label>
      </g>

      {/* Yapı notu */}
      <g transform="translate(20, 395)">
        <rect x={0} y={0} width={420} height={32} rx={6} fill="#1a1a0a" stroke="#4a3a1a" />
        <Label x={210} y={14} size={9} color={C.warn}>Menteşe ahşap çerçeveye bağlanır, polikarbonata değil!</Label>
        <Label x={210} y={26} size={8} color={C.textDim}>Çerçeve kapı boşluğundan her yandan 1-2mm küçük olmalı</Label>
      </g>
    </svg>
  )
}

// ── ADIM 8: Kontrol Listesi ──
export function ChecklistDiagram() {
  const checks = [
    { text: 'Tüm vidalar sıkı mı?', sub: '(PC vidaları aşırı sıkı olmasın)' },
    { text: 'Kapılar düzgün açılıp kapanıyor mu?', sub: '' },
    { text: 'Conta sıkışıyor mu kapı kapandığında?', sub: '' },
    { text: 'Yapı sallanıyor mu?', sub: '(Sallanıyorsa → ek braketi)' },
    { text: 'Koruyucu film söküldü mü?', sub: '' },
    { text: 'Zemin kayması kontrol edildi mi?', sub: '(Kayıyorsa → keçe ayak)' },
    { text: 'Hava sızıntısı var mı?', sub: '(Varsa → silikon veya ek conta)' },
  ]
  return (
    <svg viewBox="0 0 460 330" width="100%" style={{ maxWidth: 500 }}>
      <rect width="460" height="330" rx="8" fill={C.bg} />
      <Label x={230} y={24} size={13} color="#fff">Son Kontrol Listesi</Label>
      {checks.map((c, i) => {
        const y = 44 + i * 38
        return (
          <g key={i}>
            <rect x={20} y={y} width={420} height={32} rx={6} fill={i % 2 === 0 ? '#0a1218' : 'transparent'} />
            <rect x={30} y={y + 6} width={20} height={20} rx={4} fill="none" stroke={C.accent} strokeWidth={2} />
            <Label x={62} y={y + 20} size={12} color="#d0d0d0" anchor="start">{c.text}</Label>
            {c.sub && <Label x={62} y={y + 32} size={9} color={C.textDim} anchor="start">{c.sub}</Label>}
          </g>
        )
      })}
      {/* Faz 1 tamamlandı banner */}
      <rect x={40} y={315 - 25} width={380} height={30} rx={8} fill={C.accent} opacity={0.15} stroke={C.accent} />
      <Label x={230} y={315 - 6} size={12} color={C.accent}>FAZ 1 TAMAMLANDI — Sera hazır!</Label>
    </svg>
  )
}
