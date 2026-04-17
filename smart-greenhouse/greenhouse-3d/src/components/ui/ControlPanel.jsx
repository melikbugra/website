const TOGGLES = [
  { key: 'frame',       label: 'Ahşap İskelet',        icon: '▦' },
  { key: 'panels',      label: 'Polikarbonat Panel',   icon: '◻' },
  { key: 'door',        label: 'Çift Kapı',            icon: '🚪' },
  { key: 'fans',        label: 'Fanlar (3×120mm)',      icon: '⊙' },
  { key: 'electronics', label: 'Elektronik (RPi...)',   icon: '⎇' },
  { key: 'irrigation',  label: 'Sulama Sistemi',        icon: '💧' },
  { key: 'plants',      label: 'Bitkiler & Saksılar',  icon: '🌿' },
  { key: 'screws',      label: 'Vidalar & Civatalar',  icon: '⚙' },
]

export default function ControlPanel({
  vis, onToggle,
  doorOpen, onDoorToggle,
}) {
  return (
    <aside style={styles.panel}>
      <div style={styles.title}>Sera 3D Planlayıcı</div>
      <div style={styles.subtitle}>120 × 80 × 100 cm</div>

      {/* Bileşen toggle'ları */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>BİLEŞENLER</div>
        {TOGGLES.map(({ key, label, icon }) => (
          <label key={key} style={styles.toggleRow} onClick={() => onToggle(key)}>
            <span style={styles.icon}>{icon}</span>
            <span style={{ flex: 1, color: vis[key] ? '#e0e0e0' : '#666' }}>{label}</span>
            <span style={{
              ...styles.checkbox,
              background: vis[key] ? '#4caf50' : '#333',
              borderColor: vis[key] ? '#4caf50' : '#555',
            }}>
              {vis[key] ? '✓' : ''}
            </span>
          </label>
        ))}
      </section>

      {/* Bağlantı bilgisi */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>BAĞLANTI</div>
        <div style={{
          ...styles.radioCard,
          borderColor: '#4caf50',
          background: '#1a2a1a',
        }}>
          <div style={{ color: '#e0e0e0', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
            L-Köşe Braketi + Ahşap Vida
          </div>
          <div style={styles.radioDesc}>
            50×50mm metal L-braketi, 4×40mm çinko kaplama Phillips vida. Her köşede 2 braketi.
          </div>
        </div>
      </section>

      {/* Kapı kontrolü */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>KAPI</div>
        <button style={styles.doorBtn} onClick={onDoorToggle}>
          {doorOpen ? '🚪  Kapıyı Kapat' : '🚪  Kapıyı Aç'}
        </button>
        <div style={styles.hint}>veya kapıya tıkla</div>
      </section>

      {/* Kullanım ipuçları */}
      <section style={{ ...styles.section, marginTop: 'auto' }}>
        <div style={styles.sectionTitle}>KONTROLLER</div>
        <div style={styles.tip}>🖱 Sol tuş — döndür</div>
        <div style={styles.tip}>🖱 Sağ tuş — kaydır</div>
        <div style={styles.tip}>⚲ Tekerlek — zoom</div>
      </section>
    </aside>
  )
}

const styles = {
  panel: {
    width: 220,
    minWidth: 220,
    height: '100vh',
    background: '#0e1220',
    borderRight: '1px solid #1e2a3a',
    display: 'flex',
    flexDirection: 'column',
    padding: '16px 12px',
    gap: 4,
    overflowY: 'auto',
    userSelect: 'none',
    fontFamily: "'Segoe UI', system-ui, sans-serif",
  },
  title: {
    color: '#a0e0a0',
    fontSize: 14,
    fontWeight: 700,
    letterSpacing: '0.04em',
    marginBottom: 2,
  },
  subtitle: {
    color: '#556677',
    fontSize: 11,
    marginBottom: 12,
  },
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    color: '#445566',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '0.1em',
    marginBottom: 8,
    paddingBottom: 4,
    borderBottom: '1px solid #1e2a3a',
  },
  toggleRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '5px 4px',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 12,
    transition: 'background 0.15s',
    marginBottom: 2,
  },
  icon: {
    fontSize: 13,
    width: 18,
    textAlign: 'center',
    color: '#778899',
  },
  checkbox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    border: '1px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 11,
    color: '#fff',
    flexShrink: 0,
    transition: 'all 0.15s',
  },
  radioCard: {
    display: 'block',
    border: '1px solid',
    borderRadius: 8,
    padding: '8px 10px',
    marginBottom: 8,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  radioRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  radio: {
    width: 14,
    height: 14,
    borderRadius: '50%',
    border: '2px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  radioDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: '#4caf50',
  },
  badge: {
    fontSize: 9,
    background: '#1a3a1a',
    color: '#4caf50',
    border: '1px solid #2a5a2a',
    borderRadius: 4,
    padding: '1px 5px',
    marginLeft: 'auto',
    fontWeight: 700,
    letterSpacing: '0.05em',
  },
  radioDesc: {
    color: '#556677',
    fontSize: 11,
    lineHeight: 1.4,
  },
  doorBtn: {
    width: '100%',
    padding: '8px 12px',
    background: '#1a2a3a',
    border: '1px solid #2a4a6a',
    borderRadius: 8,
    color: '#a0c8e0',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontFamily: 'inherit',
  },
  hint: {
    color: '#445566',
    fontSize: 10,
    textAlign: 'center',
    marginTop: 4,
  },
  tip: {
    color: '#445566',
    fontSize: 11,
    marginBottom: 3,
    paddingLeft: 4,
  },
}
