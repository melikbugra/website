// 3D sahnenin üzerine bindirilen canlı sensör okumaları (dijital ikiz HUD'u).

function Stat({ label, value, unit, accent }) {
  return (
    <div style={s.stat}>
      <div style={s.statLabel}>{label}</div>
      <div style={{ ...s.statValue, color: accent || '#e0e0e0' }}>
        {value}
        {unit && <span style={s.statUnit}>{unit}</span>}
      </div>
    </div>
  )
}

export default function LiveHud({ live }) {
  const t = live.telemetry
  const connected = live.connected
  const online = live.online

  const dotColor = !connected ? '#888' : online ? '#4caf50' : '#ffaa00'
  const statusText = !connected ? 'Sunucuya bağlanılıyor…' : online ? 'Cihaz çevrimiçi' : 'Cihaz çevrimdışı'

  const soilAvg = t?.soil?.length ? (t.soil.reduce((a, b) => a + b, 0) / t.soil.length) : null

  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <span style={{ ...s.dot, background: dotColor }} />
        <span style={s.status}>{statusText}</span>
        <span style={s.mode}>{live.mode === 'auto' ? 'OTO' : 'MANUEL'}</span>
      </div>

      {t ? (
        <div style={s.grid}>
          <Stat label="Sıcaklık" value={fmt(t.temp, 1)} unit="°C" accent={tempColor(t.temp)} />
          <Stat label="Nem" value={fmt(t.humidity, 0)} unit="%" accent="#5fb4ff" />
          <Stat label="Toprak (ort.)" value={fmt(soilAvg, 0)} unit="%" accent={soilColor(soilAvg)} />
          <Stat label="Işık" value={fmt(t.lux, 0)} unit="lx" accent="#ffd166" />
          <Stat label="Fan" value={pct(live.command?.fan_pwm)} unit="%" accent="#a0e0a0" />
          <Stat label="Su deposu" value={t.water_ok == null ? '—' : (t.water_ok ? 'Dolu' : 'BOŞ')}
                accent={t.water_ok === false ? '#ff6b6b' : '#a0e0a0'} />
        </div>
      ) : (
        <div style={s.empty}>Telemetri bekleniyor…</div>
      )}
    </div>
  )
}

function fmt(v, d) { return v == null ? '—' : Number(v).toFixed(d) }
function pct(pwm) { return pwm == null ? '—' : Math.round((pwm / 255) * 100) }
function tempColor(t) { if (t == null) return '#e0e0e0'; return t > 30 ? '#ff6b6b' : t < 16 ? '#5fb4ff' : '#a0e0a0' }
function soilColor(s) { if (s == null) return '#e0e0e0'; return s < 30 ? '#ff9800' : '#8ac88a' }

const s = {
  wrap: {
    position: 'absolute', top: 12, left: 12, zIndex: 5,
    background: 'rgba(10,12,20,0.78)', backdropFilter: 'blur(6px)',
    border: '1px solid #1e2a3a', borderRadius: 12, padding: '10px 12px',
    fontFamily: "'Segoe UI', system-ui, sans-serif", minWidth: 240,
    pointerEvents: 'none', userSelect: 'none',
  },
  header: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 },
  dot: { width: 9, height: 9, borderRadius: '50%', flexShrink: 0 },
  status: { color: '#aab', fontSize: 12, flex: 1 },
  mode: { color: '#88aacc', fontSize: 10, fontWeight: 700, border: '1px solid #2a4a6a', borderRadius: 5, padding: '1px 6px' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 },
  stat: {},
  statLabel: { color: '#667', fontSize: 10, marginBottom: 2 },
  statValue: { fontSize: 18, fontWeight: 700, lineHeight: 1 },
  statUnit: { fontSize: 11, fontWeight: 500, marginLeft: 2, color: '#889' },
  empty: { color: '#667', fontSize: 12, padding: '6px 0' },
}
