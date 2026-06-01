import { useEffect, useState } from 'react'
import { api } from '../../api'

// CANLI KONTROL: mod anahtarı (oto/manuel) + manuel modda aktüatör kontrolleri.
export default function LiveControl({ live }) {
  const cmd = live.command || { fan_pwm: 0, pump: false, humidifier: false, light: false }
  const manual = live.mode === 'manual'
  const disabled = !live.connected || !manual

  // Fan slider'ı için yerel durum (sürükleme akıcı olsun); komut değişince senkronla.
  const [fan, setFan] = useState(cmd.fan_pwm)
  useEffect(() => { setFan(cmd.fan_pwm) }, [cmd.fan_pwm])

  async function setMode(mode) {
    try { await api.setMode(mode) } catch (e) { console.error(e) }
  }
  async function send(patch) {
    try { await api.sendCommand(patch) } catch (e) { console.error(e) }
  }

  return (
    <section style={s.section}>
      <div style={s.sectionTitle}>CANLI KONTROL</div>

      {/* Mod anahtarı */}
      <div style={s.modeRow}>
        {['auto', 'manual'].map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            disabled={!live.connected}
            style={{
              ...s.modeBtn,
              background: live.mode === m ? '#2a6aaa' : '#1a1a2a',
              color: live.mode === m ? '#fff' : '#8888aa',
              opacity: live.connected ? 1 : 0.5,
            }}
          >
            {m === 'auto' ? 'Otomatik' : 'Manuel'}
          </button>
        ))}
      </div>
      {!manual && (
        <div style={s.note}>Otomatik modda aktüatörler eşiklere göre yönetilir.</div>
      )}

      {/* Fan hızı */}
      <div style={{ ...s.ctrl, opacity: disabled ? 0.45 : 1 }}>
        <div style={s.ctrlHead}>
          <span style={s.ctrlLabel}>⊙ Fan</span>
          <span style={s.ctrlVal}>{Math.round((fan / 255) * 100)}%</span>
        </div>
        <input
          type="range" min={0} max={255} value={fan} disabled={disabled}
          onChange={(e) => setFan(Number(e.target.value))}
          onMouseUp={() => send({ fan_pwm: fan })}
          onTouchEnd={() => send({ fan_pwm: fan })}
          style={s.slider}
        />
      </div>

      {/* Aç/kapa yükler */}
      <ToggleBtn icon="💧" label="Pompa" on={cmd.pump} disabled={disabled}
                 onClick={() => send({ pump: !cmd.pump })} />
      <ToggleBtn icon="🌫" label="Nemlendirici" on={cmd.humidifier} disabled={disabled}
                 onClick={() => send({ humidifier: !cmd.humidifier })} />
      <ToggleBtn icon="💡" label="Aydınlatma" on={cmd.light} disabled={disabled}
                 onClick={() => send({ light: !cmd.light })} />
    </section>
  )
}

function ToggleBtn({ icon, label, on, disabled, onClick }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      ...s.toggle,
      opacity: disabled ? 0.45 : 1,
      borderColor: on ? '#4caf50' : '#2a2a3a',
      background: on ? '#16301a' : '#14141f',
    }}>
      <span style={s.toggleIcon}>{icon}</span>
      <span style={{ flex: 1, textAlign: 'left', color: on ? '#cfe8cf' : '#99a' }}>{label}</span>
      <span style={{ ...s.led, background: on ? '#4caf50' : '#444' }} />
    </button>
  )
}

const s = {
  section: { marginBottom: 16 },
  sectionTitle: {
    color: '#445566', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
    marginBottom: 8, paddingBottom: 4, borderBottom: '1px solid #1e2a3a',
  },
  modeRow: { display: 'flex', gap: 6, marginBottom: 6 },
  modeBtn: {
    flex: 1, padding: '7px 0', borderRadius: 7, border: '1px solid #2a2a3a',
    fontSize: 12, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit',
  },
  note: { color: '#556677', fontSize: 10, lineHeight: 1.4, marginBottom: 8 },
  ctrl: { marginBottom: 10 },
  ctrlHead: { display: 'flex', justifyContent: 'space-between', marginBottom: 4 },
  ctrlLabel: { color: '#aab', fontSize: 12 },
  ctrlVal: { color: '#a0e0a0', fontSize: 12, fontWeight: 700 },
  slider: { width: '100%', accentColor: '#4caf50', cursor: 'pointer' },
  toggle: {
    width: '100%', display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 10px', marginBottom: 6, borderRadius: 8, border: '1px solid',
    cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
  },
  toggleIcon: { fontSize: 14, width: 18, textAlign: 'center' },
  led: { width: 10, height: 10, borderRadius: '50%', flexShrink: 0 },
}
