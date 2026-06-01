// Backend adresi: VITE_API_BASE ayarlıysa onu, değilse localhost:8000 kullan.
// Üretimde (nginx proxy manager arkasında aynı origin) VITE_API_BASE'i boş bırakıp
// göreli yol kullanabilirsin.
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export function wsUrl() {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL
  if (API_BASE) return API_BASE.replace(/^http/, 'ws') + '/ws'
  // aynı origin
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}

async function req(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  getState: () => req('/api/state'),
  setMode: (mode) => req('/api/mode', { method: 'PUT', body: JSON.stringify({ mode }) }),
  sendCommand: (cmd) => req('/api/command', { method: 'POST', body: JSON.stringify(cmd) }),
  setThresholds: (th) => req('/api/thresholds', { method: 'PUT', body: JSON.stringify(th) }),
}
