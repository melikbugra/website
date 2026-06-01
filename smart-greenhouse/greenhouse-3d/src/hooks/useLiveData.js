import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../api'

/**
 * Backend WebSocket'ine bağlanır, canlı durumu tutar ve kopunca yeniden bağlanır.
 * Döndürür: { connected, online, mode, telemetry, thresholds, command }
 */
export function useLiveData() {
  const [state, setState] = useState({
    connected: false,
    online: false,
    mode: 'auto',
    telemetry: null,
    thresholds: null,
    command: null,
  })
  const wsRef = useRef(null)
  const retryRef = useRef(null)

  useEffect(() => {
    let closed = false

    function connect() {
      const ws = new WebSocket(wsUrl())
      wsRef.current = ws

      ws.onopen = () => setState(s => ({ ...s, connected: true }))

      ws.onmessage = (ev) => {
        let msg
        try { msg = JSON.parse(ev.data) } catch { return }
        setState(s => {
          switch (msg.type) {
            case 'snapshot':
              return {
                ...s,
                online: msg.online ?? s.online,
                mode: msg.mode ?? s.mode,
                telemetry: msg.telemetry ?? s.telemetry,
                thresholds: msg.thresholds ?? s.thresholds,
                command: msg.command ?? s.command,
              }
            case 'command':  return { ...s, command: msg.command }
            case 'mode':     return { ...s, mode: msg.mode }
            case 'status':   return { ...s, online: msg.online }
            case 'thresholds': return { ...s, thresholds: msg.thresholds }
            default: return s
          }
        })
      }

      ws.onclose = () => {
        setState(s => ({ ...s, connected: false }))
        if (!closed) retryRef.current = setTimeout(connect, 3000)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      closed = true
      clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [])

  return state
}
