import { useState } from 'react'
import GreenhouseScene from './components/scene/GreenhouseScene'
import ControlPanel from './components/ui/ControlPanel'
import ConstructionGuide from './components/ui/ConstructionGuide'
import './App.css'

export default function App() {
  const [mode, setMode] = useState('guide') // 'guide' | '3d'
  const [vis, setVis] = useState({
    frame: true,
    panels: true,
    door: true,
    fans: true,
    electronics: true,
    irrigation: true,
    plants: true,
    screws: true,
  })
  const [doorOpen, setDoorOpen] = useState(false)

  function toggle(key) {
    setVis(v => ({ ...v, [key]: !v[key] }))
  }

  return (
    <div className="app">
      <div className="tab-bar">
        <button
          className={`tab ${mode === 'guide' ? 'active' : ''}`}
          onClick={() => setMode('guide')}
        >
          Yapım Kılavuzu
        </button>
        <button
          className={`tab ${mode === '3d' ? 'active' : ''}`}
          onClick={() => setMode('3d')}
        >
          3D Görünüm
        </button>
      </div>

      {mode === 'guide' ? (
        <div className="guide-wrapper">
          <ConstructionGuide />
        </div>
      ) : (
        <div className="scene-layout">
          <ControlPanel
            vis={vis}
            onToggle={toggle}
            doorOpen={doorOpen}
            onDoorToggle={() => setDoorOpen(v => !v)}
          />
          <div className="canvas-wrapper">
            <GreenhouseScene
              vis={vis}
              doorOpen={doorOpen}
              onDoorToggle={() => setDoorOpen(v => !v)}
            />
          </div>
        </div>
      )}
    </div>
  )
}
