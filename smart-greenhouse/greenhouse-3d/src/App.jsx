import { useState } from 'react'
import GreenhouseScene from './components/scene/GreenhouseScene'
import ControlPanel from './components/ui/ControlPanel'
import LiveHud from './components/ui/LiveHud'
import { useLiveData } from './hooks/useLiveData'
import './App.css'

export default function App() {
  const [vis, setVis] = useState({
    frame: true,
    panels: true,
    door: true,
    fans: true,
    electronics: true,
    irrigation: true,
    plants: true,
  })
  const [doorOpen, setDoorOpen] = useState(false)

  const live = useLiveData()

  function toggle(key) {
    setVis(v => ({ ...v, [key]: !v[key] }))
  }

  return (
    <div className="app">
      <div className="scene-layout">
        <ControlPanel
          vis={vis}
          onToggle={toggle}
          doorOpen={doorOpen}
          onDoorToggle={() => setDoorOpen(v => !v)}
          live={live}
        />
        <div className="canvas-wrapper">
          <LiveHud live={live} />
          <GreenhouseScene
            vis={vis}
            doorOpen={doorOpen}
            onDoorToggle={() => setDoorOpen(v => !v)}
            fanPwm={live.command?.fan_pwm ?? null}
          />
        </div>
      </div>
    </div>
  )
}
