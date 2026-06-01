import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import WoodFrame from './WoodFrame'
import PolycarbonatePanels from './PolycarbonatePanels'
import DoorSystem from './DoorSystem'
import Fans from './Fans'
import Electronics from './Electronics'
import IrrigationSystem from './IrrigationSystem'
import Plants from './Plants'

export default function GreenhouseScene({ vis, doorOpen, onDoorToggle, fanPwm = null }) {
  return (
    <Canvas
      shadows
      camera={{ position: [2.2, 1.6, 2.8], fov: 45 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={['#0d0d1a']} />

      {/* Işıklar */}
      <ambientLight intensity={0.5} />
      <directionalLight
        position={[3, 5, 3]}
        intensity={1.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <directionalLight position={[-2, 3, -2]} intensity={0.3} color="#b0d4ff" />
      <pointLight position={[0, 0.8, 0]} intensity={0.4} color="#fff8e7" distance={3} />

      {/* Kamera kontrolü */}
      <OrbitControls
        target={[0, 0.5, 0]}
        minDistance={0.5}
        maxDistance={8}
        enablePan={true}
      />

      {/* Zemin ızgarası */}
      <Grid
        position={[0, 0, 0]}
        args={[6, 6]}
        cellSize={0.1}
        cellThickness={0.5}
        cellColor="#1a2a3a"
        sectionSize={0.5}
        sectionThickness={1}
        sectionColor="#2a3a4a"
        fadeDistance={5}
        fadeStrength={1}
        infiniteGrid
      />

      {/* Bileşenler */}
      {vis.frame && (
        <WoodFrame showScrews={false} />
      )}
      {vis.panels && <PolycarbonatePanels />}
      {vis.door && (
        <DoorSystem open={doorOpen} onToggle={onDoorToggle} showScrews={false} />
      )}
      {vis.fans && <Fans fanPwm={fanPwm} />}
      {vis.electronics && <Electronics />}
      {vis.irrigation && <IrrigationSystem />}
      {vis.plants && <Plants />}
    </Canvas>
  )
}
