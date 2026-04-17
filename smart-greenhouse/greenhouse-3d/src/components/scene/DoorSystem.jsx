import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { GH } from '../../constants/dimensions'
import Screw from '../primitives/Screw'

const W = GH.W
const D = GH.D
const H = GH.H
const p = GH.P
const pt = GH.PT

const DOOR_W = (W - p * 3) / 2  // Her kapı kanadının genişliği (orta profil dahil)
const DOOR_H = H - p * 2

const PANEL_MAT = {
  color: '#cce8ff',
  transparent: true,
  opacity: 0.35,
  roughness: 0.05,
  metalness: 0.0,
  side: 2,
}

const SEAL_MAT = {
  color: '#1a1a1a',
  roughness: 0.8,
}

// Tek kapı kanadı — pivot kendi sol/sağ kenarında
function DoorPanel({ pivotX, openAngle, open, onClick, showScrews, side }) {
  const groupRef = useRef()

  useFrame((_, delta) => {
    if (!groupRef.current) return
    const target = open ? openAngle : 0
    groupRef.current.rotation.y += (target - groupRef.current.rotation.y) * Math.min(1, delta * 5)
  })

  // Panel pivot noktasından uzaklığı — kendi merkezi
  const centerX = side === 'left' ? DOOR_W / 2 : -DOOR_W / 2

  return (
    <group position={[pivotX, 0, D / 2]}>
      <group ref={groupRef}>
        {/* Ana kapı paneli */}
        <mesh
          position={[centerX, H / 2, 0]}
          onClick={onClick}
          castShadow
        >
          <boxGeometry args={[DOOR_W, DOOR_H, pt]} />
          <meshStandardMaterial {...PANEL_MAT} />
        </mesh>

        {/* Yalıtım fitili — iç kenar */}
        <mesh position={[centerX, H / 2, pt / 2 + 0.002]}>
          <boxGeometry args={[DOOR_W, DOOR_H, 0.004]} />
          <meshStandardMaterial {...SEAL_MAT} transparent opacity={0.7} />
        </mesh>

        {/* Kapı kolu */}
        <mesh position={[side === 'left' ? DOOR_W - 0.08 : -DOOR_W + 0.08, H / 2, pt]}>
          <cylinderGeometry args={[0.008, 0.008, 0.1, 8]} />
          <meshStandardMaterial color="#c0c0c0" metalness={0.8} roughness={0.2} />
        </mesh>

        {/* Menteşe üst */}
        {showScrews && (
          <>
            <Screw
              position={[0, H * 0.8, 0.005]}
              rotation={[Math.PI / 2, 0, 0]}
              scale={1.3}
            />
            <Screw
              position={[0, H * 0.2, 0.005]}
              rotation={[Math.PI / 2, 0, 0]}
              scale={1.3}
            />
          </>
        )}
      </group>
    </group>
  )
}

export default function DoorSystem({ open, onToggle, showScrews }) {
  return (
    <group>
      {/* Sol kapı kanadı — pivot sol kenarda (x = -W/2 + p) */}
      <DoorPanel
        pivotX={-W / 2 + p}
        openAngle={-Math.PI * 0.82}
        open={open}
        onClick={onToggle}
        showScrews={showScrews}
        side="left"
      />

      {/* Sağ kapı kanadı — pivot sağ kenarda (x = W/2 - p), ters yön */}
      <DoorPanel
        pivotX={W / 2 - p}
        openAngle={Math.PI * 0.82}
        open={open}
        onClick={onToggle}
        showScrews={showScrews}
        side="right"
      />

      {/* Menteşe bloğu sol */}
      <mesh position={[-W / 2 + p / 2, H * 0.8, D / 2 + 0.005]}>
        <boxGeometry args={[p, 0.025, 0.012]} />
        <meshStandardMaterial color="#999" metalness={0.7} roughness={0.3} />
      </mesh>
      <mesh position={[-W / 2 + p / 2, H * 0.2, D / 2 + 0.005]}>
        <boxGeometry args={[p, 0.025, 0.012]} />
        <meshStandardMaterial color="#999" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* Menteşe bloğu sağ */}
      <mesh position={[W / 2 - p / 2, H * 0.8, D / 2 + 0.005]}>
        <boxGeometry args={[p, 0.025, 0.012]} />
        <meshStandardMaterial color="#999" metalness={0.7} roughness={0.3} />
      </mesh>
      <mesh position={[W / 2 - p / 2, H * 0.2, D / 2 + 0.005]}>
        <boxGeometry args={[p, 0.025, 0.012]} />
        <meshStandardMaterial color="#999" metalness={0.7} roughness={0.3} />
      </mesh>
    </group>
  )
}
