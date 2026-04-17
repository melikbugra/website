import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { GH } from '../../constants/dimensions'

const W = GH.W
const D = GH.D
const H = GH.H

const FAN_SIZE = GH.FAN
const BLADE_COUNT = 7

function FanBlade({ index, total, radius }) {
  const angle = (index / total) * Math.PI * 2
  const x = Math.cos(angle) * radius * 0.55
  const y = Math.sin(angle) * radius * 0.55
  const bladeAngle = angle + Math.PI / 4

  return (
    <mesh
      position={[x, y, 0]}
      rotation={[0, 0, bladeAngle]}
    >
      <planeGeometry args={[radius * 0.7, radius * 0.28]} />
      <meshStandardMaterial
        color="#1a1a1a"
        roughness={0.6}
        metalness={0.3}
        side={2}
      />
    </mesh>
  )
}

function Fan120mm({ position, rotation = [0, 0, 0], speed = 8 }) {
  const bladesRef = useRef()

  useFrame((_, delta) => {
    if (bladesRef.current) {
      bladesRef.current.rotation.z += delta * speed
    }
  })

  const r = FAN_SIZE / 2
  const frameThick = 0.025

  return (
    <group position={position} rotation={rotation}>
      {/* Kare çerçeve */}
      <mesh castShadow>
        <boxGeometry args={[FAN_SIZE + 0.01, FAN_SIZE + 0.01, frameThick]} />
        <meshStandardMaterial color="#111" roughness={0.7} metalness={0.2} />
      </mesh>

      {/* Fan ızgarası (dairesel çerçeve) */}
      <mesh position={[0, 0, frameThick / 2 + 0.001]}>
        <torusGeometry args={[r - 0.005, 0.005, 8, 24]} />
        <meshStandardMaterial color="#222" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Dönen kanatlar */}
      <group ref={bladesRef} position={[0, 0, frameThick / 2 + 0.003]}>
        {Array.from({ length: BLADE_COUNT }).map((_, i) => (
          <FanBlade key={i} index={i} total={BLADE_COUNT} radius={r - 0.01} />
        ))}
        {/* Merkez hub */}
        <mesh>
          <cylinderGeometry args={[0.018, 0.018, 0.01, 12]} />
          <meshStandardMaterial color="#333" metalness={0.8} roughness={0.2} />
        </mesh>
      </group>

      {/* Fan köşe vidaları */}
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sy], i) => (
        <mesh
          key={i}
          position={[sx * (FAN_SIZE / 2 - 0.01), sy * (FAN_SIZE / 2 - 0.01), frameThick / 2 + 0.001]}
        >
          <cylinderGeometry args={[0.003, 0.003, 0.003, 6]} />
          <meshStandardMaterial color="#555" metalness={0.8} roughness={0.2} />
        </mesh>
      ))}
    </group>
  )
}

export default function Fans() {
  return (
    <group>
      {/* Egzoz fan 1 — tavanda sol, aşağı bakıyor */}
      <Fan120mm
        position={[-W / 4, H - 0.013, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        speed={9}
      />
      {/* Egzoz fan 2 — tavanda sağ */}
      <Fan120mm
        position={[W / 4, H - 0.013, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        speed={9}
      />
      {/* Intake fan — arka alt duvar */}
      <Fan120mm
        position={[0, 0.2, -D / 2 + 0.013]}
        rotation={[0, 0, 0]}
        speed={7}
      />
    </group>
  )
}
