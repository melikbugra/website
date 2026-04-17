import { GH } from '../../constants/dimensions'

const W = GH.W
const D = GH.D
const H = GH.H
const p = GH.P
const pt = GH.PT

// Oluklu polikarbonat efekti için yatay çizgiler
function CorrugatLines({ width, height, spacing = 0.025 }) {
  const lines = []
  const count = Math.floor(height / spacing)
  for (let i = 0; i <= count; i++) {
    const y = -height / 2 + i * spacing
    lines.push(
      <mesh key={i} position={[0, y, 0.001]}>
        <boxGeometry args={[width - 0.01, 0.002, 0.001]} />
        <meshStandardMaterial color="#c8e0f0" transparent opacity={0.4} />
      </mesh>
    )
  }
  return <>{lines}</>
}

function Panel({ position, rotation, width, height, opacity = 0.32 }) {
  return (
    <group position={position} rotation={rotation}>
      <mesh receiveShadow>
        <boxGeometry args={[width, height, pt]} />
        <meshStandardMaterial
          color="#d6eeff"
          transparent
          opacity={opacity}
          roughness={0.05}
          metalness={0.0}
          side={2}
        />
      </mesh>
      <CorrugatLines width={width} height={height} />
    </group>
  )
}

export default function PolycarbonatePanels() {
  return (
    <group>
      {/* Arka panel (120 × 100 cm) */}
      <Panel
        position={[0, H / 2, -D / 2]}
        rotation={[0, 0, 0]}
        width={W - p * 2}
        height={H - p * 2}
      />

      {/* Sol yan panel (80 × 100 cm) */}
      <Panel
        position={[-W / 2, H / 2, 0]}
        rotation={[0, Math.PI / 2, 0]}
        width={D - p * 2}
        height={H - p * 2}
      />

      {/* Sağ yan panel (80 × 100 cm) */}
      <Panel
        position={[W / 2, H / 2, 0]}
        rotation={[0, Math.PI / 2, 0]}
        width={D - p * 2}
        height={H - p * 2}
      />

      {/* Tavan (120 × 80 cm) */}
      <Panel
        position={[0, H, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        width={W - p * 2}
        height={D - p * 2}
        opacity={0.25}
      />
    </group>
  )
}
