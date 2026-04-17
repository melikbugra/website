import { useMemo } from 'react'
import { CatmullRomCurve3, Vector3, TubeGeometry } from 'three'
import { GH } from '../../constants/dimensions'

const W = GH.W
const D = GH.D
const H = GH.H

// Hortum — CatmullRom eğrisi üzerinde TubeGeometry
function Hose({ points, radius = 0.006, color = '#333' }) {
  const geometry = useMemo(() => {
    const curve = new CatmullRomCurve3(points.map(p => new Vector3(...p)))
    return new TubeGeometry(curve, points.length * 4, radius, 6, false)
  }, [points, radius])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} roughness={0.8} />
    </mesh>
  )
}

// Damla sulama kazığı
function DripStake({ position }) {
  return (
    <group position={position}>
      {/* Küçük T-birleştirici */}
      <mesh position={[0, 0.025, 0]}>
        <cylinderGeometry args={[0.004, 0.004, 0.01, 8]} />
        <meshStandardMaterial color="#1a5c8a" roughness={0.5} />
      </mesh>
      {/* Yere giren kazık */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.001, 0.05, 6]} />
        <meshStandardMaterial color="#2a7ac0" roughness={0.4} />
      </mesh>
      {/* Su damlası efekti */}
      <mesh position={[0, -0.025, 0]}>
        <sphereGeometry args={[0.004, 8, 8]} />
        <meshStandardMaterial color="#4af" transparent opacity={0.7} roughness={0.1} />
      </mesh>
    </group>
  )
}

// Su deposu
function WaterReservoir({ position }) {
  return (
    <group position={position}>
      {/* Ana depo */}
      <mesh castShadow>
        <boxGeometry args={[0.20, 0.25, 0.15]} />
        <meshStandardMaterial
          color="#2a6a9a"
          transparent
          opacity={0.55}
          roughness={0.2}
          metalness={0.1}
        />
      </mesh>
      {/* Su seviyesi (içeride) */}
      <mesh position={[0, -0.03, 0]}>
        <boxGeometry args={[0.18, 0.18, 0.13]} />
        <meshStandardMaterial
          color="#4af"
          transparent
          opacity={0.3}
          roughness={0.05}
        />
      </mesh>
      {/* Kapak */}
      <mesh position={[0, 0.13, 0]}>
        <boxGeometry args={[0.20, 0.01, 0.15]} />
        <meshStandardMaterial color="#1a4a6a" roughness={0.6} />
      </mesh>
      {/* Etiket */}
      <mesh position={[0, 0, 0.076]}>
        <boxGeometry args={[0.12, 0.06, 0.001]} />
        <meshStandardMaterial color="#fff" roughness={0.9} />
      </mesh>
    </group>
  )
}

// 12V dalgıç pompa
function SubmersiblePump({ position }) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[0.055, 0.04, 0.035]} />
        <meshStandardMaterial color="#1a1a4a" roughness={0.7} />
      </mesh>
      {/* Emme ağzı */}
      <mesh position={[0, -0.022, 0]}>
        <cylinderGeometry args={[0.012, 0.012, 0.006, 8]} />
        <meshStandardMaterial color="#333" roughness={0.6} />
      </mesh>
      {/* Çıkış borusu */}
      <mesh position={[0, 0.03, 0]}>
        <cylinderGeometry args={[0.005, 0.005, 0.04, 8]} />
        <meshStandardMaterial color="#555" roughness={0.5} />
      </mesh>
    </group>
  )
}

export default function IrrigationSystem() {
  // Saksı pozisyonları (Plants bileşeniyle senkron)
  const potPositions = [-0.42, -0.21, 0, 0.21, 0.42]

  // Ana hortum — pompadan başlayıp saksılar üzerinden geçer
  const mainHosePoints = [
    [-W / 2 + 0.12, 0.26, -D / 2 + 0.18],   // pompadan çıkış
    [-W / 2 + 0.12, 0.36, -D / 2 + 0.22],    // yukarı
    [-W / 2 + 0.12, 0.36, 0.1],              // ileri
    [-0.42, 0.36, 0.1],                       // sol saksı
    [-0.21, 0.36, 0.1],
    [0, 0.36, 0.1],
    [0.21, 0.36, 0.1],
    [0.42, 0.36, 0.1],                        // sağ saksı
    [W / 2 - 0.12, 0.36, 0.1],               // bitiş
  ]

  return (
    <group>
      {/* Su deposu — arka sol köşe */}
      <WaterReservoir position={[-W / 2 + 0.12, 0.125, -D / 2 + 0.1]} />

      {/* Pompa — depo içinde */}
      <SubmersiblePump position={[-W / 2 + 0.12, 0.04, -D / 2 + 0.1]} />

      {/* Ana hortum */}
      <Hose points={mainHosePoints} radius={0.005} color="#2a3a2a" />

      {/* Her saksıya inen damla hortumları */}
      {potPositions.map((x, i) => {
        const branch = [
          [x, 0.36, 0.1],
          [x, 0.36, 0.15],
          [x, 0.32, 0.18],
          [x, 0.28, 0.2],
        ]
        return (
          <group key={i}>
            <Hose points={branch} radius={0.003} color="#3a4a3a" />
            <DripStake position={[x, 0.26, 0.2]} />
          </group>
        )
      })}
    </group>
  )
}
