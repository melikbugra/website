import { GH } from '../../constants/dimensions'

const W = GH.W
const D = GH.D
const H = GH.H

// PCB bileşeni — yeşil devre kartı
function PCB({ position, rotation = [0, 0, 0], size, label }) {
  return (
    <group position={position} rotation={rotation}>
      <mesh castShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial color="#1a5c2e" roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Altın pin sırası */}
      {Array.from({ length: Math.floor(size[0] / 0.008) }).map((_, i) => (
        <mesh
          key={i}
          position={[
            -size[0] / 2 + 0.006 + i * 0.008,
            size[1] / 2 + 0.002,
            -size[2] / 2 + 0.003,
          ]}
        >
          <boxGeometry args={[0.002, 0.005, 0.002]} />
          <meshStandardMaterial color="#c8a000" metalness={0.9} roughness={0.1} />
        </mesh>
      ))}
    </group>
  )
}

// LED göstergesi
function LED({ position, color = '#ff2020' }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.004, 8, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={2}
        roughness={0.2}
      />
    </mesh>
  )
}

// DHT22 sensör — beyaz dikdörtgen
function DHT22Sensor({ position }) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[0.015, 0.036, 0.007]} />
        <meshStandardMaterial color="#e8e8e8" roughness={0.8} />
      </mesh>
      {/* Delik ızgarası (nem girişi) */}
      {Array.from({ length: 4 }).map((_, i) => (
        <mesh key={i} position={[0, -0.008 + i * 0.005, 0.004]}>
          <boxGeometry args={[0.01, 0.001, 0.001]} />
          <meshStandardMaterial color="#aaa" />
        </mesh>
      ))}
      {/* Pinler */}
      {[-0.004, 0, 0.004].map((x, i) => (
        <mesh key={i} position={[x, -0.022, 0]}>
          <cylinderGeometry args={[0.0008, 0.0008, 0.01, 4]} />
          <meshStandardMaterial color="#c8a000" metalness={0.9} />
        </mesh>
      ))}
      <LED position={[0.005, 0.01, 0.004]} color="#22ff44" />
    </group>
  )
}

// Röle kartı
function RelayCard({ position }) {
  return (
    <group position={position}>
      <PCB size={[0.10, 0.012, 0.075]} position={[0, 0, 0]} />
      {/* 4 röle bloğu */}
      {[0, 1, 2, 3].map(i => (
        <group key={i} position={[-0.035 + i * 0.024, 0.012, 0]}>
          <mesh>
            <boxGeometry args={[0.018, 0.015, 0.028]} />
            <meshStandardMaterial color="#1a3a6a" roughness={0.6} />
          </mesh>
          <LED position={[0, 0.012, 0.012]} color={i < 2 ? '#ff2020' : '#222'} />
        </group>
      ))}
      {/* Vidalı terminal */}
      {[0, 1, 2].map(i => (
        <mesh key={i} position={[0.03, 0.012, -0.025 + i * 0.02]}>
          <boxGeometry args={[0.012, 0.015, 0.012]} />
          <meshStandardMaterial color="#228822" roughness={0.6} />
        </mesh>
      ))}
    </group>
  )
}

// Raspberry Pi
function RaspberryPi({ position }) {
  return (
    <group position={position}>
      <PCB size={[0.085, 0.015, 0.056]} position={[0, 0, 0]} />
      {/* USB portlar */}
      {[0, 1].map(i => (
        <mesh key={i} position={[0.038, 0.015, -0.01 + i * 0.018]}>
          <boxGeometry args={[0.016, 0.016, 0.014]} />
          <meshStandardMaterial color="#333" roughness={0.7} />
        </mesh>
      ))}
      {/* Ethernet port */}
      <mesh position={[0.038, 0.015, -0.025]}>
        <boxGeometry args={[0.016, 0.014, 0.016]} />
        <meshStandardMaterial color="#555" roughness={0.7} />
      </mesh>
      {/* GPIO pinleri */}
      {Array.from({ length: 20 }).map((_, i) => (
        <mesh key={i} position={[-0.038 + i * 0.002, 0.015, -0.025]}>
          <cylinderGeometry args={[0.0008, 0.0008, 0.008, 4]} />
          <meshStandardMaterial color="#c8a000" metalness={0.9} />
        </mesh>
      ))}
      {/* SD kart yuvası */}
      <mesh position={[-0.038, 0.012, 0]}>
        <boxGeometry args={[0.004, 0.01, 0.016]} />
        <meshStandardMaterial color="#888" roughness={0.5} />
      </mesh>
      {/* Güç LED */}
      <LED position={[0, 0.015, 0.025]} color="#ff2020" />
      {/* Aktivite LED */}
      <LED position={[0.008, 0.015, 0.025]} color="#22ff44" />
    </group>
  )
}

// Toprak nem sensörü
function SoilSensor({ position }) {
  return (
    <group position={position}>
      {/* Küçük PCB */}
      <mesh position={[0, 0.025, 0]}>
        <boxGeometry args={[0.018, 0.022, 0.003]} />
        <meshStandardMaterial color="#1a5c2e" roughness={0.7} />
      </mesh>
      {/* İki metal çubuk (toprak elektrotları) */}
      <mesh position={[-0.004, 0, 0]}>
        <cylinderGeometry args={[0.0015, 0.0015, 0.06, 6]} />
        <meshStandardMaterial color="#aaa" metalness={0.85} roughness={0.2} />
      </mesh>
      <mesh position={[0.004, 0, 0]}>
        <cylinderGeometry args={[0.0015, 0.0015, 0.06, 6]} />
        <meshStandardMaterial color="#aaa" metalness={0.85} roughness={0.2} />
      </mesh>
    </group>
  )
}

// 12V Güç adaptörü
function PowerAdapter({ position }) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[0.08, 0.04, 0.04]} />
        <meshStandardMaterial color="#222" roughness={0.8} />
      </mesh>
      {/* Kablo çıkışı */}
      <mesh position={[0.04, 0, 0]}>
        <cylinderGeometry args={[0.004, 0.004, 0.06, 8]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#333" roughness={0.9} />
      </mesh>
    </group>
  )
}

export default function Electronics() {
  // Arka duvar sağ tarafına monte edilmiş
  const backZ = -D / 2 + 0.02

  return (
    <group>
      {/* Raspberry Pi — arka duvar sol orta */}
      <RaspberryPi position={[-W / 2 + 0.1, 0.62, backZ + 0.008]} />

      {/* Röle kartı — arka duvar merkez */}
      <RelayCard position={[0.05, 0.65, backZ + 0.008]} />

      {/* DHT22 — seranın üst ortası */}
      <DHT22Sensor position={[0, H - 0.08, 0]} />

      {/* Toprak nem sensörleri — saksılarda */}
      <SoilSensor position={[-0.35, 0.03, 0.05]} />
      <SoilSensor position={[0.15, 0.03, 0.05]} />

      {/* Güç adaptörü — arka köşe */}
      <PowerAdapter position={[-W / 2 + 0.06, 0.48, backZ + 0.025]} />
    </group>
  )
}
