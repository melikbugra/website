import { GH } from '../../constants/dimensions'

const W = GH.W

// Dikdörtgen balkon saksısı
function Pot({ position, width = 0.18, depth = 0.16, height = 0.20 }) {
  return (
    <group position={position}>
      {/* Saksı gövdesi — hafif konik */}
      <mesh castShadow position={[0, height / 2, 0]}>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial color="#b05a30" roughness={0.9} />
      </mesh>
      {/* Toprak üst katmanı */}
      <mesh position={[0, height + 0.005, 0]}>
        <boxGeometry args={[width - 0.01, 0.012, depth - 0.01]} />
        <meshStandardMaterial color="#3d2010" roughness={1.0} />
      </mesh>
      {/* Alt boşaltma deliği */}
      <mesh position={[0, 0.005, 0]}>
        <cylinderGeometry args={[0.01, 0.01, 0.012, 6]} />
        <meshStandardMaterial color="#8a4020" roughness={1.0} />
      </mesh>
    </group>
  )
}

// Bodur çeri domates
function CherryTomato({ position }) {
  return (
    <group position={position}>
      {/* Ana gövde (sap) */}
      <mesh position={[0, 0.18, 0]}>
        <cylinderGeometry args={[0.006, 0.008, 0.36, 6]} />
        <meshStandardMaterial color="#5a8a30" roughness={0.8} />
      </mesh>
      {/* Yaprak kümesi */}
      <mesh position={[0, 0.32, 0]}>
        <sphereGeometry args={[0.07, 8, 6]} />
        <meshStandardMaterial color="#2d7a20" roughness={0.9} />
      </mesh>
      <mesh position={[0.04, 0.28, 0.03]}>
        <sphereGeometry args={[0.05, 8, 6]} />
        <meshStandardMaterial color="#3a8a28" roughness={0.9} />
      </mesh>
      <mesh position={[-0.04, 0.26, -0.02]}>
        <sphereGeometry args={[0.045, 8, 6]} />
        <meshStandardMaterial color="#358025" roughness={0.9} />
      </mesh>
      {/* Kırmızı meyveler */}
      {[
        [0.03, 0.22, 0.05],
        [-0.05, 0.20, 0.02],
        [0.01, 0.18, -0.04],
        [0.06, 0.25, -0.02],
        [-0.02, 0.26, 0.06],
        [0.04, 0.30, 0.04],
      ].map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.012, 8, 8]} />
          <meshStandardMaterial color={i % 2 === 0 ? '#e02020' : '#f04040'} roughness={0.3} />
        </mesh>
      ))}
    </group>
  )
}

// Biber bitkisi
function Pepper({ position }) {
  return (
    <group position={position}>
      {/* Gövde */}
      <mesh position={[0, 0.12, 0]}>
        <cylinderGeometry args={[0.005, 0.007, 0.24, 6]} />
        <meshStandardMaterial color="#4a7a28" roughness={0.8} />
      </mesh>
      {/* Yapraklar — koni şekilli */}
      {[0, 1, 2, 3].map(i => {
        const a = (i / 4) * Math.PI * 2
        return (
          <mesh
            key={i}
            position={[Math.cos(a) * 0.04, 0.15 + i * 0.03, Math.sin(a) * 0.04]}
            rotation={[0.3, a, 0.3]}
          >
            <coneGeometry args={[0.028, 0.06, 4]} />
            <meshStandardMaterial color="#3d8020" roughness={0.8} />
          </mesh>
        )
      })}
      {/* Sarı/kırmızı biberler */}
      {[
        [[0.03, 0.16, 0.04], '#f0c020'],
        [[-0.04, 0.19, -0.02], '#e83020'],
        [[0.01, 0.22, -0.05], '#f0c020'],
      ].map(([pos, color], i) => (
        <mesh key={i} position={pos}>
          <cylinderGeometry args={[0.008, 0.006, 0.055, 5]} />
          <meshStandardMaterial color={color} roughness={0.4} />
        </mesh>
      ))}
    </group>
  )
}

// Salatalık (trellis'e sarılan)
function Cucumber({ position }) {
  return (
    <group position={position}>
      {/* Gövde */}
      <mesh position={[0, 0.22, 0]}>
        <cylinderGeometry args={[0.005, 0.007, 0.44, 6]} />
        <meshStandardMaterial color="#4a8a20" roughness={0.8} />
      </mesh>
      {/* Tırmanma dalları — zig-zag */}
      {[0.1, 0.2, 0.3, 0.4].map((y, i) => (
        <mesh
          key={i}
          position={[(i % 2 === 0 ? 1 : -1) * 0.04, y, 0]}
          rotation={[0, 0, (i % 2 === 0 ? 1 : -1) * 0.8]}
        >
          <cylinderGeometry args={[0.003, 0.003, 0.06, 4]} />
          <meshStandardMaterial color="#3a7818" roughness={0.8} />
        </mesh>
      ))}
      {/* Büyük yuvarlak yapraklar */}
      {[[0.02, 0.25, 0.03], [-0.03, 0.35, -0.02], [0.04, 0.42, 0.01]].map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.04, 6, 4]} />
          <meshStandardMaterial color="#2d7010" roughness={0.9} />
        </mesh>
      ))}
      {/* Salatalık meyvesi */}
      <mesh position={[-0.04, 0.28, 0.02]} rotation={[0.3, 0, 0.5]}>
        <cylinderGeometry args={[0.014, 0.012, 0.09, 8]} />
        <meshStandardMaterial color="#4ca820" roughness={0.6} />
      </mesh>
    </group>
  )
}

// Çilek bitkisi
function Strawberry({ position }) {
  return (
    <group position={position}>
      {/* Yapraklar */}
      {[0, 1, 2, 3, 4].map(i => {
        const a = (i / 5) * Math.PI * 2
        return (
          <mesh
            key={i}
            position={[Math.cos(a) * 0.045, 0.04, Math.sin(a) * 0.045]}
            rotation={[-0.4, a, 0]}
          >
            <boxGeometry args={[0.038, 0.005, 0.055]} />
            <meshStandardMaterial color="#228820" roughness={0.8} />
          </mesh>
        )
      })}
      {/* Çilek meyvesi */}
      {[
        [0.04, 0.02, 0.03],
        [-0.04, 0.02, -0.02],
        [0, 0.01, -0.04],
      ].map((p, i) => (
        <mesh key={i} position={p} rotation={[0, 0, 0.2]}>
          <coneGeometry args={[0.012, 0.025, 8]} />
          <meshStandardMaterial color="#d82020" roughness={0.4} />
        </mesh>
      ))}
      {/* Beyaz çiçek */}
      <mesh position={[0.01, 0.05, 0.02]}>
        <sphereGeometry args={[0.01, 8, 8]} />
        <meshStandardMaterial color="#fffaf0" roughness={0.5} />
      </mesh>
    </group>
  )
}

// Arka trellis ızgarası (salatalık için)
function Trellis({ position }) {
  const cols = 6
  const rows = 5
  const w = 0.16
  const h = 0.55

  return (
    <group position={position}>
      {/* Yatay çizgiler */}
      {Array.from({ length: rows }).map((_, i) => (
        <mesh key={`h${i}`} position={[0, i * (h / (rows - 1)), 0]}>
          <boxGeometry args={[w, 0.002, 0.002]} />
          <meshStandardMaterial color="#8a6a3a" roughness={0.9} />
        </mesh>
      ))}
      {/* Dikey çizgiler */}
      {Array.from({ length: cols }).map((_, i) => (
        <mesh key={`v${i}`} position={[-w / 2 + i * (w / (cols - 1)), h / 2, 0]}>
          <boxGeometry args={[0.002, h, 0.002]} />
          <meshStandardMaterial color="#8a6a3a" roughness={0.9} />
        </mesh>
      ))}
    </group>
  )
}

export default function Plants() {
  // 5 saksı — x boyunca eşit aralıklı
  const pots = [
    { x: -0.42, plant: 'tomato' },
    { x: -0.21, plant: 'pepper' },
    { x: 0,     plant: 'cucumber' },
    { x: 0.21,  plant: 'strawberry' },
    { x: 0.42,  plant: 'tomato' },
  ]

  const potZ = 0.18   // saksıların öne yakın konumu
  const potH = 0.20   // saksı yüksekliği

  return (
    <group>
      {/* Trellis — arka duvar salatalık için */}
      <Trellis position={[0, potH + 0.02, -GH.D / 2 + 0.04]} />

      {pots.map(({ x, plant }, i) => (
        <group key={i} position={[x, 0, potZ]}>
          <Pot position={[0, 0, 0]} />
          {plant === 'tomato' && <CherryTomato position={[0, potH, 0]} />}
          {plant === 'pepper' && <Pepper position={[0, potH, 0]} />}
          {plant === 'cucumber' && <Cucumber position={[0, potH, 0]} />}
          {plant === 'strawberry' && <Strawberry position={[0, potH, 0]} />}
        </group>
      ))}
    </group>
  )
}
