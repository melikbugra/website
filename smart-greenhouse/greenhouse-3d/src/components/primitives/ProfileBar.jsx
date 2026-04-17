// 45×45 mm Emprenye Çam Çıta
// Ahşap damar efekti için hafif renk varyasyonlu materyal

const woodMat = {
  color: '#a0703c',
  roughness: 0.85,
  metalness: 0.0,
}

export default function ProfileBar({ position, size, rotation = [0, 0, 0], color }) {
  return (
    <group position={position} rotation={rotation}>
      {/* Ana ahşap gövde */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial {...woodMat} color={color || woodMat.color} />
      </mesh>
      {/* Ahşap damar efekti — ince koyu çizgiler */}
      {Array.from({ length: 3 }).map((_, i) => {
        const offset = (i - 1) * (size[0] > size[2] ? size[2] : size[0]) * 0.2
        const isHorizontal = size[0] > size[2]
        return (
          <mesh
            key={i}
            position={isHorizontal
              ? [0, size[1] / 2 + 0.0003, offset]
              : [offset, size[1] / 2 + 0.0003, 0]
            }
          >
            <boxGeometry args={isHorizontal
              ? [size[0] * 0.95, 0.0005, 0.002]
              : [0.002, 0.0005, size[2] * 0.95]
            } />
            <meshStandardMaterial color="#7a5028" roughness={0.9} transparent opacity={0.5} />
          </mesh>
        )
      })}
    </group>
  )
}
