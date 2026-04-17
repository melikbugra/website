// M5 Phillips vida — kafa + gövde + çapraz oluk
export default function Screw({ position = [0, 0, 0], rotation = [0, 0, 0], scale = 1 }) {
  const headR = 0.005 * scale
  const shaftR = 0.0025 * scale
  const headH = 0.004 * scale
  const shaftH = 0.022 * scale
  const grooveW = 0.001 * scale
  const grooveD = 0.001 * scale
  const grooveH = headR * 1.8

  return (
    <group position={position} rotation={rotation}>
      {/* Vida kafası */}
      <mesh position={[0, -headH / 2, 0]}>
        <cylinderGeometry args={[headR, headR, headH, 12]} />
        <meshStandardMaterial color="#5a5a5a" metalness={0.9} roughness={0.15} />
      </mesh>

      {/* Phillips çapraz oluk — yatay */}
      <mesh position={[0, -grooveD / 2, 0]}>
        <boxGeometry args={[grooveH, grooveD, grooveW]} />
        <meshStandardMaterial color="#333" metalness={0.8} roughness={0.3} />
      </mesh>
      {/* Phillips çapraz oluk — dikey */}
      <mesh position={[0, -grooveD / 2, 0]}>
        <boxGeometry args={[grooveW, grooveD, grooveH]} />
        <meshStandardMaterial color="#333" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Vida gövdesi */}
      <mesh position={[0, -headH - shaftH / 2, 0]}>
        <cylinderGeometry args={[shaftR, shaftR * 0.8, shaftH, 8]} />
        <meshStandardMaterial color="#707070" metalness={0.85} roughness={0.2} />
      </mesh>
    </group>
  )
}
