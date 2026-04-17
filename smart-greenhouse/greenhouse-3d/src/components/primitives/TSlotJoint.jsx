import { GH } from '../../constants/dimensions'

// T-Slot bağlantı detayı — profil içi T-nut + civata
// Sigma profilin kanalına oturan T-somun ve üstten geçen civata gösterilir
export default function TSlotJoint({ position = [0, 0, 0], rotation = [0, 0, 0] }) {
  const p = GH.P

  return (
    <group position={position} rotation={rotation}>
      {/* T-nut (profil kanalı içinde) — yatay profilde */}
      <mesh position={[p * 0.3, 0, 0]}>
        <boxGeometry args={[0.018, 0.006, 0.016]} />
        <meshStandardMaterial color="#c0a020" metalness={0.8} roughness={0.25} />
      </mesh>

      {/* T-nut — dikey profilde */}
      <mesh position={[0, p * 0.3, 0]}>
        <boxGeometry args={[0.006, 0.018, 0.016]} />
        <meshStandardMaterial color="#c0a020" metalness={0.8} roughness={0.25} />
      </mesh>

      {/* Civata gövdesi — dikey profil içinden geçen */}
      <mesh position={[0, p * 0.6, 0]}>
        <cylinderGeometry args={[0.003, 0.003, p * 0.8, 6]} />
        <meshStandardMaterial color="#606060" metalness={0.9} roughness={0.15} />
      </mesh>

      {/* Altıgen civata başı */}
      <mesh position={[0, p * 1.05, 0]}>
        <cylinderGeometry args={[0.006, 0.006, 0.005, 6]} />
        <meshStandardMaterial color="#505050" metalness={0.85} roughness={0.2} />
      </mesh>

      {/* Profil kanal çizgisi (yatay profil) */}
      <mesh position={[p * 0.3, 0, p * 0.5 + 0.001]}>
        <boxGeometry args={[p * 0.6, 0.002, 0.0015]} />
        <meshStandardMaterial color="#666" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Profil kanal çizgisi (dikey profil) */}
      <mesh position={[p * 0.5 + 0.001, p * 0.3, 0]}>
        <boxGeometry args={[0.0015, p * 0.6, 0.002]} />
        <meshStandardMaterial color="#666" metalness={0.5} roughness={0.5} />
      </mesh>
    </group>
  )
}
