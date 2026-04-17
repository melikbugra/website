import Screw from './Screw'

// Metal L-köşe braketi (50×50mm) — ahşap çerçeve köşe bağlantısı
export default function LBracket({ position = [0, 0, 0], rotation = [0, 0, 0], showScrews = true }) {
  const armLen = 0.050
  const armW = 0.040
  const thick = 0.003

  return (
    <group position={position} rotation={rotation}>
      {/* Yatay kol */}
      <mesh position={[armLen / 2 + thick / 2, 0, 0]}>
        <boxGeometry args={[armLen, thick, armW]} />
        <meshStandardMaterial color="#8a8a8a" metalness={0.65} roughness={0.3} />
      </mesh>

      {/* Dikey kol */}
      <mesh position={[0, armLen / 2 + thick / 2, 0]}>
        <boxGeometry args={[thick, armLen, armW]} />
        <meshStandardMaterial color="#8a8a8a" metalness={0.65} roughness={0.3} />
      </mesh>

      {/* Köşe pekiştirme */}
      <mesh position={[thick / 2, thick / 2, 0]}>
        <boxGeometry args={[thick * 2, thick * 2, armW]} />
        <meshStandardMaterial color="#8a8a8a" metalness={0.65} roughness={0.3} />
      </mesh>

      {/* Vida delikleri (yatay kol) */}
      {showScrews && (
        <>
          <WoodScrew position={[0.028, thick / 2 + 0.001, armW * 0.25]} />
          <WoodScrew position={[0.028, thick / 2 + 0.001, -armW * 0.25]} />
          <WoodScrew position={[-thick / 2 - 0.001, 0.028, armW * 0.25]} rotation={[0, 0, -Math.PI / 2]} />
          <WoodScrew position={[-thick / 2 - 0.001, 0.028, -armW * 0.25]} rotation={[0, 0, -Math.PI / 2]} />
        </>
      )}
    </group>
  )
}

// Ahşap vida — Phillips başlı, konik gövde
function WoodScrew({ position = [0, 0, 0], rotation = [0, 0, 0] }) {
  return (
    <group position={position} rotation={rotation}>
      {/* Phillips vida kafası */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.005, 0.005, 0.003, 12]} />
        <meshStandardMaterial color="#c0c0c0" metalness={0.8} roughness={0.2} />
      </mesh>
      {/* Çapraz oluk */}
      <mesh position={[0, 0.002, 0]}>
        <boxGeometry args={[0.008, 0.001, 0.001]} />
        <meshStandardMaterial color="#666" />
      </mesh>
      <mesh position={[0, 0.002, 0]}>
        <boxGeometry args={[0.001, 0.001, 0.008]} />
        <meshStandardMaterial color="#666" />
      </mesh>
      {/* Vida gövdesi (konik uçlu) */}
      <mesh position={[0, -0.015, 0]}>
        <cylinderGeometry args={[0.0025, 0.001, 0.030, 8]} />
        <meshStandardMaterial color="#aaa" metalness={0.7} roughness={0.3} />
      </mesh>
    </group>
  )
}
