import { GH, HALF } from '../../constants/dimensions'
import ProfileBar from '../primitives/ProfileBar'
import LBracket from '../primitives/LBracket'
import TSlotJoint from '../primitives/TSlotJoint'
import Screw from '../primitives/Screw'

const p = GH.P
const W = GH.W
const D = GH.D
const H = GH.H

// 8 köşe noktası
const corners = [
  [-W / 2, 0,  D / 2],  // alt ön sol
  [ W / 2, 0,  D / 2],  // alt ön sağ
  [-W / 2, 0, -D / 2],  // alt arka sol
  [ W / 2, 0, -D / 2],  // alt arka sağ
  [-W / 2, H,  D / 2],  // üst ön sol
  [ W / 2, H,  D / 2],  // üst ön sağ
  [-W / 2, H, -D / 2],  // üst arka sol
  [ W / 2, H, -D / 2],  // üst arka sağ
]

// Köşede bağlantı bileşeni — rotation ile yönlendir
function CornerJoint({ pos, rotations, type, showScrews }) {
  if (type === 'l-bracket') {
    return rotations.map((rot, i) => (
      <LBracket key={i} position={pos} rotation={rot} showScrews={showScrews} />
    ))
  }
  return rotations.map((rot, i) => (
    <TSlotJoint key={i} position={pos} rotation={rot} />
  ))
}

export default function AluminumFrame({ connection, showScrews }) {
  return (
    <group>
      {/* ── ALT ÇERÇEVE ── */}
      {/* Ön alt bar */}
      <ProfileBar position={[0, 0, D / 2]} size={[W, p, p]} />
      {/* Arka alt bar */}
      <ProfileBar position={[0, 0, -D / 2]} size={[W, p, p]} />
      {/* Sol alt bar */}
      <ProfileBar position={[-W / 2, 0, 0]} size={[p, p, D]} />
      {/* Sağ alt bar */}
      <ProfileBar position={[W / 2, 0, 0]} size={[p, p, D]} />

      {/* ── ÜST ÇERÇEVE ── */}
      {/* Ön üst bar */}
      <ProfileBar position={[0, H, D / 2]} size={[W, p, p]} />
      {/* Arka üst bar */}
      <ProfileBar position={[0, H, -D / 2]} size={[W, p, p]} />
      {/* Sol üst bar */}
      <ProfileBar position={[-W / 2, H, 0]} size={[p, p, D]} />
      {/* Sağ üst bar */}
      <ProfileBar position={[W / 2, H, 0]} size={[p, p, D]} />

      {/* ── DİKEY KÖŞE PROFILLER ── */}
      <ProfileBar position={[-W / 2, H / 2,  D / 2]} size={[p, H, p]} />
      <ProfileBar position={[ W / 2, H / 2,  D / 2]} size={[p, H, p]} />
      <ProfileBar position={[-W / 2, H / 2, -D / 2]} size={[p, H, p]} />
      <ProfileBar position={[ W / 2, H / 2, -D / 2]} size={[p, H, p]} />

      {/* ── KAPI ORTA DİKEY (kapı bölücü) ── */}
      <ProfileBar position={[0, H / 2, D / 2]} size={[p, H, p]} />

      {/* ── KÖŞE BAĞLANTI DETAYLARI ── */}
      {/* Alt ön sol */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[-W / 2 + p, p, D / 2 - p]}
          rotation={[0, 0, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[-W / 2 + p, p, D / 2 - p]} rotation={[0, 0, 0]} />
      )}

      {/* Alt ön sağ */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[W / 2 - p, p, D / 2 - p]}
          rotation={[0, Math.PI / 2, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[W / 2 - p, p, D / 2 - p]} rotation={[0, Math.PI / 2, 0]} />
      )}

      {/* Alt arka sol */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[-W / 2 + p, p, -D / 2 + p]}
          rotation={[0, -Math.PI / 2, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[-W / 2 + p, p, -D / 2 + p]} rotation={[0, -Math.PI / 2, 0]} />
      )}

      {/* Alt arka sağ */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[W / 2 - p, p, -D / 2 + p]}
          rotation={[0, Math.PI, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[W / 2 - p, p, -D / 2 + p]} rotation={[0, Math.PI, 0]} />
      )}

      {/* Üst ön sol */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[-W / 2 + p, H - p, D / 2 - p]}
          rotation={[Math.PI / 2, 0, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[-W / 2 + p, H - p, D / 2 - p]} rotation={[Math.PI / 2, 0, 0]} />
      )}

      {/* Üst ön sağ */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[W / 2 - p, H - p, D / 2 - p]}
          rotation={[Math.PI / 2, Math.PI / 2, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[W / 2 - p, H - p, D / 2 - p]} rotation={[Math.PI / 2, Math.PI / 2, 0]} />
      )}

      {/* Üst arka sol */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[-W / 2 + p, H - p, -D / 2 + p]}
          rotation={[Math.PI / 2, -Math.PI / 2, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[-W / 2 + p, H - p, -D / 2 + p]} rotation={[Math.PI / 2, -Math.PI / 2, 0]} />
      )}

      {/* Üst arka sağ */}
      {connection === 'l-bracket' ? (
        <LBracket
          position={[W / 2 - p, H - p, -D / 2 + p]}
          rotation={[Math.PI / 2, Math.PI, 0]}
          showScrews={showScrews}
        />
      ) : (
        <TSlotJoint position={[W / 2 - p, H - p, -D / 2 + p]} rotation={[Math.PI / 2, Math.PI, 0]} />
      )}

      {/* ── KÖŞE VİDALARI (genel, frame görünür) ── */}
      {showScrews && connection === 't-slot' && corners.map((c, i) => (
        <Screw
          key={i}
          position={[c[0], c[1] + (c[1] === 0 ? 0.02 : -0.02), c[2]]}
          rotation={c[1] === 0 ? [0, 0, 0] : [Math.PI, 0, 0]}
          scale={1.5}
        />
      ))}
    </group>
  )
}
