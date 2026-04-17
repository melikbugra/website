import { GH } from '../../constants/dimensions'
import ProfileBar from '../primitives/ProfileBar'
import LBracket from '../primitives/LBracket'

const p = GH.P
const W = GH.W
const D = GH.D
const H = GH.H

// 8 köşe pozisyonu + bracket rotasyonları
const cornerConfigs = [
  // Alt köşeler
  { pos: [-W/2 + p, p, D/2 - p],     rot: [0, 0, 0] },                // alt ön sol
  { pos: [W/2 - p, p, D/2 - p],      rot: [0, Math.PI/2, 0] },        // alt ön sağ
  { pos: [-W/2 + p, p, -D/2 + p],    rot: [0, -Math.PI/2, 0] },       // alt arka sol
  { pos: [W/2 - p, p, -D/2 + p],     rot: [0, Math.PI, 0] },          // alt arka sağ
  // Üst köşeler
  { pos: [-W/2 + p, H - p, D/2 - p],  rot: [Math.PI/2, 0, 0] },       // üst ön sol
  { pos: [W/2 - p, H - p, D/2 - p],   rot: [Math.PI/2, Math.PI/2, 0] },  // üst ön sağ
  { pos: [-W/2 + p, H - p, -D/2 + p], rot: [Math.PI/2, -Math.PI/2, 0] }, // üst arka sol
  { pos: [W/2 - p, H - p, -D/2 + p],  rot: [Math.PI/2, Math.PI, 0] },    // üst arka sağ
]

export default function WoodFrame({ showScrews }) {
  return (
    <group>
      {/* ── ALT ÇERÇEVE (4 çıta) ── */}
      <ProfileBar position={[0, 0, D/2]}  size={[W, p, p]} />   {/* Ön */}
      <ProfileBar position={[0, 0, -D/2]} size={[W, p, p]} />   {/* Arka */}
      <ProfileBar position={[-W/2, 0, 0]} size={[p, p, D]} />   {/* Sol */}
      <ProfileBar position={[W/2, 0, 0]}  size={[p, p, D]} />   {/* Sağ */}

      {/* ── ÜST ÇERÇEVE (4 çıta) ── */}
      <ProfileBar position={[0, H, D/2]}  size={[W, p, p]} />
      <ProfileBar position={[0, H, -D/2]} size={[W, p, p]} />
      <ProfileBar position={[-W/2, H, 0]} size={[p, p, D]} />
      <ProfileBar position={[W/2, H, 0]}  size={[p, p, D]} />

      {/* ── DİKEY KÖŞE DİKMELERİ (4 çıta) ── */}
      <ProfileBar position={[-W/2, H/2, D/2]}  size={[p, H, p]} />
      <ProfileBar position={[W/2, H/2, D/2]}   size={[p, H, p]} />
      <ProfileBar position={[-W/2, H/2, -D/2]} size={[p, H, p]} />
      <ProfileBar position={[W/2, H/2, -D/2]}  size={[p, H, p]} />

      {/* ── KAPI ORTA DİKEY BÖLÜCÜ ── */}
      <ProfileBar position={[0, H/2, D/2]} size={[p, H, p]} />

      {/* ── L-KÖŞE BRAKETLERİ ── */}
      {cornerConfigs.map(({ pos, rot }, i) => (
        <LBracket key={i} position={pos} rotation={rot} showScrews={showScrews} />
      ))}
    </group>
  )
}
