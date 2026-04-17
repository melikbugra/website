// Tüm ölçüler metre cinsinden (Three.js birim = 1 metre)
// Gerçek ölçüler: 120 x 80 x 100 cm

export const GH = {
  W: 1.2,      // Genişlik (x) — 120 cm
  D: 0.8,      // Derinlik (z) — 80 cm
  H: 1.0,      // Yükseklik (y) — 100 cm
  P: 0.045,    // Ahşap çıta kesit boyutu — 45×45 mm
  PT: 0.009,   // Panel kalınlığı (4mm → 9mm, görsellik için büyütüldü)
  FAN: 0.12,   // Fan çapı — 120mm
  SD: 0.008,   // Vida/civata çapı
  SL: 0.040,   // Vida uzunluğu (4×40mm ahşap vida)
}

// Yerleşim kolaylığı için türetilen değerler
export const HALF = {
  W: GH.W / 2,
  D: GH.D / 2,
  H: GH.H / 2,
  P: GH.P / 2,
}
