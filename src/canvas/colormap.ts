// Soft numeric color palette (dark blue → teal → soft green), suited to the dark theme and easy on the eyes.
function clamp01(t: number): number {
  return Math.max(0, Math.min(1, t));
}

const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [38, 46, 66]], // #262e42 dark blue-gray
  [0.5, [42, 96, 110]], // #2a606e teal
  [1.0, [111, 199, 151]], // #6fc797 soft green
];

export function valueColor(t: number): string {
  t = clamp01(t);
  let i = 0;
  while (i < STOPS.length - 1 && t > STOPS[i + 1][0]) i++;
  const [t0, c0] = STOPS[i];
  const [t1, c1] = STOPS[Math.min(i + 1, STOPS.length - 1)];
  const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
  return `rgb(${r},${g},${b})`;
}

export function normalize(v: number, min: number, max: number): number {
  if (max - min < 1e-9) return 0.5;
  return (v - min) / (max - min);
}
