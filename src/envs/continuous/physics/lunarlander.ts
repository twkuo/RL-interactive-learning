// Simplified LunarLander dynamics (a faithful Box2D port isn't feasible in-browser). A rigid lander
// with gravity + a main engine (thrust along its "up") and two side engines (rotation + a little
// lateral push). Matches Gymnasium's spirit: 8D observation, 4 discrete actions, shaping reward,
// land gently on the pad (+) or crash (−). Origin (0,0) is the center of the landing pad.

export const L_DT = 0.05;
export const L_GRAVITY = 0.5;
export const L_MAIN = 1.3; // main-engine acceleration (> gravity, so it can hover/ascend)
export const L_SIDE_TORQUE = 3.0;
export const L_SIDE_LAT = 0.15;
export const PAD_HALF = 0.25; // landing pad half-width
export const START_Y = 1.3;
export const X_LIMIT = 1.5; // out-of-bounds horizontally
export const LANDER_MAX_STEPS = 300;
// Gentle-landing tolerances.
export const LAND_VX = 0.5;
export const LAND_VY = 0.6;
export const LAND_ANGLE = 0.3;

export type LanderState = [number, number, number, number, number, number]; // x, y, vx, vy, angle, omega

export function lunarReset(rng: { next(): number }): LanderState {
  return [
    (rng.next() - 0.5) * 0.6, // x
    START_Y, // y
    (rng.next() - 0.5) * 0.4, // vx
    -0.1 - rng.next() * 0.2, // vy (gentle downward)
    (rng.next() - 0.5) * 0.2, // angle
    (rng.next() - 0.5) * 0.2, // omega
  ];
}

// One Euler step of the dynamics (no ground handling — the env decides termination).
export function lunarStep(s: LanderState, action: number): LanderState {
  let [x, y, vx, vy, angle, omega] = s;
  let ax = 0;
  let ay = -L_GRAVITY;
  if (action === 2) {
    // main engine: thrust along the lander's up-axis
    ax += -Math.sin(angle) * L_MAIN;
    ay += Math.cos(angle) * L_MAIN;
  }
  if (action === 1) {
    // left engine: rotate clockwise + push right
    omega -= L_SIDE_TORQUE * L_DT;
    ax += L_SIDE_LAT;
  }
  if (action === 3) {
    // right engine: rotate counter-clockwise + push left
    omega += L_SIDE_TORQUE * L_DT;
    ax -= L_SIDE_LAT;
  }
  vx += ax * L_DT;
  vy += ay * L_DT;
  angle += omega * L_DT;
  x += vx * L_DT;
  y += vy * L_DT;
  return [x, y, vx, vy, angle, omega];
}

const legContact = (s: LanderState): number => (s[1] < 0.08 ? 1 : 0);

// 8D observation: [x, y, vx, vy, angle, omega, leg1, leg2].
export function lunarObs(s: LanderState): number[] {
  const leg = legContact(s);
  return [s[0], s[1], s[2], s[3], s[4], s[5], leg, leg];
}

// Potential-based shaping (Gymnasium-style): closer to the pad, slower, more upright, legs down = higher.
export function lunarShaping(s: LanderState): number {
  const [x, y, vx, vy, angle] = s;
  const dist = Math.sqrt(x * x + y * y);
  const speed = Math.sqrt(vx * vx + vy * vy);
  return -100 * dist - 100 * speed - 100 * Math.abs(angle) + 20 * legContact(s);
}
