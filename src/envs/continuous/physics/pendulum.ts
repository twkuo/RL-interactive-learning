// Shared Pendulum-v1 physics, matching Gymnasium. θ = 0 is upright (the goal); gravity pulls toward
// the bottom. A single continuous torque is applied. There is no termination — only truncation.

export const PEND_G = 10.0;
export const PEND_M = 1.0;
export const PEND_L = 1.0;
export const PEND_DT = 0.05;
export const MAX_TORQUE = 2.0;
export const MAX_SPEED = 8.0;
export const PENDULUM_MAX_STEPS = 200;

export type PendulumState = [number, number]; // theta (0 = up), thetadot

const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

// Wrap an angle to [-π, π] (used for the reward cost, which is minimized at θ = 0).
export function angleNormalize(x: number): number {
  const tau = 2 * Math.PI;
  return ((((x + Math.PI) % tau) + tau) % tau) - Math.PI;
}

export function pendulumReset(rng: { next(): number }): PendulumState {
  const th = (rng.next() * 2 - 1) * Math.PI; // U(-π, π)
  const thdot = (rng.next() * 2 - 1) * 1; // U(-1, 1)
  return [th, thdot];
}

// One step. Returns the new state, the applied (clamped) torque, and the pre-step angle/velocity
// (the cost is computed from those, matching Gymnasium).
export function pendulumStep(
  s: PendulumState,
  torque: number,
): { state: PendulumState; u: number; th: number; thdot: number } {
  const u = clamp(torque, -MAX_TORQUE, MAX_TORQUE);
  const [th, thdot] = s;
  let newthdot = thdot + ((3 * PEND_G) / (2 * PEND_L) * Math.sin(th) + (3 / (PEND_M * PEND_L * PEND_L)) * u) * PEND_DT;
  newthdot = clamp(newthdot, -MAX_SPEED, MAX_SPEED);
  const newth = th + newthdot * PEND_DT;
  return { state: [newth, newthdot], u, th, thdot };
}

// 3D observation: [cos θ, sin θ, θ̇].
export function pendulumObs(s: PendulumState): number[] {
  return [Math.cos(s[0]), Math.sin(s[0]), s[1]];
}
