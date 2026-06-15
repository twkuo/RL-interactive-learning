// Shared Acrobot-v1 physics, matching Gymnasium (book formulation, RK4 integration).
// Two-link pendulum; torque is applied at the second joint. Goal: swing the tip above the bar.

export const LINK_LENGTH_1 = 1;
export const LINK_LENGTH_2 = 1;
const M1 = 1;
const M2 = 1;
const L1 = 1; // length of link 1
const LC1 = 0.5; // center of mass of link 1
const LC2 = 0.5;
const I1 = 1; // moment of inertia
const I2 = 1;
const G = 9.8;
export const ACROBOT_DT = 0.2;
export const MAX_VEL_1 = 4 * Math.PI;
export const MAX_VEL_2 = 9 * Math.PI;
export const ACROBOT_MAX_STEPS = 500;
export const AVAIL_TORQUE = [-1, 0, 1];

export type AcrobotState = [number, number, number, number]; // theta1, theta2, dtheta1, dtheta2

export function acrobotReset(rng: { next(): number }): AcrobotState {
  const u = () => (rng.next() - 0.5) * 0.2; // U(-0.1, 0.1)
  return [u(), u(), u(), u()];
}

// Continuous-time dynamics: returns [dθ1, dθ2, ddθ1, ddθ2] for a constant torque.
function dsdt(s: AcrobotState, torque: number): AcrobotState {
  const [th1, th2, dth1, dth2] = s;
  const d1 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2 * L1 * LC2 * Math.cos(th2)) + I1 + I2;
  const d2 = M2 * (LC2 * LC2 + L1 * LC2 * Math.cos(th2)) + I2;
  const phi2 = M2 * LC2 * G * Math.cos(th1 + th2 - Math.PI / 2);
  const phi1 =
    -M2 * L1 * LC2 * dth2 * dth2 * Math.sin(th2) -
    2 * M2 * L1 * LC2 * dth2 * dth1 * Math.sin(th2) +
    (M1 * LC1 + M2 * L1) * G * Math.cos(th1 - Math.PI / 2) +
    phi2;
  const ddth2 =
    (torque + (d2 / d1) * phi1 - M2 * L1 * LC2 * dth1 * dth1 * Math.sin(th2) - phi2) /
    (M2 * LC2 * LC2 + I2 - (d2 * d2) / d1);
  const ddth1 = -(d2 * ddth2 + phi1) / d1;
  return [dth1, dth2, ddth1, ddth2];
}

const add = (a: AcrobotState, b: AcrobotState, k: number): AcrobotState => [
  a[0] + k * b[0],
  a[1] + k * b[1],
  a[2] + k * b[2],
  a[3] + k * b[3],
];

// One RK4 step over dt with constant torque (Gymnasium integrates the augmented state in one interval).
function rk4(s: AcrobotState, torque: number, dt: number): AcrobotState {
  const k1 = dsdt(s, torque);
  const k2 = dsdt(add(s, k1, dt / 2), torque);
  const k3 = dsdt(add(s, k2, dt / 2), torque);
  const k4 = dsdt(add(s, k3, dt), torque);
  return [
    s[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    s[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    s[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    s[3] + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
  ];
}

function wrap(x: number): number {
  const tau = 2 * Math.PI;
  let r = (x + Math.PI) % tau;
  if (r < 0) r += tau;
  return r - Math.PI;
}

const bound = (x: number, m: number): number => Math.max(-m, Math.min(m, x));

export function acrobotStep(s: AcrobotState, action: number): AcrobotState {
  const torque = AVAIL_TORQUE[action];
  const ns = rk4(s, torque, ACROBOT_DT);
  return [wrap(ns[0]), wrap(ns[1]), bound(ns[2], MAX_VEL_1), bound(ns[3], MAX_VEL_2)];
}

// Terminated when the tip rises above the bar: -cos(θ1) - cos(θ1+θ2) > 1.
export function acrobotTerminated(s: AcrobotState): boolean {
  return -Math.cos(s[0]) - Math.cos(s[0] + s[1]) > 1;
}

// 6D observation: [cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2].
export function acrobotObs(s: AcrobotState): number[] {
  return [Math.cos(s[0]), Math.sin(s[0]), Math.cos(s[1]), Math.sin(s[1]), s[2], s[3]];
}
