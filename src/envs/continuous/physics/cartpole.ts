// Shared CartPole-v1 physics (explicit Euler), matching Gymnasium 1:1.
// Used by both the tabular CartPole (discretized) and the continuous CartPoleVec (raw vector),
// so the two environments are guaranteed to share identical dynamics.

export const GRAVITY = 9.8;
export const MASSCART = 1.0;
export const MASSPOLE = 0.1;
export const TOTAL_MASS = MASSCART + MASSPOLE;
export const LENGTH = 0.5; // half pole length
export const POLEMASS_LENGTH = MASSPOLE * LENGTH;
export const FORCE_MAG = 10.0;
export const TAU = 0.02; // seconds between state updates
export const THETA_THRESHOLD = (12 * 2 * Math.PI) / 360; // ±12°
export const X_THRESHOLD = 2.4;
export const CARTPOLE_MAX_STEPS = 500;

export type CartPoleState = [number, number, number, number]; // x, xDot, theta, thetaDot

// Initial state: each component drawn from U(-0.05, 0.05).
export function cartpoleReset(rng: { next(): number }): CartPoleState {
  const u = () => (rng.next() - 0.5) * 0.1;
  return [u(), u(), u(), u()];
}

// One explicit-Euler step (position updated using the old velocity first).
export function cartpoleStep(s: CartPoleState, action: number): CartPoleState {
  let [x, xDot, theta, thetaDot] = s;
  const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const temp = (force + POLEMASS_LENGTH * thetaDot * thetaDot * sinT) / TOTAL_MASS;
  const thetaAcc =
    (GRAVITY * sinT - cosT * temp) / (LENGTH * (4.0 / 3.0 - (MASSPOLE * cosT * cosT) / TOTAL_MASS));
  const xAcc = temp - (POLEMASS_LENGTH * thetaAcc * cosT) / TOTAL_MASS;
  x += TAU * xDot;
  xDot += TAU * xAcc;
  theta += TAU * thetaDot;
  thetaDot += TAU * thetaAcc;
  return [x, xDot, theta, thetaDot];
}

// Natural termination: cart out of bounds or pole past the angle threshold.
export function cartpoleTerminated(s: CartPoleState): boolean {
  const [x, , theta] = s;
  return x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD;
}
