import { describe, it, expect } from 'vitest';
import { RNG } from '../../core/rng';
import {
  MAX_VEL_1,
  MAX_VEL_2,
  acrobotObs,
  acrobotReset,
  acrobotStep,
  acrobotTerminated,
  type AcrobotState,
} from '../continuous/physics/acrobot';
import {
  MAX_SPEED,
  MAX_TORQUE,
  angleNormalize,
  pendulumObs,
  pendulumReset,
  pendulumStep,
  type PendulumState,
} from '../continuous/physics/pendulum';
import {
  START_Y,
  lunarObs,
  lunarReset,
  lunarShaping,
  lunarStep,
  type LanderState,
} from '../continuous/physics/lunarlander';

describe('Acrobot physics', () => {
  it('reset is within bounds and deterministic per seed', () => {
    const a = acrobotReset(new RNG(1));
    const b = acrobotReset(new RNG(1));
    expect(a).toEqual(b);
    for (const v of a) expect(Math.abs(v)).toBeLessThanOrEqual(0.1 + 1e-9);
  });

  it('observation is 6D with consistent cos/sin', () => {
    const o = acrobotObs([0.3, -0.7, 1, -2]);
    expect(o).toHaveLength(6);
    expect(o[0] ** 2 + o[1] ** 2).toBeCloseTo(1, 10);
    expect(o[2] ** 2 + o[3] ** 2).toBeCloseTo(1, 10);
    expect(o[4]).toBe(1);
    expect(o[5]).toBe(-2);
  });

  it('terminates when the tip is raised, not when hanging', () => {
    expect(acrobotTerminated([0, 0, 0, 0])).toBe(false); // -cos0 - cos0 = -2
    expect(acrobotTerminated([Math.PI, 0, 0, 0])).toBe(true); // 1 + 1 = 2 > 1
  });

  it('keeps velocities within clamps under sustained torque', () => {
    let s: AcrobotState = acrobotReset(new RNG(3));
    for (let i = 0; i < 400; i++) s = acrobotStep(s, 2); // torque +1
    expect(Math.abs(s[2])).toBeLessThanOrEqual(MAX_VEL_1 + 1e-9);
    expect(Math.abs(s[3])).toBeLessThanOrEqual(MAX_VEL_2 + 1e-9);
  });
});

describe('Pendulum physics', () => {
  it('angleNormalize wraps to [-π, π]', () => {
    expect(angleNormalize(0)).toBeCloseTo(0, 10);
    expect(Math.abs(angleNormalize(3 * Math.PI))).toBeCloseTo(Math.PI, 6);
    expect(angleNormalize(2 * Math.PI)).toBeCloseTo(0, 10);
  });

  it('reset within bounds; observation is 3D with unit cos/sin', () => {
    const s = pendulumReset(new RNG(2));
    expect(Math.abs(s[0])).toBeLessThanOrEqual(Math.PI + 1e-9);
    expect(Math.abs(s[1])).toBeLessThanOrEqual(1 + 1e-9);
    const o = pendulumObs([0.5, 3]);
    expect(o).toHaveLength(3);
    expect(o[0] ** 2 + o[1] ** 2).toBeCloseTo(1, 10);
    expect(o[2]).toBe(3);
  });

  it('rests at the bottom (θ=π) with no torque', () => {
    const bottom: PendulumState = [Math.PI, 0];
    const { state } = pendulumStep(bottom, 0);
    expect(state[1]).toBeCloseTo(0, 6); // sin(π)=0 → angular velocity stays ~0
  });

  it('clamps torque and angular speed', () => {
    const r = pendulumStep([Math.PI / 2, 0], 100);
    expect(r.u).toBe(MAX_TORQUE); // torque clamped
    const st = pendulumStep([0.1, MAX_SPEED], 2); // already at max speed, push harder
    expect(Math.abs(st.state[1])).toBeLessThanOrEqual(MAX_SPEED + 1e-9);
  });
});

describe('LunarLander physics', () => {
  it('reset starts high; observation is 8D with legs off the ground', () => {
    const s = lunarReset(new RNG(5));
    expect(s[1]).toBeCloseTo(START_Y, 6);
    const o = lunarObs(s);
    expect(o).toHaveLength(8);
    expect(o[6]).toBe(0);
    expect(o[7]).toBe(0);
  });

  it('gravity pulls down with no thrust; the main engine fights it', () => {
    const high: LanderState = [0, 1, 0, 0, 0, 0];
    expect(lunarStep(high, 0)[3]).toBeLessThan(0); // no-op → falling
    expect(lunarStep(high, 2)[3]).toBeGreaterThan(0); // main engine upright → rising
  });

  it('side engines change angular velocity in opposite directions', () => {
    const high: LanderState = [0, 1, 0, 0, 0, 0];
    const left = lunarStep(high, 1)[5];
    const right = lunarStep(high, 3)[5];
    expect(left).not.toBe(0);
    expect(Math.sign(left)).toBe(-Math.sign(right));
  });

  it('shaping is highest at the pad, lower when far and fast', () => {
    const atPad = lunarShaping([0, 0, 0, 0, 0, 0]);
    const farFast = lunarShaping([1, 1.3, 1, -1, 0.5, 0]);
    expect(atPad).toBeGreaterThan(farFast);
  });
});
