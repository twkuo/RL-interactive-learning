import { describe, it, expect } from 'vitest';
import { CartPole } from '../continuous/CartPole';
import { CartPoleVec } from '../continuous/CartPoleVec';

describe('CartPoleVec', () => {
  it('exposes a 4-dim continuous (box) observation', () => {
    const env = new CartPoleVec(7);
    const o = env.resetSync(7);
    expect(o).toHaveLength(4);
    expect(env.observationSpace.kind).toBe('box');
    expect(env.observationSpace.shape[0]).toBe(4);
    expect(env.actionSpace.kind).toBe('discrete');
  });

  it('shares identical physics with the tabular CartPole (same seed + actions)', () => {
    const tab = new CartPole(123);
    const vec = new CartPoleVec(123);
    tab.resetSync(123);
    const o0 = vec.resetSync(123);
    const r0 = tab.getRenderState();
    expect(o0[0]).toBeCloseTo(r0.x, 12);
    expect(o0[2]).toBeCloseTo(r0.theta, 12);

    const actions = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1];
    for (const a of actions) {
      const rTab = tab.stepSync(a);
      const rVec = vec.stepSync(a);
      const rs = tab.getRenderState();
      const ov = rVec.observation as number[];
      expect(ov[0]).toBeCloseTo(rs.x, 12);
      expect(ov[1]).toBeCloseTo(rs.xDot, 12);
      expect(ov[2]).toBeCloseTo(rs.theta, 12);
      expect(ov[3]).toBeCloseTo(rs.thetaDot, 12);
      expect(rVec.terminated).toBe(rTab.terminated);
      expect(rVec.truncated).toBe(rTab.truncated);
    }
  });

  it('truncates at maxSteps when the pole never falls (forced balance is not expected; just bound steps)', () => {
    const vec = new CartPoleVec(1);
    vec.maxSteps = 5;
    vec.resetSync(1);
    let steps = 0;
    let done = false;
    while (!done && steps < 20) {
      const r = vec.stepSync(0);
      steps += 1;
      done = r.terminated || r.truncated;
    }
    // The episode is bounded: it either falls (terminated) or hits the 5-step limit (truncated).
    expect(steps).toBeLessThanOrEqual(5);
    expect(done).toBe(true);
  });
});
