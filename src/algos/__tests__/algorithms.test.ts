import { describe, it, expect } from 'vitest';
import { QLearning } from '../tabular/QLearning';
import { Sarsa } from '../tabular/Sarsa';
import { ExpectedSarsa } from '../tabular/ExpectedSarsa';
import { DoubleQLearning } from '../tabular/DoubleQLearning';
import { MonteCarlo } from '../tabular/MonteCarlo';
import { Reinforce } from '../tabular/Reinforce';
import { RNG } from '../../core/rng';
import { epsilonGreedy } from '../tabular/policies';
import { DEFAULT_HYPERPARAMS } from '../registry';
import { makeFrozenLake } from '../../envs/discrete/FrozenLake';

const meanings = ['Left', 'Down', 'Right', 'Up'];

describe('Q-Learning update', () => {
  it('TD target when Q is all zeros', () => {
    const a = new QLearning(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 0.1, gamma: 0.95 });
    const info = a.update({
      state: 0,
      action: 0,
      reward: 1,
      nextState: 1,
      terminated: false,
      truncated: false,
      nextAction: 0,
    });
    expect(info.maxQNext).toBe(0);
    expect(info.target).toBeCloseTo(1, 10);
    expect(info.tdError).toBeCloseTo(1, 10);
    expect(info.newQ).toBeCloseTo(0.1, 10);
  });

  it('bootstraps with max Q(s′,·) when not terminal; zeroed when terminated', () => {
    const a = new QLearning(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 0.9 });
    // alpha=1 → newQ = target = reward, pulls Q(1,2) to 0.5
    a.update({ state: 1, action: 2, reward: 0.5, nextState: 0, terminated: true, truncated: false });
    const cont = a.update({
      state: 0,
      action: 0,
      reward: 0,
      nextState: 1,
      terminated: false,
      truncated: false,
      nextAction: 2,
    });
    expect(cont.maxQNext).toBeCloseTo(0.5, 10);
    expect(cont.target).toBeCloseTo(0.45, 10); // 0 + 0.9 * 0.5
    const term = a.update({
      state: 2,
      action: 0,
      reward: 0,
      nextState: 1,
      terminated: true,
      truncated: false,
    });
    expect(term.maxQNext).toBe(0);
    expect(term.target).toBe(0);
  });
});

describe('SARSA update', () => {
  it('uses the actual next action a′ (not max)', () => {
    const a = new Sarsa(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 0.9 });
    a.update({ state: 1, action: 2, reward: 0.5, nextState: 0, terminated: true, truncated: false }); // Q(1,2)=0.5
    a.update({ state: 1, action: 3, reward: 0.2, nextState: 0, terminated: true, truncated: false }); // Q(1,3)=0.2
    const info = a.update({
      state: 0,
      action: 0,
      reward: 0,
      nextState: 1,
      terminated: false,
      truncated: false,
      nextAction: 3,
    });
    expect(info.qNextSA).toBeCloseTo(0.2, 10); // uses Q(1,3) rather than max=0.5
    expect(info.target).toBeCloseTo(0.18, 10); // 0 + 0.9 * 0.2
  });
});

describe('ε-greedy', () => {
  it('always picks argmax when ε=0', () => {
    const rng = new RNG(1);
    const res = epsilonGreedy([0, 1, 0.5, -1], 0, rng);
    expect(res.isExploring).toBe(false);
    expect(res.action).toBe(1);
  });

  it('always explores when ε=1', () => {
    const rng = new RNG(1);
    for (let i = 0; i < 20; i++) {
      expect(epsilonGreedy([0, 1, 0, 0], 1, rng).isExploring).toBe(true);
    }
  });

  it('reproducible with the same seed', () => {
    const r1 = new RNG(123);
    const r2 = new RNG(123);
    const seq1 = Array.from({ length: 10 }, () => epsilonGreedy([0.1, 0.2, 0.3, 0.4], 0.5, r1).action);
    const seq2 = Array.from({ length: 10 }, () => epsilonGreedy([0.1, 0.2, 0.3, 0.4], 0.5, r2).action);
    expect(seq1).toEqual(seq2);
  });
});

describe('Expected SARSA', () => {
  it('bootstraps with the ε-greedy expected value', () => {
    const ag = new ExpectedSarsa(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 1, epsilon: 0.5 });
    ag.update({ state: 1, action: 0, reward: 1, nextState: 2, terminated: true, truncated: false }); // Q(1,0)=1
    // Q(1,*)=[1,0,0,0] → E = (1-0.5)·1 + 0.5·0.25 = 0.625
    const info = ag.update({ state: 0, action: 0, reward: 0, nextState: 1, terminated: false, truncated: false });
    expect(info.expectedQNext).toBeCloseTo(0.625, 10);
    expect(info.target).toBeCloseTo(0.625, 10);
  });
});

describe('Double Q-Learning', () => {
  it('a single step updates only one table (average Q changes by half)', () => {
    const ag = new DoubleQLearning(2, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 1 });
    const info = ag.update({ state: 0, action: 0, reward: 1, nextState: 1, terminated: true, truncated: false });
    expect(info.newQ).toBeCloseTo(1, 10); // updated table Q(0,0)=1
    expect(ag.getQ(2, 4)[0]).toBeCloseTo(0.5, 10); // average = (1+0)/2
  });
});

describe('Monte Carlo Control', () => {
  it('episode-end first-visit backfill with G_t', () => {
    const ag = new MonteCarlo(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 1 });
    ag.update({ state: 0, action: 0, reward: 1, nextState: 1, terminated: false, truncated: false });
    ag.update({ state: 1, action: 1, reward: 2, nextState: 2, terminated: false, truncated: false });
    ag.update({ state: 2, action: 2, reward: 3, nextState: 0, terminated: true, truncated: false });
    const summary = ag.onEpisodeEnd();
    expect(summary?.finalG).toBeCloseTo(6, 10); // 1+2+3
    expect(summary?.numUpdates).toBe(3);
    const Q = ag.getQ();
    expect(Q[0 * 4 + 0]).toBeCloseTo(6, 10);
    expect(Q[1 * 4 + 1]).toBeCloseTo(5, 10);
    expect(Q[2 * 4 + 2]).toBeCloseTo(3, 10);
  });

  it('terminated does not bootstrap; truncated bootstraps with V(s_T)', () => {
    const ag = new MonteCarlo(3, 4, meanings, { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 1 });
    // First use a terminated episode to set Q(2,0)=5 → maxQ(2)=5
    ag.update({ state: 2, action: 0, reward: 5, nextState: 0, terminated: true, truncated: false });
    ag.onEpisodeEnd();
    // truncated episode: last nextState=2, truncated → tail = maxQ(2) = 5
    ag.update({ state: 0, action: 1, reward: 1, nextState: 2, terminated: false, truncated: true });
    const summary = ag.onEpisodeEnd();
    expect(summary?.bootstrapped).toBe(1);
    expect(summary?.tail).toBeCloseTo(5, 10);
    expect(summary?.finalG).toBeCloseTo(6, 10); // 1 + 1·5
    expect(ag.getQ()[0 * 4 + 1]).toBeCloseTo(6, 10);
  });
});

describe('REINFORCE', () => {
  it('softmax probabilities sum to 1; the chosen action probability rises on positive return', () => {
    const ag = new Reinforce(2, 3, ['a', 'b', 'c'], { ...DEFAULT_HYPERPARAMS, alpha: 1, gamma: 1 });
    ag.update({ state: 0, action: 1, reward: 1, nextState: 1, terminated: true, truncated: false });
    ag.onEpisodeEnd();
    const P = ag.getPolicy(2, 3);
    expect(P[0] + P[1] + P[2]).toBeCloseTo(1, 10);
    expect(P[1]).toBeGreaterThan(P[0]); // probability of the chosen action 1 > action 0
  });
});

describe('editable rewards', () => {
  it('setRewardParam changes the environment reward', () => {
    const fl = makeFrozenLake(false); // no slipping, deterministic; state 1 moving down = fall into hole 5
    const before = fl.transitions(1, 1).find((o) => o.nextState === 5);
    expect(before?.reward).toBe(0); // FrozenLake default hole reward is 0
    fl.setRewardParam('holeReward', -5);
    const after = fl.transitions(1, 1).find((o) => o.nextState === 5);
    expect(after?.reward).toBe(-5);
    expect(fl.rewardParams().find((p) => p.key === 'holeReward')?.value).toBe(-5);
  });
});
