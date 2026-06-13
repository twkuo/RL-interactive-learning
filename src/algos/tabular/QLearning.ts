// Q-Learning (off-policy TD control).
// Q(s,a) ← Q(s,a) + α[r + γ·maxₐ' Q(s',a')·(1−terminated) − Q(s,a)]
import type { ActionExplanation, Agent, Obs, Transition, UpdateInfo } from '../../core/types';
import { RNG } from '../../core/rng';
import { TabularQ } from '../TabularQ';
import { explainEpsGreedy } from './policies';

export class QLearning implements Agent {
  readonly id = 'q-learning';
  readonly name = 'Q-Learning';
  readonly usesQ = true;
  hyperparams: Record<string, number>;

  private q: TabularQ;
  private rng: RNG;
  private meanings: string[];
  private initialEpsilon: number;

  constructor(
    nStates: number,
    nActions: number,
    meanings: string[],
    hp: Record<string, number>,
    seed = 7,
  ) {
    this.meanings = meanings;
    this.hyperparams = { ...hp };
    this.initialEpsilon = hp.epsilon;
    this.q = new TabularQ(nStates, nActions);
    this.rng = new RNG(seed);
  }

  selectAction(state: Obs, greedy = false): number {
    const qs = this.q.values(state as number);
    return explainEpsGreedy(state, qs, this.meanings, this.hyperparams.epsilon, greedy, this.rng)
      .action;
  }

  act(state: Obs, greedy = false): { action: number; explanation: ActionExplanation } {
    const qs = this.q.values(state as number);
    return explainEpsGreedy(state, qs, this.meanings, this.hyperparams.epsilon, greedy, this.rng);
  }

  update(t: Transition): UpdateInfo {
    const s = t.state as number;
    const a = t.action;
    const sN = t.nextState as number;
    const { alpha, gamma } = this.hyperparams;
    const qSA = this.q.get(s, a);
    const maxQNext = t.terminated ? 0 : this.q.maxQ(sN);
    const target = t.reward + gamma * maxQNext;
    const tdError = target - qSA;
    const newQ = qSA + alpha * tdError;
    this.q.set(s, a, newQ);
    return {
      alpha,
      gamma,
      reward: t.reward,
      qSA,
      maxQNext,
      target,
      tdError,
      newQ,
      terminated: t.terminated ? 1 : 0,
    };
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    this.hyperparams.epsilon = Math.max(
      this.hyperparams.epsilonMin,
      this.hyperparams.epsilon * this.hyperparams.epsilonDecay,
    );
    return undefined;
  }

  getV(nStates: number): Float64Array {
    const v = new Float64Array(nStates);
    for (let s = 0; s < nStates; s++) v[s] = this.q.maxQ(s);
    return v;
  }

  getQ(): Float64Array {
    return this.q.raw().slice();
  }

  reset(): void {
    this.q.reset();
    this.hyperparams.epsilon = this.initialEpsilon;
  }
}
