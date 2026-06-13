// Double Q-Learning: two Q-tables alternating updates, correcting Q-learning's maximization (over-estimation) bias.
// Update A: a* = argmaxₐ Q_A(s',a); Q_A(s,a) ← Q_A(s,a) + α[r + γ·Q_B(s',a*) − Q_A(s,a)] (symmetric).
import type { ActionExplanation, Agent, Obs, Transition, UpdateInfo } from '../../core/types';
import { RNG } from '../../core/rng';
import { TabularQ } from '../TabularQ';
import { explainEpsGreedy } from './policies';

export class DoubleQLearning implements Agent {
  readonly id = 'double-q';
  readonly name = 'Double Q-Learning';
  readonly usesQ = true;
  hyperparams: Record<string, number>;

  private qA: TabularQ;
  private qB: TabularQ;
  private rng: RNG;
  private nActions: number;
  private meanings: string[];
  private initialEpsilon: number;

  constructor(
    nStates: number,
    nActions: number,
    meanings: string[],
    hp: Record<string, number>,
    seed = 7,
  ) {
    this.nActions = nActions;
    this.meanings = meanings;
    this.hyperparams = { ...hp };
    this.initialEpsilon = hp.epsilon;
    this.qA = new TabularQ(nStates, nActions);
    this.qB = new TabularQ(nStates, nActions);
    this.rng = new RNG(seed);
  }

  // For selection and display: average of the two tables
  private avgValues(s: number): number[] {
    const out: number[] = [];
    for (let a = 0; a < this.nActions; a++) out.push((this.qA.get(s, a) + this.qB.get(s, a)) / 2);
    return out;
  }

  selectAction(state: Obs, greedy = false): number {
    const qs = this.avgValues(state as number);
    return explainEpsGreedy(state, qs, this.meanings, this.hyperparams.epsilon, greedy, this.rng)
      .action;
  }

  act(state: Obs, greedy = false): { action: number; explanation: ActionExplanation } {
    const qs = this.avgValues(state as number);
    return explainEpsGreedy(state, qs, this.meanings, this.hyperparams.epsilon, greedy, this.rng);
  }

  update(t: Transition): UpdateInfo {
    const s = t.state as number;
    const a = t.action;
    const sN = t.nextState as number;
    const { alpha, gamma } = this.hyperparams;
    const updateA = this.rng.next() < 0.5;
    const own = updateA ? this.qA : this.qB;
    const other = updateA ? this.qB : this.qA;
    const qSA = own.get(s, a);
    const aStar = own.argmax(sN); // pick the action using its own table
    const otherNext = t.terminated ? 0 : other.get(sN, aStar); // evaluate using the other table
    const target = t.reward + gamma * otherNext;
    const tdError = target - qSA;
    const newQ = qSA + alpha * tdError;
    own.set(s, a, newQ);
    return {
      alpha,
      gamma,
      reward: t.reward,
      qSA,
      otherNext,
      target,
      tdError,
      newQ,
      terminated: t.terminated ? 1 : 0,
      updatedTable: updateA ? 0 : 1,
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
    for (let s = 0; s < nStates; s++) {
      const qs = this.avgValues(s);
      v[s] = Math.max(...qs);
    }
    return v;
  }

  getQ(nStates: number, nActions: number): Float64Array {
    const out = new Float64Array(nStates * nActions);
    for (let s = 0; s < nStates; s++)
      for (let a = 0; a < nActions; a++) out[s * nActions + a] = (this.qA.get(s, a) + this.qB.get(s, a)) / 2;
    return out;
  }

  reset(): void {
    this.qA.reset();
    this.qB.reset();
    this.hyperparams.epsilon = this.initialEpsilon;
  }
}
