// Monte Carlo control (first-visit, ε-greedy): updates with the actual return G_t over the whole episode, no bootstrap.
// During the episode only record the trajectory; update only at episode end: Q(s,a) ← Q(s,a) + α[G_t − Q(s,a)] (for the first-visit (s,a)).
import type { ActionExplanation, Agent, Obs, Transition, UpdateInfo } from '../../core/types';
import { RNG } from '../../core/rng';
import { TabularQ } from '../TabularQ';
import { explainEpsGreedy } from './policies';

interface Step {
  s: number;
  a: number;
  r: number;
}

export class MonteCarlo implements Agent {
  readonly id = 'mc';
  readonly name = 'Monte Carlo Control';
  readonly usesQ = true;
  hyperparams: Record<string, number>;

  private q: TabularQ;
  private rng: RNG;
  private nActions: number;
  private meanings: string[];
  private initialEpsilon: number;
  private traj: Step[] = [];
  private lastNextState = 0;
  private lastTerminated = false;

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
    // During the episode only accumulate the trajectory, no update
    this.traj.push({ s: t.state as number, a: t.action, r: t.reward });
    this.lastNextState = t.nextState as number;
    this.lastTerminated = t.terminated;
    return {};
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    const { alpha, gamma } = this.hyperparams;
    const T = this.traj.length;
    // terminated (natural termination) → tail value 0 (pure MC, no bootstrap);
    // truncated (time cutoff) → bootstrap with V(s_T)=maxₐQ(s_T,a) to fill in the unobserved tail (rigorous handling).
    const tail = this.lastTerminated ? 0 : this.q.maxQ(this.lastNextState);
    const G = new Array<number>(T);
    let g = tail;
    for (let t = T - 1; t >= 0; t--) {
      g = this.traj[t].r + gamma * g;
      G[t] = g;
    }
    const seen = new Set<number>();
    let numUpdates = 0;
    for (let t = 0; t < T; t++) {
      const { s, a } = this.traj[t];
      const key = s * this.nActions + a;
      if (!seen.has(key)) {
        seen.add(key);
        const qSA = this.q.get(s, a);
        this.q.set(s, a, qSA + alpha * (G[t] - qSA));
        numUpdates += 1;
      }
    }
    const finalG = T > 0 ? G[0] : 0;
    this.traj = [];
    this.hyperparams.epsilon = Math.max(
      this.hyperparams.epsilonMin,
      this.hyperparams.epsilon * this.hyperparams.epsilonDecay,
    );
    return { gamma, alpha, finalG, numUpdates, bootstrapped: this.lastTerminated ? 0 : 1, tail };
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
    this.traj = [];
    this.hyperparams.epsilon = this.initialEpsilon;
  }
}
