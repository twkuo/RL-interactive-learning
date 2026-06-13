// REINFORCE (tabular softmax policy gradient): learns the policy π directly, not Q/V.
// During the episode only record the trajectory; at episode end update the preferences θ via policy gradient:
//   θ(s,a) ← θ(s,a) + α·γ^t·G_t·(1[a=a_t] − π(a|s))
import type { ActionExplanation, Agent, Obs, Transition, UpdateInfo } from '../../core/types';
import { RNG } from '../../core/rng';
import { argmax, softmax } from '../../core/utils';

interface Step {
  s: number;
  a: number;
  r: number;
}

export class Reinforce implements Agent {
  readonly id = 'reinforce';
  readonly name = 'REINFORCE';
  readonly usesQ = false;
  hyperparams: Record<string, number>;

  private theta: Float64Array;
  private rng: RNG;
  private nActions: number;
  private meanings: string[];
  private traj: Step[] = [];

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
    this.theta = new Float64Array(nStates * nActions);
    this.rng = new RNG(seed);
  }

  private probs(s: number): number[] {
    const base = s * this.nActions;
    const logits: number[] = [];
    for (let a = 0; a < this.nActions; a++) logits.push(this.theta[base + a]);
    return softmax(logits);
  }

  selectAction(state: Obs, greedy = false): number {
    const p = this.probs(state as number);
    if (greedy) return argmax(p);
    return this.sample(p);
  }

  private sample(p: number[]): number {
    const u = this.rng.next();
    let acc = 0;
    for (let a = 0; a < p.length; a++) {
      acc += p[a];
      if (u < acc) return a;
    }
    return p.length - 1;
  }

  act(state: Obs, greedy = false): { action: number; explanation: ActionExplanation } {
    const s = state as number;
    const p = this.probs(s);
    const g = argmax(p);
    if (greedy) {
      return {
        action: g,
        explanation: {
          state,
          actionMeanings: this.meanings,
          policyKind: 'softmax',
          actionProbs: p,
          greedyAction: g,
          chosenAction: g,
          rationale: `Greedy: pick the highest-probability action "${this.meanings[g]}" (π=${p[g].toFixed(2)})`,
        },
      };
    }
    const u = this.rng.next();
    let acc = 0;
    let action = p.length - 1;
    for (let a = 0; a < p.length; a++) {
      acc += p[a];
      if (u < acc) {
        action = a;
        break;
      }
    }
    return {
      action,
      explanation: {
        state,
        actionMeanings: this.meanings,
        policyKind: 'softmax',
        actionProbs: p,
        greedyAction: g,
        randomDraw: u,
        chosenAction: action,
        rationale: `Sampled from policy π: drew ${u.toFixed(3)} → "${this.meanings[action]}" (π=${p[
          action
        ].toFixed(2)})`,
      },
    };
  }

  update(t: Transition): UpdateInfo {
    this.traj.push({ s: t.state as number, a: t.action, r: t.reward });
    return {};
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    const { alpha, gamma } = this.hyperparams;
    const T = this.traj.length;
    const G = new Array<number>(T);
    let g = 0;
    for (let t = T - 1; t >= 0; t--) {
      g = this.traj[t].r + gamma * g;
      G[t] = g;
    }
    let gpow = 1; // γ^t
    for (let t = 0; t < T; t++) {
      const { s, a } = this.traj[t];
      const base = s * this.nActions;
      const p = this.probs(s);
      for (let b = 0; b < this.nActions; b++) {
        const grad = (b === a ? 1 : 0) - p[b];
        this.theta[base + b] += alpha * gpow * G[t] * grad;
      }
      gpow *= gamma;
    }
    const finalG = T > 0 ? G[0] : 0;
    this.traj = [];
    return { gamma, alpha, finalG, numUpdates: T };
  }

  getPolicy(nStates: number, nActions: number): Float64Array {
    const out = new Float64Array(nStates * nActions);
    for (let s = 0; s < nStates; s++) {
      const p = this.probs(s);
      for (let a = 0; a < nActions; a++) out[s * nActions + a] = p[a];
    }
    return out;
  }

  reset(): void {
    this.theta.fill(0);
    this.traj = [];
  }
}
