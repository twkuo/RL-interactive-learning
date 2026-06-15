// Proximal Policy Optimization (PPO) — actor-critic, GAE(λ) advantages, clipped surrogate
// objective, value loss, and an entropy bonus, trained over minibatch epochs on rollouts.
//
// Like DQN this module statically imports TensorFlow.js and must only be reached via a DYNAMIC
// import (registry `() => import('./deep/PPO')` and the training worker), keeping tfjs out of the
// main bundle.
//
// Main thread: actor for synchronous act() (samples from π, drives the DecisionPanel) + critic for
// V(s). update() is a no-op — learning happens in the worker via rollout collection + learn().
import * as tf from '@tensorflow/tfjs';
import type {
  Action,
  ActionExplanation,
  Agent,
  Obs,
  SyncEnvironment,
  Transition,
  UpdateInfo,
} from '../../core/types';
import { RNG } from '../../core/rng';
import { argmax, softmax } from '../../core/utils';
import { buildMLP } from '../../core/nn/mlp';
import { dumpWeights, loadWeights, type WeightDump } from '../../core/nn/weights';

export interface PPOMetrics {
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKL: number;
  clipFrac: number;
}

// A collected rollout (built by the worker). nextVals are already zeroed on terminated steps;
// boundaries mark episode ends (terminated OR truncated) so GAE doesn't cross episodes.
export interface Rollout {
  states: number[][];
  actions: number[];
  logps: number[];
  rewards: number[];
  values: number[];
  nextVals: number[];
  boundaries: number[];
}

function sampleFrom(probs: number[], u: number): number {
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (u < c) return i;
  }
  return probs.length - 1;
}

export class PPO implements Agent {
  readonly id = 'ppo';
  readonly name = 'PPO';
  readonly usesQ = false;
  hyperparams: Record<string, number>;

  private inputDim: number;
  private nActions: number;
  private actionLabels: string[];
  private rng: RNG;
  private obsCenter: number[];
  private obsHalf: number[];
  private normalize: boolean;

  private lr: number;
  private gamma: number;
  private lambda: number;
  private clip: number;
  private entropyCoef: number;
  private vfCoef: number;
  private epochs: number;
  private minibatch: number;
  private hidden: number;

  private actor!: tf.Sequential;
  private critic!: tf.Sequential;
  private optimizer!: tf.Optimizer;

  constructor(
    inputDim: number,
    nActions: number,
    actionLabels: string[],
    hp: Record<string, number>,
    low: number[],
    high: number[],
    seed = 12345,
  ) {
    this.inputDim = inputDim;
    this.nActions = nActions;
    this.actionLabels = actionLabels;
    this.hyperparams = { ...hp };
    this.rng = new RNG(seed);
    this.obsCenter = low.map((lo, i) => (lo + high[i]) / 2);
    this.obsHalf = low.map((lo, i) => Math.max((high[i] - lo) / 2, 1e-6));
    this.normalize = (hp.normalize ?? 1) >= 0.5;

    this.lr = hp.lr ?? 0.0003;
    this.gamma = hp.gamma ?? 0.99;
    this.lambda = hp.gaeLambda ?? 0.95;
    this.clip = hp.clip ?? 0.2;
    this.entropyCoef = hp.entropyCoef ?? 0.01;
    this.vfCoef = hp.vfCoef ?? 0.5;
    this.epochs = Math.round(hp.epochs ?? 6);
    this.minibatch = Math.round(hp.minibatch ?? 64);
    this.hidden = Math.round(hp.hiddenUnits ?? 64);
    this.build();
  }

  private build(): void {
    this.actor = buildMLP({ inputDim: this.inputDim, hidden: [this.hidden, this.hidden], outputDim: this.nActions });
    this.critic = buildMLP({ inputDim: this.inputDim, hidden: [this.hidden, this.hidden], outputDim: 1 });
    this.optimizer = tf.train.adam(this.lr);
  }

  private norm(s: number[]): number[] {
    if (!this.normalize) return s;
    const out = new Array<number>(s.length);
    for (let i = 0; i < s.length; i++) out[i] = (s[i] - this.obsCenter[i]) / this.obsHalf[i];
    return out;
  }

  private policyProbs(state: number[]): number[] {
    return tf.tidy(() => {
      const logits = this.actor.predict(tf.tensor2d([this.norm(state)], [1, this.inputDim])) as tf.Tensor2D;
      return softmax(Array.from(logits.dataSync()));
    });
  }

  // V(s) from the critic — used for display and (in the worker) GAE bootstrap.
  // (Named stateValue, not valueOf, to avoid shadowing Object.prototype.valueOf.)
  stateValue(state: number[]): number {
    return tf.tidy(() => {
      const v = this.critic.predict(tf.tensor2d([this.norm(state)], [1, this.inputDim])) as tf.Tensor2D;
      return v.dataSync()[0];
    });
  }

  selectAction(state: Obs, greedy = false): Action {
    const p = this.policyProbs(state as number[]);
    return greedy ? argmax(p) : sampleFrom(p, this.rng.next());
  }

  act(state: Obs, greedy = false): { action: Action; explanation: ActionExplanation } {
    const s = state as number[];
    const p = this.policyProbs(s);
    const greedyAction = argmax(p);
    const draw = this.rng.next();
    const action = greedy ? greedyAction : sampleFrom(p, draw);
    const rationale = greedy
      ? `Greedy: highest-probability action "${this.actionLabels[greedyAction]}" (π=${p[greedyAction].toFixed(2)})`
      : `Sampled from the policy π: drew "${this.actionLabels[action]}" (π=${p[action].toFixed(2)})`;
    return {
      action,
      explanation: {
        state: s,
        actionMeanings: this.actionLabels,
        policyKind: 'softmax',
        actionProbs: p,
        greedyAction,
        randomDraw: draw,
        chosenAction: action,
        rationale,
      },
    };
  }

  // Interactive single-step never trains (that happens in the worker via learn()).
  update(_t: Transition): UpdateInfo {
    return {};
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    return undefined;
  }

  // ---- training (worker) ----
  // Sample an action for rollout collection, returning log π(a|s) and V(s).
  rolloutAction(state: number[]): { action: number; logp: number; value: number } {
    return tf.tidy(() => {
      const x = tf.tensor2d([this.norm(state)], [1, this.inputDim]);
      const logits = this.actor.predict(x) as tf.Tensor2D;
      const logpAll = Array.from(tf.logSoftmax(logits).dataSync());
      const probs = softmax(Array.from(logits.dataSync()));
      const action = sampleFrom(probs, this.rng.next());
      const value = (this.critic.predict(x) as tf.Tensor2D).dataSync()[0];
      return { action, logp: logpAll[action], value };
    });
  }

  // One PPO update on a rollout: GAE(λ) → normalized advantages → K epochs of minibatch SGD.
  learn(r: Rollout): PPOMetrics {
    const N = r.states.length;
    const adv = new Array<number>(N).fill(0);
    let last = 0;
    for (let t = N - 1; t >= 0; t--) {
      const delta = r.rewards[t] + this.gamma * r.nextVals[t] - r.values[t];
      last = delta + this.gamma * this.lambda * (1 - r.boundaries[t]) * last;
      adv[t] = last;
    }
    const returns = adv.map((a, t) => a + r.values[t]);
    const mean = adv.reduce((x, y) => x + y, 0) / N;
    const std = Math.sqrt(adv.reduce((x, y) => x + (y - mean) * (y - mean), 0) / N) + 1e-8;
    const advN = adv.map((a) => (a - mean) / std);
    const normStates = r.states.map((s) => this.norm(s));

    const clip = this.clip;
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Fisher-Yates shuffle (seeded RNG for reproducibility).
      for (let i = N - 1; i > 0; i--) {
        const j = this.rng.int(i + 1);
        const tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
      }
      for (let start = 0; start < N; start += this.minibatch) {
        const mb = idx.slice(start, start + this.minibatch);
        const sMb = mb.map((k) => normStates[k]);
        const aMb = mb.map((k) => r.actions[k]);
        const oldMb = mb.map((k) => r.logps[k]);
        const advMb = mb.map((k) => advN[k]);
        const retMb = mb.map((k) => returns[k]);
        tf.tidy(() => {
          const S = tf.tensor2d(sMb);
          const A = tf.oneHot(tf.tensor1d(aMb, 'int32'), this.nActions);
          const oldlp = tf.tensor1d(oldMb);
          const advT = tf.tensor1d(advMb);
          const retT = tf.tensor1d(retMb);
          this.optimizer.minimize(() => {
            const logits = this.actor.predict(S) as tf.Tensor2D;
            const logpAll = tf.logSoftmax(logits);
            const logpA = logpAll.mul(A).sum(1);
            const ratio = logpA.sub(oldlp).exp();
            const surr1 = ratio.mul(advT);
            const surr2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advT);
            const pLoss = tf.minimum(surr1, surr2).mean().mul(-1);
            const probs = tf.softmax(logits);
            const entropy = probs.mul(logpAll).sum(1).mean().mul(-1);
            const v = (this.critic.predict(S) as tf.Tensor2D).reshape([-1]);
            const vLoss = v.sub(retT).square().mean();
            return pLoss.add(vLoss.mul(this.vfCoef)).sub(entropy.mul(this.entropyCoef)) as tf.Scalar;
          });
        });
      }
    }

    // Diagnostics on the full batch (one forward pass; for the dashboard).
    return tf.tidy(() => {
      const S = tf.tensor2d(normStates);
      const A = tf.oneHot(tf.tensor1d(r.actions, 'int32'), this.nActions);
      const logits = this.actor.predict(S) as tf.Tensor2D;
      const logpAll = tf.logSoftmax(logits);
      const logpA = logpAll.mul(A).sum(1);
      const oldlp = tf.tensor1d(r.logps);
      const ratio = logpA.sub(oldlp).exp();
      const advT = tf.tensor1d(advN);
      const surr1 = ratio.mul(advT);
      const surr2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advT);
      const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1).dataSync()[0];
      const probs = tf.softmax(logits);
      const entropy = probs.mul(logpAll).sum(1).mean().mul(-1).dataSync()[0];
      const v = (this.critic.predict(S) as tf.Tensor2D).reshape([-1]);
      const valueLoss = v.sub(tf.tensor1d(returns)).square().mean().dataSync()[0];
      const approxKL = oldlp.sub(logpA).mean().dataSync()[0];
      const clipFrac = tf.abs(ratio.sub(1)).greater(clip).cast('float32').mean().dataSync()[0];
      return { policyLoss, valueLoss, entropy, approxKL, clipFrac };
    });
  }

  // ---- weight transfer (worker → main thread): [actor, critic] ----
  dumpWeights(): WeightDump[] {
    return [dumpWeights(this.actor), dumpWeights(this.critic)];
  }

  loadWeightDump(dumps: WeightDump[]): void {
    loadWeights(this.actor, dumps[0]);
    loadWeights(this.critic, dumps[1]);
  }

  reset(): void {
    this.dispose();
    this.build();
  }

  dispose(): void {
    this.actor.dispose();
    this.critic.dispose();
    this.optimizer.dispose();
  }
}

export function create(env: SyncEnvironment, hp: Record<string, number>): PPO {
  const obs = env.observationSpace;
  const inputDim = obs.kind === 'box' ? obs.shape[0] : obs.n;
  const nActions = env.actionSpace.kind === 'discrete' ? env.actionSpace.n : 1;
  const low = obs.kind === 'box' ? obs.low : new Array<number>(inputDim).fill(-1);
  const high = obs.kind === 'box' ? obs.high : new Array<number>(inputDim).fill(1);
  return new PPO(inputDim, nActions, env.actionMeanings(), hp, low, high);
}
