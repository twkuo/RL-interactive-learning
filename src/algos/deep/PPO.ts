// Proximal Policy Optimization (PPO) — actor-critic, GAE(λ) advantages, clipped surrogate
// objective, value loss, and an entropy bonus, trained over minibatch epochs on rollouts.
//
// Supports BOTH action kinds:
//   - discrete   → softmax policy (actor outputs logits over n actions)
//   - continuous → diagonal Gaussian policy (actor outputs [mean, log-std] for a 1-D action,
//                  sampled and clamped to the action bounds; e.g. Pendulum torque)
//
// Like DQN this statically imports TensorFlow.js and must only be reached via a DYNAMIC import.
// Main thread: actor for synchronous act() + critic for V(s). update() is a no-op — learning runs
// in the worker via rollout collection + learn().
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

// A collected rollout (built by the worker). For discrete actions, `actions` holds indices; for
// continuous, it holds the scalar (clamped) action. nextVals are zeroed on terminated steps;
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

export type PPOActionSpec =
  | { kind: 'discrete'; n: number; labels: string[] }
  | { kind: 'continuous'; low: number; high: number; labels: string[] };

const LOG_STD_MIN = -5;
const LOG_STD_MAX = 0.2; // cap σ ≈ 1.2: a state-dependent log-std can otherwise drift up and make rollouts near-random
const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI);
const GAUSS_ENTROPY_CONST = 0.5 * Math.log(2 * Math.PI * Math.E);

function sampleFrom(probs: number[], u: number): number {
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (u < c) return i;
  }
  return probs.length - 1;
}

// Standard normal via Box-Muller.
function gauss(rng: RNG): number {
  const u1 = Math.max(rng.next(), 1e-9);
  const u2 = rng.next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

export class PPO implements Agent {
  readonly id = 'ppo';
  readonly name = 'PPO';
  readonly usesQ = false;
  hyperparams: Record<string, number>;

  private inputDim: number;
  private spec: PPOActionSpec;
  private continuous: boolean;
  private outDim: number; // actor output width: nActions (discrete) or 2 = [mean, log-std] (continuous)
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
    low: number[],
    high: number[],
    spec: PPOActionSpec,
    hp: Record<string, number>,
    seed = 12345,
  ) {
    this.inputDim = inputDim;
    this.spec = spec;
    this.continuous = spec.kind === 'continuous';
    this.outDim = spec.kind === 'continuous' ? 2 : spec.n;
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
    this.actor = buildMLP({ inputDim: this.inputDim, hidden: [this.hidden, this.hidden], outputDim: this.outDim });
    this.critic = buildMLP({ inputDim: this.inputDim, hidden: [this.hidden, this.hidden], outputDim: 1 });
    this.optimizer = tf.train.adam(this.lr);
  }

  private norm(s: number[]): number[] {
    if (!this.normalize) return s;
    const out = new Array<number>(s.length);
    for (let i = 0; i < s.length; i++) out[i] = (s[i] - this.obsCenter[i]) / this.obsHalf[i];
    return out;
  }

  // Raw actor output for a single state.
  private actorOut(state: number[]): number[] {
    return tf.tidy(() => {
      const o = this.actor.predict(tf.tensor2d([this.norm(state)], [1, this.inputDim])) as tf.Tensor2D;
      return Array.from(o.dataSync());
    });
  }

  // Discrete: π(a|s). (Continuous has no categorical distribution.)
  private policyProbs(state: number[]): number[] {
    return softmax(this.actorOut(state));
  }

  // Continuous: (mean, std) of the Gaussian for a state.
  private gaussianParams(state: number[]): { mean: number; std: number } {
    const o = this.actorOut(state);
    return { mean: o[0], std: Math.exp(clamp(o[1], LOG_STD_MIN, LOG_STD_MAX)) };
  }

  stateValue(state: number[]): number {
    return tf.tidy(() => {
      const v = this.critic.predict(tf.tensor2d([this.norm(state)], [1, this.inputDim])) as tf.Tensor2D;
      return v.dataSync()[0];
    });
  }

  private bounds(): { low: number; high: number } {
    return this.spec.kind === 'continuous' ? { low: this.spec.low, high: this.spec.high } : { low: 0, high: 0 };
  }

  selectAction(state: Obs, greedy = false): Action {
    const s = state as number[];
    if (this.continuous) {
      const { mean, std } = this.gaussianParams(s);
      const { low, high } = this.bounds();
      const a = greedy ? mean : mean + std * gauss(this.rng);
      return clamp(a, low, high);
    }
    const p = this.policyProbs(s);
    return greedy ? argmax(p) : sampleFrom(p, this.rng.next());
  }

  act(state: Obs, greedy = false): { action: Action; explanation: ActionExplanation } {
    const s = state as number[];
    if (this.continuous) {
      const { mean, std } = this.gaussianParams(s);
      const { low, high } = this.bounds();
      const a = clamp(greedy ? mean : mean + std * gauss(this.rng), low, high);
      const label = this.spec.kind === 'continuous' ? this.spec.labels[0] : 'action';
      return {
        action: a,
        explanation: {
          state: s,
          actionMeanings: this.spec.labels,
          policyKind: 'gaussian',
          mean,
          std,
          continuousAction: a,
          chosenAction: 0,
          rationale: greedy
            ? `Greedy: ${label} = mean μ = ${mean.toFixed(2)} (clamped to [${low}, ${high}])`
            : `Sampled ${label} = ${a.toFixed(2)} from N(μ=${mean.toFixed(2)}, σ=${std.toFixed(2)})`,
        },
      };
    }
    const p = this.policyProbs(s);
    const greedyAction = argmax(p);
    const draw = this.rng.next();
    const action = greedy ? greedyAction : sampleFrom(p, draw);
    return {
      action,
      explanation: {
        state: s,
        actionMeanings: this.spec.labels,
        policyKind: 'softmax',
        actionProbs: p,
        greedyAction,
        randomDraw: draw,
        chosenAction: action,
        rationale: greedy
          ? `Greedy: highest-probability action "${this.spec.labels[greedyAction]}" (π=${p[greedyAction].toFixed(2)})`
          : `Sampled from the policy π: drew "${this.spec.labels[action]}" (π=${p[action].toFixed(2)})`,
      },
    };
  }

  update(_t: Transition): UpdateInfo {
    return {};
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    return undefined;
  }

  // ---- training (worker) ----
  rolloutAction(state: number[]): { action: number; logp: number; value: number } {
    return tf.tidy(() => {
      const x = tf.tensor2d([this.norm(state)], [1, this.inputDim]);
      const value = (this.critic.predict(x) as tf.Tensor2D).dataSync()[0];
      const o = Array.from((this.actor.predict(x) as tf.Tensor2D).dataSync());
      if (this.continuous) {
        const { low, high } = this.bounds();
        const mean = o[0];
        const std = Math.exp(clamp(o[1], LOG_STD_MIN, LOG_STD_MAX));
        const a = clamp(mean + std * gauss(this.rng), low, high);
        const z = (a - mean) / std;
        const logp = -0.5 * z * z - Math.log(std) - HALF_LOG_2PI;
        return { action: a, logp, value };
      }
      const probs = softmax(o);
      const action = sampleFrom(probs, this.rng.next());
      const logpAll = Array.from(tf.logSoftmax(tf.tensor1d(o)).dataSync());
      return { action, logp: logpAll[action], value };
    });
  }

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
    const cont = this.continuous;
    const nA = this.outDim;

    // Per-minibatch log π(a|s) under the current policy (+ a scalar entropy proxy for the batch).
    const logpAndEntropy = (
      S: tf.Tensor2D,
      acts: number[],
    ): { logpA: tf.Tensor1D; entropy: tf.Scalar } => {
      const out = this.actor.predict(S) as tf.Tensor2D;
      if (cont) {
        const meanT = out.slice([0, 0], [-1, 1]).reshape([-1]);
        const logstd = tf.clipByValue(out.slice([0, 1], [-1, 1]).reshape([-1]), LOG_STD_MIN, LOG_STD_MAX);
        const stdT = logstd.exp();
        const z = tf.tensor1d(acts).sub(meanT).div(stdT);
        const logpA = z.square().mul(-0.5).sub(logstd).sub(HALF_LOG_2PI) as tf.Tensor1D;
        const entropy = logstd.add(GAUSS_ENTROPY_CONST).mean() as tf.Scalar;
        return { logpA, entropy };
      }
      const logpAll = tf.logSoftmax(out);
      const oneHot = tf.oneHot(tf.tensor1d(acts, 'int32'), nA);
      const logpA = logpAll.mul(oneHot).sum(1) as tf.Tensor1D;
      const entropy = tf.softmax(out).mul(logpAll).sum(1).mean().mul(-1) as tf.Scalar;
      return { logpA, entropy };
    };

    const idx = Array.from({ length: N }, (_, i) => i);
    for (let epoch = 0; epoch < this.epochs; epoch++) {
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
          const oldlp = tf.tensor1d(oldMb);
          const advT = tf.tensor1d(advMb);
          const retT = tf.tensor1d(retMb);
          this.optimizer.minimize(() => {
            const { logpA, entropy } = logpAndEntropy(S, aMb);
            const ratio = logpA.sub(oldlp).exp();
            const surr1 = ratio.mul(advT);
            const surr2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advT);
            const pLoss = tf.minimum(surr1, surr2).mean().mul(-1);
            const v = (this.critic.predict(S) as tf.Tensor2D).reshape([-1]);
            const vLoss = v.sub(retT).square().mean();
            return pLoss.add(vLoss.mul(this.vfCoef)).sub(entropy.mul(this.entropyCoef)) as tf.Scalar;
          });
        });
      }
    }

    return tf.tidy(() => {
      const S = tf.tensor2d(normStates);
      const { logpA, entropy } = logpAndEntropy(S, r.actions);
      const oldlp = tf.tensor1d(r.logps);
      const ratio = logpA.sub(oldlp).exp();
      const advT = tf.tensor1d(advN);
      const surr1 = ratio.mul(advT);
      const surr2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advT);
      const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1).dataSync()[0];
      const v = (this.critic.predict(S) as tf.Tensor2D).reshape([-1]);
      const valueLoss = v.sub(tf.tensor1d(returns)).square().mean().dataSync()[0];
      const approxKL = oldlp.sub(logpA).mean().dataSync()[0];
      const clipFrac = tf.abs(ratio.sub(1)).greater(clip).cast('float32').mean().dataSync()[0];
      return { policyLoss, valueLoss, entropy: entropy.dataSync()[0], approxKL, clipFrac };
    });
  }

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

export function specOf(env: SyncEnvironment): PPOActionSpec {
  const a = env.actionSpace;
  if (a.kind === 'discrete') return { kind: 'discrete', n: a.n, labels: env.actionMeanings() };
  return { kind: 'continuous', low: a.low[0], high: a.high[0], labels: env.actionMeanings() };
}

export function create(env: SyncEnvironment, hp: Record<string, number>): PPO {
  const obs = env.observationSpace;
  const inputDim = obs.kind === 'box' ? obs.shape[0] : obs.n;
  const low = obs.kind === 'box' ? obs.low : new Array<number>(inputDim).fill(-1);
  const high = obs.kind === 'box' ? obs.high : new Array<number>(inputDim).fill(1);
  return new PPO(inputDim, low, high, specOf(env), hp);
}
