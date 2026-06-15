// Deep Q-Network (DQN) with experience replay, a target network, and the Double-DQN target.
//
// This module statically imports TensorFlow.js. It must therefore only ever be reached via a
// DYNAMIC import (the algo registry does `() => import('./deep/DQN')`, and the training worker
// imports it in its own chunk) so that tfjs stays OUT of the main bundle — the tabular
// experience loads tfjs-free.
//
// Two roles for one class:
//  - Main thread: holds the online model for SYNCHRONOUS inference in act() — drives the
//    step-by-step DecisionPanel (Q(s,·) from a forward pass). update() is a no-op; heavy
//    training never runs here.
//  - Worker: the same class runs the full training loop (pushTransition + trainStep + target
//    sync + ε decay), then ships its weights back to the main thread via dumpWeights().
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
import { argmax } from '../../core/utils';
import { buildMLP } from '../../core/nn/mlp';
import { ReplayBuffer } from '../../core/nn/replayBuffer';
import { dumpWeights, loadWeights, type WeightDump } from '../../core/nn/weights';

export class DQN implements Agent {
  readonly id = 'dqn';
  readonly name = 'DQN';
  readonly usesQ = true;
  hyperparams: Record<string, number>;

  private inputDim: number;
  private nActions: number;
  private actionLabels: string[];
  private rng: RNG;
  private obsCenter: number[]; // observation normalization: (s - center) / half  ->  ~[-1, 1]
  private obsHalf: number[];
  private normalize: boolean; // toggleable: off feeds the network raw observations (teaching demo)

  // hyperparameters (read once at construction; live edits apply to the next training run)
  private lr: number;
  private gamma: number;
  private epsilonStart: number;
  private epsilonMin: number;
  private epsilonDecay: number;
  private hiddenUnits: number;
  private batchSize: number;
  private bufferSize: number;
  private targetSync: number;

  // mutable training state
  private epsilon: number;
  private trainSteps = 0;
  private online!: tf.Sequential;
  private target!: tf.Sequential;
  private optimizer!: tf.Optimizer;
  private replay!: ReplayBuffer;

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

    this.lr = hp.lr ?? 0.001;
    this.gamma = hp.gamma ?? 0.99;
    this.epsilonStart = hp.epsilon ?? 1.0;
    this.epsilonMin = hp.epsilonMin ?? 0.05;
    this.epsilonDecay = hp.epsilonDecay ?? 0.97;
    this.hiddenUnits = Math.round(hp.hiddenUnits ?? 64);
    this.batchSize = Math.round(hp.batchSize ?? 64);
    this.bufferSize = Math.round(hp.bufferSize ?? 10000);
    this.targetSync = Math.round(hp.targetSync ?? 500);

    this.epsilon = this.epsilonStart;
    this.build();
  }

  private build(): void {
    const mk = () =>
      buildMLP({ inputDim: this.inputDim, hidden: [this.hiddenUnits, this.hiddenUnits], outputDim: this.nActions });
    this.online = mk();
    this.target = mk();
    this.target.setWeights(this.online.getWeights());
    this.optimizer = tf.train.adam(this.lr);
    this.replay = new ReplayBuffer(this.bufferSize, () => this.rng.next());
  }

  // Normalize a raw observation to ~[-1, 1] so the network sees balanced input scales.
  // When normalization is toggled off, the network gets the raw observation (a teaching demo).
  private norm(s: number[]): number[] {
    if (!this.normalize) return s;
    const out = new Array<number>(s.length);
    for (let i = 0; i < s.length; i++) out[i] = (s[i] - this.obsCenter[i]) / this.obsHalf[i];
    return out;
  }

  // ---- inference (main thread) ----
  private forwardQ(state: number[]): number[] {
    return tf.tidy(() => {
      const out = this.online.predict(tf.tensor2d([this.norm(state)], [1, this.inputDim])) as tf.Tensor2D;
      return Array.from(out.dataSync());
    });
  }

  selectAction(state: Obs, greedy = false): Action {
    const q = this.forwardQ(state as number[]);
    if (!greedy && this.rng.next() < this.epsilon) return this.rng.int(this.nActions);
    return argmax(q);
  }

  act(state: Obs, greedy = false): { action: Action; explanation: ActionExplanation } {
    const s = state as number[];
    const q = this.forwardQ(s);
    const greedyAction = argmax(q);
    const draw = this.rng.next();
    const exploring = !greedy && draw < this.epsilon;
    const action = exploring ? this.rng.int(this.nActions) : greedyAction;
    const rationale = exploring
      ? `Explore: random draw ${draw.toFixed(2)} < ε ${this.epsilon.toFixed(2)} → random action "${this.actionLabels[action]}"`
      : `Exploit: pick the greedy action "${this.actionLabels[greedyAction]}" with the highest Q (${q[greedyAction].toFixed(2)})`;
    return {
      action,
      explanation: {
        state: s,
        actionMeanings: this.actionLabels,
        policyKind: 'epsilon-greedy',
        qValues: q,
        greedyAction,
        epsilon: this.epsilon,
        randomDraw: draw,
        isExploring: exploring,
        chosenAction: action,
        rationale,
      },
    };
  }

  // Interactive single-step never trains the network (that happens in the worker).
  update(_t: Transition): UpdateInfo {
    return {};
  }

  onEpisodeEnd(): UpdateInfo | undefined {
    return undefined;
  }

  // ---- training (worker) ----
  pushTransition(s: number[], a: number, r: number, s2: number[], terminated: boolean): void {
    this.replay.push({ s, a, r, s2, done: terminated ? 1 : 0 });
  }

  get replaySize(): number {
    return this.replay.size;
  }

  get bufferCapacity(): number {
    return this.bufferSize;
  }

  get epsilonValue(): number {
    return this.epsilon;
  }

  get stepsTrained(): number {
    return this.trainSteps;
  }

  get targetSyncEvery(): number {
    return this.targetSync;
  }

  // One minibatch gradient step on the Huber TD loss. Returns null until the buffer has a batch.
  trainStep(): { loss: number; tdError: number } | null {
    if (this.replay.size < this.batchSize) return null;
    const batch = this.replay.sample(this.batchSize);
    const sArr = batch.map((b) => this.norm(b.s));
    const s2Arr = batch.map((b) => this.norm(b.s2));
    const aArr = batch.map((b) => b.a);
    const rArr = batch.map((b) => b.r);
    const dArr = batch.map((b) => b.done);

    const result = tf.tidy(() => {
      const sT = tf.tensor2d(sArr);
      const s2T = tf.tensor2d(s2Arr);
      const aOneHot = tf.oneHot(tf.tensor1d(aArr, 'int32'), this.nActions);

      // Double-DQN target (constant w.r.t. the gradient tape — computed outside minimize):
      // a* = argmax_a online(s2);  y = r + γ · target(s2)[a*] · (1 − done)
      const aStar = (this.online.predict(s2T) as tf.Tensor2D).argMax(1);
      const targetNext = this.target.predict(s2T) as tf.Tensor2D;
      const targetQ = targetNext.mul(tf.oneHot(aStar, this.nActions)).sum(1);
      const notDone = tf.scalar(1).sub(tf.tensor1d(dArr));
      const y = tf.tensor1d(rArr).add(targetQ.mul(notDone).mul(this.gamma)); // [B]

      const lossScalar = this.optimizer.minimize(() => {
        const qAll = this.online.predict(sT) as tf.Tensor2D;
        const qAtA = qAll.mul(aOneHot).sum(1); // [B]
        return tf.losses.huberLoss(y, qAtA) as tf.Scalar;
      }, true) as tf.Scalar;

      // Report mean |TD error| for the dashboard histogram.
      const qNow = (this.online.predict(sT) as tf.Tensor2D).mul(aOneHot).sum(1);
      const tdError = y.sub(qNow).abs().mean().dataSync()[0];
      return { loss: lossScalar.dataSync()[0], tdError };
    });

    this.trainSteps += 1;
    if (this.trainSteps % this.targetSync === 0) this.syncTarget();
    return result;
  }

  decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  private syncTarget(): void {
    this.target.setWeights(this.online.getWeights());
  }

  // ---- weight transfer (worker → main thread) ----
  dumpWeights(): WeightDump[] {
    return [dumpWeights(this.online)];
  }

  loadWeightDump(dumps: WeightDump[]): void {
    loadWeights(this.online, dumps[0]);
    this.syncTarget();
  }

  setEpsilon(e: number): void {
    this.epsilon = e;
  }

  reset(): void {
    this.disposeModels();
    this.epsilon = this.epsilonStart;
    this.trainSteps = 0;
    this.build();
  }

  private disposeModels(): void {
    this.online.dispose();
    this.target.dispose();
    this.optimizer.dispose();
  }

  dispose(): void {
    this.disposeModels();
  }
}

// Factory used by the algo registry (main-thread interactive agent).
export function create(env: SyncEnvironment, hp: Record<string, number>): DQN {
  const obs = env.observationSpace;
  const inputDim = obs.kind === 'box' ? obs.shape[0] : obs.n;
  const nActions = env.actionSpace.kind === 'discrete' ? env.actionSpace.n : 1;
  const low = obs.kind === 'box' ? obs.low : new Array<number>(inputDim).fill(-1);
  const high = obs.kind === 'box' ? obs.high : new Array<number>(inputDim).fill(1);
  return new DQN(inputDim, nActions, env.actionMeanings(), hp, low, high);
}
