// Multi-armed bandit: a single-state MDP. Each "arm" is an action whose reward is drawn from a
// hidden Gaussian. The Q(s0, a) learned by the tabular agents IS the per-arm value estimate, so
// epsilon-greedy exploration vs. exploitation shows up directly. (Chapter 3: bandit.md)
import type {
  DiscreteSpace,
  ObsField,
  RewardParam,
  StepResult,
  TabularEnvironment,
} from '../../core/types';
import { discrete } from '../../core/spaces';
import { RNG } from '../../core/rng';

export interface BanditConfig {
  id: string;
  name: string;
  arms: number;
  noiseStd: number;
  maxSteps: number;
}

export interface BanditRenderState {
  arms: number;
  trueMeans: number[]; // hidden q*(a)
  pulls: number[]; // pulls per arm this episode
  empirical: number[]; // env's own running sample mean per arm this episode
  lastArm: number; // -1 if none yet
  lastReward: number;
  totalPulls: number;
  noiseStd: number;
}

export class Bandit implements TabularEnvironment {
  readonly id: string;
  readonly name: string;
  readonly actionSpace: DiscreteSpace;
  readonly observationSpace: DiscreteSpace = discrete(1);
  maxSteps: number;

  private cfg: BanditConfig;
  private rng: RNG;
  private trueMeans: number[] = [];
  private pulls: number[];
  private sum: number[];
  private steps = 0;
  private lastArm = -1;
  private lastReward = 0;
  private noiseStd: number;
  private spare: number | null = null; // cached Box–Muller value

  constructor(cfg: BanditConfig, seed = 12345) {
    this.cfg = cfg;
    this.id = cfg.id;
    this.name = cfg.name;
    this.maxSteps = cfg.maxSteps;
    this.noiseStd = cfg.noiseStd;
    this.actionSpace = discrete(cfg.arms);
    this.rng = new RNG(seed);
    this.pulls = new Array(cfg.arms).fill(0);
    this.sum = new Array(cfg.arms).fill(0);
    this.drawTrueMeans();
  }

  // Standard normal via Box–Muller (the project RNG only provides uniform draws).
  private gaussian(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let w = 0;
    while (u === 0) u = this.rng.next();
    while (w === 0) w = this.rng.next();
    const mag = Math.sqrt(-2 * Math.log(u));
    this.spare = mag * Math.sin(2 * Math.PI * w);
    return mag * Math.cos(2 * Math.PI * w);
  }

  // Hidden arm means q*(a) ~ N(0, 1), Sutton–Barto 10-armed-testbed style. Fixed per instance.
  private drawTrueMeans(): void {
    this.spare = null;
    this.trueMeans = [];
    for (let a = 0; a < this.cfg.arms; a++) this.trueMeans.push(this.gaussian());
  }

  stateCount(): number {
    return 1;
  }

  currentState(): number {
    return 0;
  }

  actionMeanings(): string[] {
    return Array.from({ length: this.cfg.arms }, (_, a) => `Arm ${a + 1}`);
  }

  resetSync(seed?: number): number {
    // A seeded reset is a full restart (new run): redraw the hidden means and clear all stats.
    if (seed !== undefined) {
      this.rng.setSeed(seed);
      this.drawTrueMeans();
      this.pulls.fill(0);
      this.sum.fill(0);
      this.steps = 0;
      this.lastArm = -1;
      this.lastReward = 0;
    }
    // A plain (no-seed) reset just starts the next one-pull "episode"; the running per-arm estimates
    // and pull counts persist across pulls, mirroring the agent's persistent Q.
    return 0;
  }

  stepSync(action: number): StepResult {
    const a = action;
    const reward = this.trueMeans[a] + this.gaussian() * this.noiseStd;
    this.pulls[a] += 1;
    this.sum[a] += reward;
    this.steps += 1;
    this.lastArm = a;
    this.lastReward = reward;
    // Each pull is an independent one-shot decision, so terminate it: the TD target becomes just r
    // (no bootstrapping on the single state). Q(s0, a) then converges to a sample-average estimate
    // of the arm's mean reward — i.e. to the hidden true mean — exactly as a bandit should.
    return { observation: 0, reward, terminated: true, truncated: false };
  }

  async reset(seed?: number): Promise<number> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    return [
      { label: 'Total pulls', value: this.steps },
      { label: 'Last reward', value: Number(this.lastReward.toFixed(2)) },
    ];
  }

  getRenderState(): BanditRenderState {
    const empirical = this.sum.map((s, a) => (this.pulls[a] > 0 ? s / this.pulls[a] : 0));
    return {
      arms: this.cfg.arms,
      trueMeans: this.trueMeans.slice(),
      pulls: this.pulls.slice(),
      empirical,
      lastArm: this.lastArm,
      lastReward: this.lastReward,
      totalPulls: this.steps,
      noiseStd: this.noiseStd,
    };
  }

  rewardDescription(): string {
    return `Each pull of arm a returns a reward drawn from N(q*(a), ${this.noiseStd.toFixed(
      1,
    )}²). The arm means q*(a) are hidden — estimate them by pulling.`;
  }

  rewardParams(): RewardParam[] {
    return [
      {
        key: 'noiseStd',
        label: 'Reward noise (std)',
        value: this.noiseStd,
        min: 0,
        max: 2,
        step: 0.1,
        decimals: 1,
        hint: 'Higher noise → each pull is less informative, so more exploration is needed to tell arms apart.',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'noiseStd') this.noiseStd = value;
  }
}
