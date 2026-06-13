// MountainCar-v0 (continuous state, discrete actions). Physics matches Gymnasium 1:1.
// Reward is −1 per step; the car must build momentum by rocking back and forth to reach the flag on the right; exploration is hard (optimistic initial Q=0 helps).
import type {
  DiscreteSpace,
  ObsField,
  RewardParam,
  StepResult,
  TabularEnvironment,
} from '../../core/types';
import { discrete } from '../../core/spaces';
import { RNG } from '../../core/rng';
import { Discretizer } from '../discretize';

// Hill height (consistent with the rendering): used for the optional "position shaping" reward.
const height = (x: number) => Math.sin(3 * x) * 0.45 + 0.55;

const MIN_POS = -1.2;
const MAX_POS = 0.6;
const MAX_SPEED = 0.07;
const GOAL = 0.5;
const FORCE = 0.001;
const GRAVITY = 0.0025;
const MAX_STEPS = 200;

export interface MountainCarRenderState {
  position: number;
  velocity: number;
  minPos: number;
  maxPos: number;
  goal: number;
}

export class MountainCar implements TabularEnvironment {
  readonly id = 'mountaincar';
  readonly name = 'MountainCar';
  readonly renderKind = 'mountaincar';
  readonly actionSpace: DiscreteSpace = discrete(3);
  readonly observationSpace: DiscreteSpace;
  maxSteps = MAX_STEPS;

  private disc: Discretizer;
  private rng: RNG;
  private pos = 0;
  private vel = 0;
  private steps = 0;
  private cur = 0;
  private stepReward = -1;
  private goalReward = 0;
  private shapingCoef = 0;

  constructor(seed = 12345) {
    this.disc = new Discretizer([
      { low: MIN_POS, high: MAX_POS, bins: 18 },
      { low: -MAX_SPEED, high: MAX_SPEED, bins: 14 },
    ]);
    this.observationSpace = discrete(this.disc.stateCount);
    this.rng = new RNG(seed);
  }

  stateCount(): number {
    return this.disc.stateCount;
  }

  currentState(): number {
    return this.cur;
  }

  actionMeanings(): string[] {
    return ['Push left', 'No-op', 'Push right'];
  }

  resetSync(seed?: number): number {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.pos = -0.6 + this.rng.next() * 0.2; // U(-0.6, -0.4)
    this.vel = 0;
    this.steps = 0;
    this.cur = this.disc.index([this.pos, this.vel]);
    return this.cur;
  }

  stepSync(action: number): StepResult {
    this.vel += (action - 1) * FORCE - Math.cos(3 * this.pos) * GRAVITY;
    this.vel = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, this.vel));
    this.pos += this.vel;
    this.pos = Math.max(MIN_POS, Math.min(MAX_POS, this.pos));
    if (this.pos === MIN_POS && this.vel < 0) this.vel = 0;
    this.steps += 1;

    const terminated = this.pos >= GOAL;
    const truncated = !terminated && this.steps >= this.maxSteps;
    this.cur = this.disc.index([this.pos, this.vel]);
    const reward =
      this.stepReward + this.shapingCoef * height(this.pos) + (terminated ? this.goalReward : 0);
    return { observation: this.cur, reward, terminated, truncated };
  }

  async reset(seed?: number): Promise<number> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    return [
      { label: 'Position x', value: this.pos, unit: 'm' },
      { label: 'Velocity v', value: this.vel, unit: 'm/s' },
    ];
  }

  getRenderState(): MountainCarRenderState {
    return { position: this.pos, velocity: this.vel, minPos: MIN_POS, maxPos: MAX_POS, goal: GOAL };
  }

  rewardDescription(): string {
    const parts = [`Per step ${this.stepReward}`];
    if (this.goalReward !== 0) parts.push(`extra +${this.goalReward} at the summit`);
    if (this.shapingCoef !== 0) parts.push(`position shaping ×${this.shapingCoef} (higher is better)`);
    return parts.join('; ') + '. The default sparse reward is hard to learn; add shaping to aid exploration.';
  }

  rewardParams(): RewardParam[] {
    return [
      { key: 'stepReward', label: 'Per step', value: this.stepReward, min: -2, max: 0, step: 0.5, decimals: 1 },
      { key: 'goalReward', label: 'Summit reward', value: this.goalReward, min: 0, max: 50, step: 1, decimals: 0 },
      {
        key: 'shapingCoef',
        label: 'Shaping coefficient',
        value: this.shapingCoef,
        min: 0,
        max: 3,
        step: 0.1,
        decimals: 1,
        hint: 'Reward shaping: >0 gives more reward the higher/further-right the car is, greatly aiding exploration. Try 1–2!',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'stepReward') this.stepReward = value;
    else if (key === 'goalReward') this.goalReward = value;
    else if (key === 'shapingCoef') this.shapingCoef = value;
  }
}
