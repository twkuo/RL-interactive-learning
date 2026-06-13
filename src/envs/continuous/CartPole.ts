// CartPole-v1 (continuous state, discrete actions). Physics matches Gymnasium 1:1.
// Runs continuous physics internally; returns the discretized state index to the agent, and provides the continuous state to the renderer.
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

const GRAVITY = 9.8;
const MASSCART = 1.0;
const MASSPOLE = 0.1;
const TOTAL_MASS = MASSCART + MASSPOLE;
const LENGTH = 0.5; // half pole length
const POLEMASS_LENGTH = MASSPOLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU = 0.02;
const THETA_THRESHOLD = (12 * 2 * Math.PI) / 360; // ±12°
const X_THRESHOLD = 2.4;
const MAX_STEPS = 500;

export interface CartPoleRenderState {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
  xThreshold: number;
  thetaThreshold: number;
}

export class CartPole implements TabularEnvironment {
  readonly id = 'cartpole';
  readonly name = 'CartPole';
  readonly renderKind = 'cartpole';
  readonly actionSpace: DiscreteSpace = discrete(2);
  readonly observationSpace: DiscreteSpace;
  maxSteps = MAX_STEPS;

  private disc: Discretizer;
  private rng: RNG;
  private cont: [number, number, number, number] = [0, 0, 0, 0]; // x, xDot, theta, thetaDot
  private steps = 0;
  private cur = 0;
  private stepReward = 1;
  private failPenalty = 0;

  constructor(seed = 12345) {
    this.disc = new Discretizer([
      { low: -X_THRESHOLD, high: X_THRESHOLD, bins: 3 },
      { low: -3, high: 3, bins: 3 },
      { low: -THETA_THRESHOLD, high: THETA_THRESHOLD, bins: 6 },
      { low: -3.5, high: 3.5, bins: 6 },
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
    return ['Push left', 'Push right'];
  }

  resetSync(seed?: number): number {
    if (seed !== undefined) this.rng.setSeed(seed);
    const u = () => (this.rng.next() - 0.5) * 0.1; // U(-0.05, 0.05)
    this.cont = [u(), u(), u(), u()];
    this.steps = 0;
    this.cur = this.disc.index(this.cont);
    return this.cur;
  }

  stepSync(action: number): StepResult {
    let [x, xDot, theta, thetaDot] = this.cont;
    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const temp = (force + POLEMASS_LENGTH * thetaDot * thetaDot * sinT) / TOTAL_MASS;
    const thetaAcc =
      (GRAVITY * sinT - cosT * temp) /
      (LENGTH * (4.0 / 3.0 - (MASSPOLE * cosT * cosT) / TOTAL_MASS));
    const xAcc = temp - (POLEMASS_LENGTH * thetaAcc * cosT) / TOTAL_MASS;
    // Explicit Euler (update position using the old velocity first)
    x += TAU * xDot;
    xDot += TAU * xAcc;
    theta += TAU * thetaDot;
    thetaDot += TAU * thetaAcc;
    this.cont = [x, xDot, theta, thetaDot];
    this.steps += 1;

    const terminated =
      x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD;
    const truncated = !terminated && this.steps >= this.maxSteps;
    this.cur = this.disc.index(this.cont);
    const reward = this.stepReward + (terminated ? this.failPenalty : 0);
    return { observation: this.cur, reward, terminated, truncated };
  }

  async reset(seed?: number): Promise<number> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    const [x, xDot, theta, thetaDot] = this.cont;
    return [
      { label: 'Position x', value: x, unit: 'm' },
      { label: 'Velocity ẋ', value: xDot, unit: 'm/s' },
      { label: 'Angle θ', value: (theta * 180) / Math.PI, unit: '°' },
      { label: 'Angular velocity θ̇', value: thetaDot, unit: 'rad/s' },
    ];
  }

  getRenderState(): CartPoleRenderState {
    return {
      x: this.cont[0],
      xDot: this.cont[1],
      theta: this.cont[2],
      thetaDot: this.cont[3],
      xThreshold: X_THRESHOLD,
      thetaThreshold: THETA_THRESHOLD,
    };
  }

  rewardDescription(): string {
    const fp = this.failPenalty !== 0 ? `; extra ${this.failPenalty} for falling/going out of bounds` : '';
    return `+${this.stepReward} for every step the pole stays up${fp}. Goal: balance as long as possible (up to ${this.maxSteps} steps).`;
  }

  rewardParams(): RewardParam[] {
    return [
      { key: 'stepReward', label: 'Per step upright', value: this.stepReward, min: 0, max: 2, step: 0.5, decimals: 1 },
      {
        key: 'failPenalty',
        label: 'Fall/out-of-bounds penalty',
        value: this.failPenalty,
        min: -50,
        max: 0,
        step: 1,
        decimals: 0,
        hint: 'Added on the "termination" step; a negative value more strongly discourages falling.',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'stepReward') this.stepReward = value;
    else if (key === 'failPenalty') this.failPenalty = value;
  }
}
