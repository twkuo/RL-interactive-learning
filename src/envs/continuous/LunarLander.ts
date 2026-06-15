// LunarLander (simplified) — 8D continuous observation, 4 discrete engine actions, for deep RL.
// The iconic showcase: land gently on the pad between the flags. Physics in ./physics/lunarlander.
import type {
  BoxSpace,
  DiscreteSpace,
  ObsField,
  RewardParam,
  StepResult,
  VecEnvironment,
} from '../../core/types';
import { box, discrete } from '../../core/spaces';
import { RNG } from '../../core/rng';
import {
  LANDER_MAX_STEPS,
  LAND_ANGLE,
  LAND_VX,
  LAND_VY,
  PAD_HALF,
  X_LIMIT,
  lunarObs,
  lunarReset,
  lunarShaping,
  lunarStep,
  type LanderState,
} from './physics/lunarlander';

export interface LunarRenderState {
  x: number;
  y: number;
  angle: number;
  lastAction: number;
  padHalf: number;
  landed: boolean;
  crashed: boolean;
}

export class LunarLander implements VecEnvironment {
  readonly id = 'lunarlander';
  readonly name = 'LunarLander';
  readonly renderKind = 'lunarlander';
  readonly actionSpace: DiscreteSpace = discrete(4);
  readonly observationSpace: BoxSpace = box(
    [-X_LIMIT, 0, -2, -2, -Math.PI, -5, 0, 0],
    [X_LIMIT, 1.5, 2, 2, Math.PI, 5, 1, 1],
  );
  maxSteps = LANDER_MAX_STEPS;

  private rng: RNG;
  private cont: LanderState = [0, 0, 0, 0, 0, 0];
  private steps = 0;
  private prevShaping: number | null = null;
  private lastAction = 0;
  private landed = false;
  private crashed = false;
  private landingReward = 100;
  private crashPenalty = 100;

  constructor(seed = 12345) {
    this.rng = new RNG(seed);
  }

  actionMeanings(): string[] {
    return ['No-op', 'Fire left engine', 'Fire main engine', 'Fire right engine'];
  }

  resetSync(seed?: number): number[] {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.cont = lunarReset(this.rng);
    this.steps = 0;
    this.prevShaping = null;
    this.lastAction = 0;
    this.landed = false;
    this.crashed = false;
    return lunarObs(this.cont);
  }

  stepSync(action: number): StepResult {
    this.cont = lunarStep(this.cont, action);
    this.lastAction = action;
    this.steps += 1;
    let [x, y] = [this.cont[0], this.cont[1]];

    let terminated = false;
    this.landed = false;
    this.crashed = false;
    if (y <= 0) {
      this.cont[1] = 0;
      y = 0;
      terminated = true;
      const gentle =
        Math.abs(this.cont[3]) < LAND_VY &&
        Math.abs(this.cont[2]) < LAND_VX &&
        Math.abs(this.cont[4]) < LAND_ANGLE;
      const onPad = Math.abs(x) < PAD_HALF;
      this.landed = gentle && onPad;
      this.crashed = !this.landed;
    } else if (Math.abs(x) > X_LIMIT) {
      terminated = true;
      this.crashed = true;
    }

    const shaping = lunarShaping(this.cont);
    let reward = this.prevShaping === null ? 0 : shaping - this.prevShaping;
    this.prevShaping = shaping;
    if (action === 2) reward -= 0.3; // main-engine fuel
    else if (action === 1 || action === 3) reward -= 0.03; // side-engine fuel
    if (this.landed) reward += this.landingReward;
    if (this.crashed) reward -= this.crashPenalty;

    const truncated = !terminated && this.steps >= this.maxSteps;
    return { observation: lunarObs(this.cont), reward, terminated, truncated };
  }

  async reset(seed?: number): Promise<number[]> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    const [x, y, vx, vy, angle] = this.cont;
    return [
      { label: 'Horizontal x (pad at 0)', value: x },
      { label: 'Altitude y', value: y },
      { label: 'Velocity vx', value: vx },
      { label: 'Velocity vy', value: vy },
      { label: 'Angle', value: (angle * 180) / Math.PI, unit: '°' },
      { label: 'Leg contact', value: y < 0.08 ? 1 : 0 },
    ];
  }

  getRenderState(): LunarRenderState {
    return {
      x: this.cont[0],
      y: this.cont[1],
      angle: this.cont[4],
      lastAction: this.lastAction,
      padHalf: PAD_HALF,
      landed: this.landed,
      crashed: this.crashed,
    };
  }

  rewardDescription(): string {
    return `Shaping toward the pad (closer, slower, upright, legs down) minus fuel; +${this.landingReward} for a gentle landing on the pad, −${this.crashPenalty} for a crash or going out of bounds.`;
  }

  rewardParams(): RewardParam[] {
    return [
      {
        key: 'landingReward',
        label: 'Landing reward',
        value: this.landingReward,
        min: 0,
        max: 200,
        step: 10,
        decimals: 0,
      },
      {
        key: 'crashPenalty',
        label: 'Crash penalty',
        value: this.crashPenalty,
        min: 0,
        max: 200,
        step: 10,
        decimals: 0,
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'landingReward') this.landingReward = value;
    else if (key === 'crashPenalty') this.crashPenalty = value;
  }
}
