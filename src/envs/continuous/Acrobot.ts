// Acrobot-v1 (continuous state, 3 discrete torque actions) for deep RL. Returns the raw 6D
// observation [cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2]. Physics in ./physics/acrobot.
// Classic showcase for function approximation: a 6D state where tabular discretization explodes.
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
  ACROBOT_MAX_STEPS,
  LINK_LENGTH_1,
  LINK_LENGTH_2,
  MAX_VEL_1,
  MAX_VEL_2,
  acrobotObs,
  acrobotReset,
  acrobotStep,
  acrobotTerminated,
  type AcrobotState,
} from './physics/acrobot';

export interface AcrobotRenderState {
  theta1: number;
  theta2: number;
  l1: number;
  l2: number;
}

export class Acrobot implements VecEnvironment {
  readonly id = 'acrobot';
  readonly name = 'Acrobot';
  readonly renderKind = 'acrobot';
  readonly actionSpace: DiscreteSpace = discrete(3);
  readonly observationSpace: BoxSpace = box(
    [-1, -1, -1, -1, -MAX_VEL_1, -MAX_VEL_2],
    [1, 1, 1, 1, MAX_VEL_1, MAX_VEL_2],
  );
  maxSteps = ACROBOT_MAX_STEPS;

  private rng: RNG;
  private cont: AcrobotState = [0, 0, 0, 0];
  private steps = 0;
  private stepReward = -1;
  private goalReward = 0;

  constructor(seed = 12345) {
    this.rng = new RNG(seed);
  }

  actionMeanings(): string[] {
    return ['Torque −1', 'No torque', 'Torque +1'];
  }

  resetSync(seed?: number): number[] {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.cont = acrobotReset(this.rng);
    this.steps = 0;
    return acrobotObs(this.cont);
  }

  stepSync(action: number): StepResult {
    this.cont = acrobotStep(this.cont, action);
    this.steps += 1;
    const terminated = acrobotTerminated(this.cont);
    const truncated = !terminated && this.steps >= this.maxSteps;
    const reward = terminated ? this.goalReward : this.stepReward;
    return { observation: acrobotObs(this.cont), reward, terminated, truncated };
  }

  async reset(seed?: number): Promise<number[]> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    const [th1, th2, dth1, dth2] = this.cont;
    const tip = -Math.cos(th1) - Math.cos(th1 + th2);
    return [
      { label: 'Angle θ₁', value: (th1 * 180) / Math.PI, unit: '°' },
      { label: 'Angle θ₂', value: (th2 * 180) / Math.PI, unit: '°' },
      { label: 'Angular velocity θ̇₁', value: dth1, unit: 'rad/s' },
      { label: 'Angular velocity θ̇₂', value: dth2, unit: 'rad/s' },
      { label: 'Tip height (goal > 1)', value: tip },
    ];
  }

  getRenderState(): AcrobotRenderState {
    return { theta1: this.cont[0], theta2: this.cont[1], l1: LINK_LENGTH_1, l2: LINK_LENGTH_2 };
  }

  rewardDescription(): string {
    return `${this.stepReward} per step until the tip swings above the bar (then ${this.goalReward}). Goal: swing up in as few steps as possible (up to ${this.maxSteps}).`;
  }

  rewardParams(): RewardParam[] {
    return [
      { key: 'stepReward', label: 'Per step', value: this.stepReward, min: -2, max: 0, step: 0.5, decimals: 1 },
      {
        key: 'goalReward',
        label: 'Reward at the goal',
        value: this.goalReward,
        min: 0,
        max: 10,
        step: 1,
        decimals: 0,
        hint: 'Added on the step the tip reaches the bar; >0 more strongly rewards swinging up.',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'stepReward') this.stepReward = value;
    else if (key === 'goalReward') this.goalReward = value;
  }
}
