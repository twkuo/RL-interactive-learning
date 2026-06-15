// Pendulum-v1 (continuous state AND continuous action) for deep RL — the PPO Gaussian-policy
// showcase. 3D observation [cos θ, sin θ, θ̇]; a single torque action in [-2, 2]. Physics in
// ./physics/pendulum. No termination; the goal is to swing up and balance (θ = 0, upright).
import type {
  BoxSpace,
  ObsField,
  RewardParam,
  StepResult,
  VecEnvironment,
} from '../../core/types';
import { box } from '../../core/spaces';
import { RNG } from '../../core/rng';
import {
  MAX_SPEED,
  MAX_TORQUE,
  PENDULUM_MAX_STEPS,
  angleNormalize,
  pendulumObs,
  pendulumReset,
  pendulumStep,
  type PendulumState,
} from './physics/pendulum';

export interface PendulumRenderState {
  theta: number;
  torque: number;
  maxTorque: number;
}

export class Pendulum implements VecEnvironment {
  readonly id = 'pendulum';
  readonly name = 'Pendulum';
  readonly renderKind = 'pendulum';
  readonly actionSpace: BoxSpace = box([-MAX_TORQUE], [MAX_TORQUE]);
  readonly observationSpace: BoxSpace = box([-1, -1, -MAX_SPEED], [1, 1, MAX_SPEED]);
  maxSteps = PENDULUM_MAX_STEPS;

  private rng: RNG;
  private cont: PendulumState = [Math.PI, 0];
  private steps = 0;
  private lastTorque = 0;
  private velocityCost = 0.1;
  private torqueCost = 0.001;

  constructor(seed = 12345) {
    this.rng = new RNG(seed);
  }

  actionMeanings(): string[] {
    return ['Torque'];
  }

  resetSync(seed?: number): number[] {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.cont = pendulumReset(this.rng);
    this.steps = 0;
    this.lastTorque = 0;
    return pendulumObs(this.cont);
  }

  stepSync(action: number): StepResult {
    const { state, u, th, thdot } = pendulumStep(this.cont, action);
    this.cont = state;
    this.lastTorque = u;
    this.steps += 1;
    const angle = angleNormalize(th);
    const cost = angle * angle + this.velocityCost * thdot * thdot + this.torqueCost * u * u;
    const truncated = this.steps >= this.maxSteps;
    return { observation: pendulumObs(this.cont), reward: -cost, terminated: false, truncated };
  }

  async reset(seed?: number): Promise<number[]> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(): ObsField[] {
    const [th, thdot] = this.cont;
    return [
      { label: 'Angle from upright', value: (angleNormalize(th) * 180) / Math.PI, unit: '°' },
      { label: 'Angular velocity θ̇', value: thdot, unit: 'rad/s' },
      { label: 'Uprightness cos θ (goal → 1)', value: Math.cos(th) },
      { label: 'Last torque', value: this.lastTorque },
    ];
  }

  getRenderState(): PendulumRenderState {
    return { theta: this.cont[0], torque: this.lastTorque, maxTorque: MAX_TORQUE };
  }

  rewardDescription(): string {
    return `Cost each step = angle² + ${this.velocityCost}·θ̇² + ${this.torqueCost}·torque² (reward = −cost; 0 is perfectly balanced and still). Swing up and stay upright.`;
  }

  rewardParams(): RewardParam[] {
    return [
      {
        key: 'velocityCost',
        label: 'Velocity cost',
        value: this.velocityCost,
        min: 0,
        max: 0.5,
        step: 0.05,
        decimals: 2,
        hint: 'Penalizes spinning fast; higher → calmer, steadier balancing.',
      },
      {
        key: 'torqueCost',
        label: 'Torque (energy) cost',
        value: this.torqueCost,
        min: 0,
        max: 0.02,
        step: 0.001,
        decimals: 3,
        hint: 'Penalizes large torques; higher → more energy-efficient swing-ups.',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'velocityCost') this.velocityCost = value;
    else if (key === 'torqueCost') this.torqueCost = value;
  }
}
