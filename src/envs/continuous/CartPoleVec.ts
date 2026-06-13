// CartPole-v1 (continuous state, discrete actions) — continuous/vector variant for deep RL.
// Returns the RAW [x, ẋ, θ, θ̇] observation so a neural network consumes it directly
// (no discretization). Physics is shared with the tabular CartPole via ./physics/cartpole,
// so trajectories match exactly for the same seed + actions.
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
import type { CartPoleRenderState } from './CartPole';
import {
  CARTPOLE_MAX_STEPS,
  THETA_THRESHOLD,
  X_THRESHOLD,
  cartpoleReset,
  cartpoleStep,
  cartpoleTerminated,
  type CartPoleState,
} from './physics/cartpole';

export class CartPoleVec implements VecEnvironment {
  readonly id = 'cartpole-vec';
  readonly name = 'CartPole';
  readonly renderKind = 'cartpole';
  readonly actionSpace: DiscreteSpace = discrete(2);
  // Observation bounds used to normalize NN inputs (velocity bounds are nominal, not hard limits).
  readonly observationSpace: BoxSpace = box(
    [-X_THRESHOLD, -3, -THETA_THRESHOLD, -3.5],
    [X_THRESHOLD, 3, THETA_THRESHOLD, 3.5],
  );
  maxSteps = CARTPOLE_MAX_STEPS;

  private rng: RNG;
  private cont: CartPoleState = [0, 0, 0, 0];
  private steps = 0;
  private stepReward = 1;
  private failPenalty = 0;

  constructor(seed = 12345) {
    this.rng = new RNG(seed);
  }

  actionMeanings(): string[] {
    return ['Push left', 'Push right'];
  }

  resetSync(seed?: number): number[] {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.cont = cartpoleReset(this.rng);
    this.steps = 0;
    return [...this.cont];
  }

  stepSync(action: number): StepResult {
    this.cont = cartpoleStep(this.cont, action);
    this.steps += 1;
    const terminated = cartpoleTerminated(this.cont);
    const truncated = !terminated && this.steps >= this.maxSteps;
    const reward = this.stepReward + (terminated ? this.failPenalty : 0);
    return { observation: [...this.cont], reward, terminated, truncated };
  }

  async reset(seed?: number): Promise<number[]> {
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
    const fp =
      this.failPenalty !== 0 ? `; extra ${this.failPenalty} for falling/going out of bounds` : '';
    return `+${this.stepReward} for every step the pole stays up${fp}. Goal: balance as long as possible (up to ${this.maxSteps} steps).`;
  }

  rewardParams(): RewardParam[] {
    return [
      {
        key: 'stepReward',
        label: 'Per step upright',
        value: this.stepReward,
        min: 0,
        max: 2,
        step: 0.5,
        decimals: 1,
      },
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
