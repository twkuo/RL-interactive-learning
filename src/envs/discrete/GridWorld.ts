// Generic grid world. FrozenLake is just one of its configurations.
// Action convention (consistent with Gymnasium FrozenLake): 0=Left, 1=Down, 2=Right, 3=Up.
import type {
  DiscreteSpace,
  GridInfo,
  ModeledEnvironment,
  ObsField,
  RewardParam,
  StepResult,
  TransitionOutcome,
} from '../../core/types';
import { discrete } from '../../core/spaces';
import { RNG } from '../../core/rng';

export interface GridWorldConfig {
  id: string;
  name: string;
  rows: number;
  cols: number;
  start: number;
  goals: number[];
  holes: number[];
  walls: number[];
  stepReward: number;
  goalReward: number;
  holeReward: number;
  slip: number; // 0=deterministic; 2/3=FrozenLake-style slip
  maxSteps: number;
}

// 0=Left, 1=Down, 2=Right, 3=Up -> [dCol, dRow]
const DELTAS: Array<[number, number]> = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];
// Perpendicular (left/right) directions for each action (used when slipping)
const PERP: number[][] = [
  [1, 3], // Left -> Down/Up
  [0, 2], // Down -> Left/Right
  [1, 3], // Right -> Down/Up
  [0, 2], // Up -> Left/Right
];

const ACTION_NAMES = ['Left', 'Down', 'Right', 'Up'];

export class GridWorld implements ModeledEnvironment {
  readonly id: string;
  readonly name: string;
  readonly actionSpace: DiscreteSpace = discrete(4);
  readonly observationSpace: DiscreteSpace;
  maxSteps: number;
  readonly grid: GridInfo;

  private cfg: GridWorldConfig;
  private cur: number;
  private steps = 0;
  private rng: RNG;
  private stepReward: number;
  private goalReward: number;
  private holeReward: number;

  constructor(cfg: GridWorldConfig, seed = 12345) {
    this.cfg = cfg;
    this.id = cfg.id;
    this.name = cfg.name;
    this.maxSteps = cfg.maxSteps;
    this.stepReward = cfg.stepReward;
    this.goalReward = cfg.goalReward;
    this.holeReward = cfg.holeReward;
    this.observationSpace = discrete(cfg.rows * cfg.cols);
    this.cur = cfg.start;
    this.rng = new RNG(seed);
    this.grid = {
      rows: cfg.rows,
      cols: cfg.cols,
      start: cfg.start,
      goals: cfg.goals,
      holes: cfg.holes,
      walls: cfg.walls,
      actionDeltas: DELTAS,
    };
  }

  stateCount(): number {
    return this.cfg.rows * this.cfg.cols;
  }

  currentState(): number {
    return this.cur;
  }

  isTerminal(s: number): boolean {
    return this.cfg.goals.includes(s) || this.cfg.holes.includes(s);
  }

  actionMeanings(): string[] {
    return ACTION_NAMES;
  }

  // Without considering slip, which cell does taking action a from s land on (stay put if hitting a wall / going out of bounds).
  private move(s: number, a: number): number {
    const { cols, rows } = this.cfg;
    const row = Math.floor(s / cols);
    const col = s % cols;
    const [dc, dr] = DELTAS[a];
    const nr = row + dr;
    const nc = col + dc;
    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) return s;
    const ns = nr * cols + nc;
    if (this.cfg.walls.includes(ns)) return s;
    return ns;
  }

  transitions(s: number, a: number): TransitionOutcome[] {
    if (this.isTerminal(s)) {
      return [{ prob: 1, nextState: s, reward: 0, done: true }];
    }
    // Intended direction + (when slipping) the two perpendicular directions
    const dirs: Array<{ a: number; p: number }> = [];
    if (this.cfg.slip > 0) {
      dirs.push({ a, p: 1 - this.cfg.slip });
      dirs.push({ a: PERP[a][0], p: this.cfg.slip / 2 });
      dirs.push({ a: PERP[a][1], p: this.cfg.slip / 2 });
    } else {
      dirs.push({ a, p: 1 });
    }
    // Aggregate probabilities by landing cell
    const probByState = new Map<number, number>();
    for (const d of dirs) {
      const ns = this.move(s, d.a);
      probByState.set(ns, (probByState.get(ns) ?? 0) + d.p);
    }
    const out: TransitionOutcome[] = [];
    probByState.forEach((p, ns) => {
      const isGoal = this.cfg.goals.includes(ns);
      const isHole = this.cfg.holes.includes(ns);
      const reward = isGoal ? this.goalReward : isHole ? this.holeReward : this.stepReward;
      out.push({ prob: p, nextState: ns, reward, done: isGoal || isHole });
    });
    return out;
  }

  resetSync(seed?: number): number {
    if (seed !== undefined) this.rng.setSeed(seed);
    this.cur = this.cfg.start;
    this.steps = 0;
    return this.cur;
  }

  stepSync(action: number): StepResult {
    const outcomes = this.transitions(this.cur, action);
    // Sample a landing cell according to the probabilities
    let r = this.rng.next();
    let chosen = outcomes[outcomes.length - 1];
    for (const o of outcomes) {
      if (r < o.prob) {
        chosen = o;
        break;
      }
      r -= o.prob;
    }
    this.cur = chosen.nextState;
    this.steps += 1;
    const truncated = !chosen.done && this.steps >= this.maxSteps;
    return {
      observation: chosen.nextState,
      reward: chosen.reward,
      terminated: chosen.done,
      truncated,
    };
  }

  async reset(seed?: number): Promise<number> {
    return this.resetSync(seed);
  }

  async step(action: number): Promise<StepResult> {
    return this.stepSync(action);
  }

  describeObs(obs: number): ObsField[] {
    const { cols } = this.cfg;
    return [
      { label: 'State s', value: obs },
      { label: 'Row', value: Math.floor(obs / cols) },
      { label: 'Col', value: obs % cols },
    ];
  }

  getRenderState(): unknown {
    return { cur: this.cur };
  }

  rewardDescription(): string {
    const f = (x: number) => (x >= 0 ? `+${x}` : `${x}`);
    return `Each step ${f(this.stepReward)}; reaching the goal ${f(this.goalReward)}; falling into a trap ${f(
      this.holeReward,
    )}.`;
  }

  rewardParams(): RewardParam[] {
    return [
      {
        key: 'stepReward',
        label: 'Step cost',
        value: this.stepReward,
        min: -1,
        max: 0.5,
        step: 0.01,
        decimals: 2,
        hint: 'Usually a small negative value, encouraging the agent to take the shortest path; a positive value makes it stall.',
      },
      { key: 'goalReward', label: 'Goal reward', value: this.goalReward, min: 0, max: 5, step: 0.5, decimals: 1 },
      {
        key: 'holeReward',
        label: 'Trap penalty',
        value: this.holeReward,
        min: -10,
        max: 1,
        step: 0.5,
        decimals: 1,
        hint: 'The more negative -> the more the agent avoids traps.',
      },
    ];
  }

  setRewardParam(key: string, value: number): void {
    if (key === 'stepReward') this.stepReward = value;
    else if (key === 'goalReward') this.goalReward = value;
    else if (key === 'holeReward') this.holeReward = value;
  }
}
