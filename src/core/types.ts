// Core abstraction layer: spaces, environments, agents, action breakdown. The foundation of the whole architecture.
// Design rules:
//  1. Keep terminated and truncated separate (bootstrapping only zeroes out the successor value on terminated).
//  2. Interface methods return Promise (remote must be async); local environments additionally implement a synchronous stepSync fast path.
//  3. Abstract out QFunction so that tabular Q and a future tile-coding linear Q are interchangeable.

// ---------- Spaces ----------
export type DiscreteSpace = { kind: 'discrete'; n: number };
export type BoxSpace = { kind: 'box'; low: number[]; high: number[]; shape: number[] };
export type Space = DiscreteSpace | BoxSpace;

export type Obs = number | number[]; // discrete = state index; continuous = vector
export type Action = number; // all actions in this project are discrete

export interface StepResult {
  observation: Obs;
  reward: number;
  terminated: boolean; // MDP natural termination (reached goal / fell in hole / pole fell)
  truncated: boolean; // truncated upon reaching maxSteps
  info?: Record<string, unknown>;
}

export interface ObsField {
  label: string;
  value: number;
  unit?: string;
}

// Editable reward parameters (reward design)
export interface RewardParam {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  hint?: string;
}

export interface Environment {
  readonly id: string;
  readonly name: string;
  readonly actionSpace: Space;
  readonly observationSpace: Space;
  reset(seed?: number): Promise<Obs>;
  step(action: Action): Promise<StepResult>;
  describeObs(obs: Obs): ObsField[]; // used by StatePanel for decoding
  actionMeanings(): string[];
  getRenderState(): unknown; // structured state used exclusively by the renderer
  // Reward description and editable parameters (reward design) — optional
  rewardDescription?(): string;
  rewardParams?(): RewardParam[];
  setRewardParam?(key: string, value: number): void;
}

// Synchronous fast path for local environments (batch training takes this route)
export interface SyncEnvironment extends Environment {
  resetSync(seed?: number): Obs;
  stepSync(action: Action): StepResult;
}

export function isSyncEnv(e: Environment): e is SyncEnvironment {
  return typeof (e as Partial<SyncEnvironment>).stepSync === 'function';
}

// Static information about the grid environment, used by the renderer for coloring
export interface GridInfo {
  rows: number;
  cols: number;
  start: number;
  goals: number[];
  holes: number[]; // traps / holes
  walls: number[];
  // action index → direction [dx, dy] (dx is column displacement, dy is row displacement, downward is positive)
  actionDeltas: Array<[number, number]>;
}

export interface TabularEnvironment extends SyncEnvironment {
  readonly actionSpace: DiscreteSpace;
  readonly observationSpace: DiscreteSpace;
  maxSteps: number; // adjustable per-episode step limit (reaching it triggers truncated)
  stateCount(): number;
  grid?: GridInfo;
  currentState(): number;
}

// Continuous-observation environment for deep-RL (function approximation).
// Returns the RAW observation vector (number[]) instead of a discretized index, so a
// neural network can consume the continuous state directly. Reuses the same synchronous
// fast path; deep agents train on number[] observations.
export interface VecEnvironment extends SyncEnvironment {
  readonly actionSpace: DiscreteSpace | BoxSpace; // discrete for DQN / discrete PPO; box for continuous action (Pendulum)
  readonly observationSpace: BoxSpace;
  maxSteps: number;
  resetSync(seed?: number): number[];
  stepSync(action: Action): StepResult; // observation is number[]
}

export function isVecEnv(e: Environment): e is VecEnvironment {
  return isSyncEnv(e) && e.observationSpace.kind === 'box';
}

// Any interactive (synchronous) environment the store can drive: tabular (discrete index) or
// vector (continuous observation). Both expose maxSteps + resetSync/stepSync.
export type AnyEnv = TabularEnvironment | VecEnvironment;

export interface TransitionOutcome {
  prob: number;
  nextState: number;
  reward: number;
  done: boolean;
}

// For DP: an enumerable transition model
export interface ModeledEnvironment extends TabularEnvironment {
  transitions(s: number, a: number): TransitionOutcome[];
  isTerminal(s: number): boolean;
}

export function isModeledEnv(e: Environment): e is ModeledEnvironment {
  return typeof (e as Partial<ModeledEnvironment>).transitions === 'function';
}

// ---------- Q function abstraction ----------
export interface QFunction {
  get(s: number, a: number): number;
  set(s: number, a: number, v: number): void;
  values(s: number): number[]; // Q(s,·)
  maxQ(s: number): number;
  argmax(s: number): number;
  raw(): Float64Array;
  reset(): void;
}

// ---------- Action selection breakdown (data source for DecisionPanel and predict mode) ----------
export type PolicyKind = 'epsilon-greedy' | 'softmax' | 'fixed' | 'greedy' | 'gaussian';

export interface ActionExplanation {
  state: Obs;
  actionMeanings: string[];
  policyKind: PolicyKind;
  qValues?: number[]; // snapshot of Q(s,·) at decision time (value-based)
  greedyAction?: number; // argmaxₐ Q(s,a)
  actionProbs?: number[]; // π(a|s) (policy-based / fixed policy)
  epsilon?: number;
  randomDraw?: number; // the U(0,1) draw consumed by this step
  isExploring?: boolean;
  chosenAction: number;
  // Continuous (Gaussian) policy: the sampled scalar action and the distribution it came from.
  continuousAction?: number;
  mean?: number;
  std?: number;
  rationale: string; // human-readable rationale
}

export type UpdateInfo = Record<string, number>;

export interface Transition {
  state: Obs;
  action: Action;
  reward: number;
  nextState: Obs;
  terminated: boolean;
  truncated: boolean;
  nextAction?: Action; // needed by SARSA
}

export interface Agent {
  readonly id: string;
  readonly name: string;
  readonly usesQ: boolean; // whether it maintains Q (determines whether to draw the Q triangles)
  // Fast path: for batch training, returns the action only
  selectAction(state: Obs, greedy?: boolean): Action;
  // Interactive path: select the action AND return the full breakdown (sharing the same RNG draw)
  act(state: Obs, greedy?: boolean): { action: Action; explanation: ActionExplanation };
  update(t: Transition): UpdateInfo;
  // End of episode: decay epsilon (step-wise methods) or perform learning (MC/REINFORCE); returns the end-of-episode update summary for the panel to display.
  onEpisodeEnd(): UpdateInfo | undefined;
  getV?(stateCount: number): Float64Array; // value-based only
  getQ?(stateCount: number, nActions: number): Float64Array; // flattened [s*nA+a]
  getPolicy?(stateCount: number, nActions: number): Float64Array; // policy-based (REINFORCE) only, flattened π
  reset(): void;
  hyperparams: Record<string, number>;
}
