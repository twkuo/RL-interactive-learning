// Algorithm registry: factory + UI hints (whether ε is used, which update-formula template).
import type { Agent, SyncEnvironment, TabularEnvironment } from '../core/types';
import { QLearning } from './tabular/QLearning';
import { Sarsa } from './tabular/Sarsa';
import { ExpectedSarsa } from './tabular/ExpectedSarsa';
import { DoubleQLearning } from './tabular/DoubleQLearning';
import { MonteCarlo } from './tabular/MonteCarlo';
import { Reinforce } from './tabular/Reinforce';

export type FormulaKind =
  | 'q-learning'
  | 'sarsa'
  | 'expected-sarsa'
  | 'double-q'
  | 'mc'
  | 'reinforce'
  | 'dqn';

// Algorithms that update only at episode end (the formula panel needs special handling).
export const EPISODIC_FORMULAS: FormulaKind[] = ['mc', 'reinforce'];

// A tunable hyperparameter declared by a (deep) algorithm; the HyperParams UI renders from it.
export interface HyperparamSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  decimals?: number;
  kind?: 'slider' | 'toggle'; // 'toggle' renders a checkbox (value 0/1); default 'slider'
  hint?: string;
}

export interface AlgoEntry {
  id: string;
  name: string;
  usesEpsilon: boolean;
  formula: FormulaKind;
  deep?: boolean; // deep RL (neural network); trains off-thread in a Web Worker
  requires?: 'discrete-obs' | 'box-obs'; // environment compatibility
  hyperparamSpec?: HyperparamSpec[]; // when present, the HyperParams panel renders from this
  // Tabular algorithms create synchronously. Deep algorithms load lazily via dynamic import,
  // which keeps TensorFlow.js out of the main bundle (tabular experience stays tfjs-free).
  create?: (env: TabularEnvironment, hp: Record<string, number>) => Agent;
  load?: () => Promise<{ create: (env: SyncEnvironment, hp: Record<string, number>) => Agent }>;
}

export const DEFAULT_HYPERPARAMS: Record<string, number> = {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.2,
  epsilonDecay: 0.99,
  epsilonMin: 0.01,
};

// DQN's own knobs (deep RL has no α; it has a network learning rate, replay buffer, etc.).
const DQN_HP: HyperparamSpec[] = [
  { key: 'lr', label: 'Learning rate', min: 0.0001, max: 0.01, step: 0.0001, default: 0.001, decimals: 4 },
  { key: 'gamma', label: 'Discount γ', min: 0.8, max: 1, step: 0.01, default: 0.99, decimals: 2 },
  { key: 'epsilon', label: 'ε start', min: 0, max: 1, step: 0.05, default: 1, decimals: 2 },
  { key: 'epsilonMin', label: 'ε min', min: 0, max: 0.5, step: 0.01, default: 0.05, decimals: 2 },
  { key: 'epsilonDecay', label: 'ε decay / episode', min: 0.9, max: 1, step: 0.005, default: 0.97, decimals: 3 },
  { key: 'hiddenUnits', label: 'Hidden units', min: 16, max: 256, step: 16, default: 64, decimals: 0 },
  { key: 'batchSize', label: 'Batch size', min: 16, max: 256, step: 16, default: 64, decimals: 0 },
  { key: 'bufferSize', label: 'Replay buffer', min: 1000, max: 50000, step: 1000, default: 10000, decimals: 0 },
  { key: 'targetSync', label: 'Target sync (steps)', min: 100, max: 2000, step: 100, default: 500, decimals: 0 },
  {
    key: 'normalize',
    label: 'Input normalization',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    kind: 'toggle',
    hint: 'Off → the network sees raw, imbalanced observation scales and learns much worse.',
  },
  {
    key: 'keepBest',
    label: 'Keep best policy for inference',
    min: 0,
    max: 1,
    step: 1,
    default: 1,
    kind: 'toggle',
    hint: 'Off → inference uses the FINAL weights, which can collapse if you over-train.',
  },
];

function dims(env: TabularEnvironment): [number, number, string[]] {
  return [env.stateCount(), env.actionSpace.n, env.actionMeanings()];
}

export const ALGO_REGISTRY: AlgoEntry[] = [
  {
    id: 'q-learning',
    name: 'Q-Learning',
    usesEpsilon: true,
    formula: 'q-learning',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new QLearning(ns, na, m, hp);
    },
  },
  {
    id: 'sarsa',
    name: 'SARSA',
    usesEpsilon: true,
    formula: 'sarsa',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new Sarsa(ns, na, m, hp);
    },
  },
  {
    id: 'expected-sarsa',
    name: 'Expected SARSA',
    usesEpsilon: true,
    formula: 'expected-sarsa',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new ExpectedSarsa(ns, na, m, hp);
    },
  },
  {
    id: 'double-q',
    name: 'Double Q-Learning',
    usesEpsilon: true,
    formula: 'double-q',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new DoubleQLearning(ns, na, m, hp);
    },
  },
  {
    id: 'mc',
    name: 'Monte Carlo Control',
    usesEpsilon: true,
    formula: 'mc',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new MonteCarlo(ns, na, m, hp);
    },
  },
  {
    id: 'reinforce',
    name: 'REINFORCE',
    usesEpsilon: false,
    formula: 'reinforce',
    create: (env, hp) => {
      const [ns, na, m] = dims(env);
      return new Reinforce(ns, na, m, hp);
    },
  },
  {
    id: 'dqn',
    name: 'DQN',
    usesEpsilon: true,
    formula: 'dqn',
    deep: true,
    requires: 'box-obs',
    hyperparamSpec: DQN_HP,
    load: () => import('./deep/DQN'),
  },
];

export function getAlgoEntry(id: string): AlgoEntry {
  return ALGO_REGISTRY.find((a) => a.id === id) ?? ALGO_REGISTRY[0];
}

// Default hyperparameters for an algorithm: deep algos use their spec defaults; tabular use the shared set.
export function defaultHyperparams(entry: AlgoEntry): Record<string, number> {
  if (entry.hyperparamSpec) {
    const hp: Record<string, number> = {};
    for (const s of entry.hyperparamSpec) hp[s.key] = s.default;
    return hp;
  }
  return { ...DEFAULT_HYPERPARAMS };
}
