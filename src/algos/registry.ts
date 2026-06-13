// Algorithm registry: factory + UI hints (whether ε is used, which update-formula template).
import type { Agent, TabularEnvironment } from '../core/types';
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
  | 'reinforce';

// Algorithms that update only at episode end (the formula panel needs special handling).
export const EPISODIC_FORMULAS: FormulaKind[] = ['mc', 'reinforce'];

export interface AlgoEntry {
  id: string;
  name: string;
  usesEpsilon: boolean;
  formula: FormulaKind;
  create: (env: TabularEnvironment, hp: Record<string, number>) => Agent;
}

export const DEFAULT_HYPERPARAMS: Record<string, number> = {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.2,
  epsilonDecay: 0.99,
  epsilonMin: 0.01,
};

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
];

export function getAlgoEntry(id: string): AlgoEntry {
  return ALGO_REGISTRY.find((a) => a.id === id) ?? ALGO_REGISTRY[0];
}
