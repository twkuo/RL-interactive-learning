// Environment registry: provides the dropdown list and factory functions.
import type { AnyEnv } from '../core/types';
import { GridWorld, type GridWorldConfig } from './discrete/GridWorld';
import { makeFrozenLake } from './discrete/FrozenLake';
import { Bandit, type BanditConfig } from './discrete/Bandit';
import { CartPole } from './continuous/CartPole';
import { CartPoleVec } from './continuous/CartPoleVec';
import { Acrobot } from './continuous/Acrobot';
import { Pendulum } from './continuous/Pendulum';
import { LunarLander } from './continuous/LunarLander';
import { MountainCar } from './continuous/MountainCar';

export type RenderKind = 'grid' | 'cartpole' | 'mountaincar' | 'acrobot' | 'pendulum' | 'lunarlander' | 'bandit';

// Default 5x5 GridWorld (deterministic, making it easy to observe argmax -> movement).
//   . . . . .
//   . # . H .
//   . . . # .
//   . H . . .
//   . . . . G
const GRIDWORLD_CFG: GridWorldConfig = {
  id: 'gridworld',
  name: 'GridWorld 5x5',
  rows: 5,
  cols: 5,
  start: 0,
  goals: [24],
  holes: [8, 16],
  walls: [6, 13],
  stepReward: -0.04,
  goalReward: 1,
  holeReward: -1,
  slip: 0,
  maxSteps: 100,
};

// Multi-armed bandit: a single state, 5 arms with hidden Gaussian rewards. The classic
// exploration-vs-exploitation testbed (Chapter 3 starts here).
const BANDIT_CFG: BanditConfig = {
  id: 'bandit',
  name: 'Multi-Armed Bandit',
  arms: 5,
  noiseStd: 1,
  maxSteps: 1, // each pull is its own terminal step (no bootstrapping); see Bandit.stepSync
};

export interface EnvEntry {
  id: string;
  name: string;
  renderKind: RenderKind;
  // observation encoding: 'discrete' = state index (tabular agents); 'box' = raw vector (deep agents)
  obsKind: 'discrete' | 'box';
  actionKind: 'discrete' | 'continuous';
  // Environments in the same group have directly comparable returns, so the comparison chart is
  // kept (not reset) when switching between them — e.g. the tabular CartPole and the deep CartPoleVec
  // are the same underlying task, enabling "tabular vs DQN on CartPole" overlays.
  compareGroup: string;
  deep?: boolean; // continuous-observation env intended for the deep-RL (function-approximation) agents
  create: (seed?: number) => AnyEnv;
}

export const ENV_REGISTRY: EnvEntry[] = [
  {
    id: 'gridworld',
    name: 'GridWorld 5x5',
    renderKind: 'grid',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'gridworld',
    create: (seed) => new GridWorld(GRIDWORLD_CFG, seed),
  },
  {
    id: 'frozenlake',
    name: 'FrozenLake (deterministic)',
    renderKind: 'grid',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'frozenlake',
    create: (seed) => makeFrozenLake(false, seed),
  },
  {
    id: 'frozenlake-slippery',
    name: 'FrozenLake (slippery)',
    renderKind: 'grid',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'frozenlake-slippery',
    create: (seed) => makeFrozenLake(true, seed),
  },
  {
    id: 'bandit',
    name: 'Multi-Armed Bandit',
    renderKind: 'bandit',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'bandit',
    create: (seed) => new Bandit(BANDIT_CFG, seed),
  },
  {
    id: 'cartpole',
    name: 'CartPole',
    renderKind: 'cartpole',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'cartpole',
    create: (seed) => new CartPole(seed),
  },
  {
    id: 'mountaincar',
    name: 'MountainCar',
    renderKind: 'mountaincar',
    obsKind: 'discrete',
    actionKind: 'discrete',
    compareGroup: 'mountaincar',
    create: (seed) => new MountainCar(seed),
  },
  {
    id: 'cartpole-vec',
    name: 'CartPole',
    renderKind: 'cartpole',
    obsKind: 'box',
    actionKind: 'discrete',
    compareGroup: 'cartpole', // same group as the tabular CartPole → comparable returns
    deep: true,
    create: (seed) => new CartPoleVec(seed),
  },
  {
    id: 'acrobot',
    name: 'Acrobot',
    renderKind: 'acrobot',
    obsKind: 'box',
    actionKind: 'discrete',
    compareGroup: 'acrobot',
    deep: true,
    create: (seed) => new Acrobot(seed),
  },
  {
    id: 'pendulum',
    name: 'Pendulum',
    renderKind: 'pendulum',
    obsKind: 'box',
    actionKind: 'continuous',
    compareGroup: 'pendulum',
    deep: true,
    create: (seed) => new Pendulum(seed),
  },
  {
    id: 'lunarlander',
    name: 'LunarLander',
    renderKind: 'lunarlander',
    obsKind: 'box',
    actionKind: 'discrete',
    compareGroup: 'lunarlander',
    deep: true,
    create: (seed) => new LunarLander(seed),
  },
];

export function getEnvEntry(id: string): EnvEntry {
  return ENV_REGISTRY.find((e) => e.id === id) ?? ENV_REGISTRY[0];
}
