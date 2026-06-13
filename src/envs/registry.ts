// Environment registry: provides the dropdown list and factory functions.
import type { TabularEnvironment } from '../core/types';
import { GridWorld, type GridWorldConfig } from './discrete/GridWorld';
import { makeFrozenLake } from './discrete/FrozenLake';
import { CartPole } from './continuous/CartPole';
import { MountainCar } from './continuous/MountainCar';

export type RenderKind = 'grid' | 'cartpole' | 'mountaincar';

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

export interface EnvEntry {
  id: string;
  name: string;
  renderKind: RenderKind;
  create: (seed?: number) => TabularEnvironment;
}

export const ENV_REGISTRY: EnvEntry[] = [
  {
    id: 'gridworld',
    name: 'GridWorld 5x5',
    renderKind: 'grid',
    create: (seed) => new GridWorld(GRIDWORLD_CFG, seed),
  },
  {
    id: 'frozenlake',
    name: 'FrozenLake (deterministic)',
    renderKind: 'grid',
    create: (seed) => makeFrozenLake(false, seed),
  },
  {
    id: 'frozenlake-slippery',
    name: 'FrozenLake (slippery)',
    renderKind: 'grid',
    create: (seed) => makeFrozenLake(true, seed),
  },
  {
    id: 'cartpole',
    name: 'CartPole',
    renderKind: 'cartpole',
    create: (seed) => new CartPole(seed),
  },
  {
    id: 'mountaincar',
    name: 'MountainCar',
    renderKind: 'mountaincar',
    create: (seed) => new MountainCar(seed),
  },
];

export function getEnvEntry(id: string): EnvEntry {
  return ENV_REGISTRY.find((e) => e.id === id) ?? ENV_REGISTRY[0];
}
