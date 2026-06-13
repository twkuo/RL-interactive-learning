// FrozenLake-v1 (4x4), implemented as a GridWorld configuration.
// Map: SFFF / FHFH / FFFH / HFFG (H=hole terminates with reward 0; G=goal with reward +1).
import { GridWorld, type GridWorldConfig } from './GridWorld';

// Role of each cell in the 4x4 map
const HOLES = [5, 7, 11, 12];
const GOAL = 15;
const START = 0;

export function frozenLakeConfig(slippery: boolean): GridWorldConfig {
  return {
    id: slippery ? 'frozenlake-slippery' : 'frozenlake',
    name: slippery ? 'FrozenLake 4x4 (slippery)' : 'FrozenLake 4x4 (deterministic)',
    rows: 4,
    cols: 4,
    start: START,
    goals: [GOAL],
    holes: HOLES,
    walls: [],
    stepReward: 0,
    goalReward: 1,
    holeReward: 0,
    slip: slippery ? 2 / 3 : 0,
    maxSteps: 100,
  };
}

export function makeFrozenLake(slippery: boolean, seed?: number): GridWorld {
  return new GridWorld(frozenLakeConfig(slippery), seed);
}
