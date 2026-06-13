// Space construction helpers.
import type { DiscreteSpace, BoxSpace } from './types';

export const discrete = (n: number): DiscreteSpace => ({ kind: 'discrete', n });

export const box = (low: number[], high: number[]): BoxSpace => ({
  kind: 'box',
  low,
  high,
  shape: [low.length],
});
