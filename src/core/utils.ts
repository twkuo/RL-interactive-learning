// Numeric helpers (hand-written, no heavy math library pulled in).
import type { RNG } from './rng';

export function argmax(arr: number[] | Float64Array): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bestV) {
      bestV = arr[i];
      best = i;
    }
  }
  return best;
}

// argmax with random tie-breaking (avoids always favoring the first action)
export function argmaxTie(arr: number[], rng: RNG): number {
  let bestV = -Infinity;
  const ties: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bestV + 1e-9) {
      bestV = arr[i];
      ties.length = 0;
      ties.push(i);
    } else if (Math.abs(arr[i] - bestV) <= 1e-9) {
      ties.push(i);
    }
  }
  return ties.length === 1 ? ties[0] : ties[rng.int(ties.length)];
}

export function softmax(arr: number[]): number[] {
  const m = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function clip(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

export function oneHot(i: number, n: number): number[] {
  const v = new Array<number>(n).fill(0);
  v[i] = 1;
  return v;
}
