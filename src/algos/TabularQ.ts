// Tabular Q function: Float64Array(stateCount * nActions), indexed as s*nA+a.
import type { QFunction } from '../core/types';

export class TabularQ implements QFunction {
  private q: Float64Array;
  private nActions: number;
  private initVal: number;

  constructor(nStates: number, nActions: number, initVal = 0) {
    this.nActions = nActions;
    this.initVal = initVal;
    this.q = new Float64Array(nStates * nActions).fill(initVal);
  }

  get(s: number, a: number): number {
    return this.q[s * this.nActions + a];
  }

  set(s: number, a: number, v: number): void {
    this.q[s * this.nActions + a] = v;
  }

  values(s: number): number[] {
    const out: number[] = [];
    const base = s * this.nActions;
    for (let a = 0; a < this.nActions; a++) out.push(this.q[base + a]);
    return out;
  }

  maxQ(s: number): number {
    const base = s * this.nActions;
    let m = -Infinity;
    for (let a = 0; a < this.nActions; a++) {
      const v = this.q[base + a];
      if (v > m) m = v;
    }
    return m;
  }

  argmax(s: number): number {
    const base = s * this.nActions;
    let bi = 0;
    let bv = -Infinity;
    for (let a = 0; a < this.nActions; a++) {
      const v = this.q[base + a];
      if (v > bv) {
        bv = v;
        bi = a;
      }
    }
    return bi;
  }

  raw(): Float64Array {
    return this.q;
  }

  reset(): void {
    this.q.fill(this.initVal);
  }
}
