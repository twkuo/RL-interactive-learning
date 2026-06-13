// Continuous-state discretization (state aggregation): bins a continuous vector into a single
// discrete index, so existing tabular agents (Q-learning/SARSA/...) can run continuous
// environments with zero changes.
export interface DimSpec {
  low: number;
  high: number;
  bins: number;
}

export class Discretizer {
  readonly stateCount: number;
  private dims: DimSpec[];

  constructor(dims: DimSpec[]) {
    this.dims = dims;
    this.stateCount = dims.reduce((p, d) => p * d.bins, 1);
  }

  // Continuous vector -> discrete index (mixed-radix; out-of-range values are clamped to the edge bin)
  index(cont: number[]): number {
    let idx = 0;
    for (let i = 0; i < this.dims.length; i++) {
      const d = this.dims[i];
      const clamped = Math.max(d.low, Math.min(d.high, cont[i]));
      let b = Math.floor(((clamped - d.low) / (d.high - d.low)) * d.bins);
      if (b >= d.bins) b = d.bins - 1;
      if (b < 0) b = 0;
      idx = idx * d.bins + b;
    }
    return idx;
  }
}
