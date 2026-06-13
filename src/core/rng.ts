// Seedable PRNG (mulberry32), guaranteeing experiments are reproducible.
export class RNG {
  private state: number;

  constructor(seed = 12345) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 0x9e3779b9;
  }

  // returns [0, 1)
  next(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // returns an integer in [0, n)
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  setSeed(seed: number): void {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 0x9e3779b9;
  }
}
