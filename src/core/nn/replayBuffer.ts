// Fixed-capacity ring replay buffer for off-policy deep RL (DQN).
// Stores raw transitions; sampling is uniform with replacement using an injected RNG
// (so training in the worker stays reproducible for a given seed).

export interface Sample {
  s: number[]; // state vector
  a: number; // action
  r: number; // reward
  s2: number[]; // next state vector
  done: number; // 1 if terminated (NOT truncated) — bootstrap target is zeroed only on terminated
}

export class ReplayBuffer {
  private buf: Sample[] = [];
  private pos = 0;
  private capacity: number;
  private rand: () => number;

  constructor(capacity: number, rand: () => number) {
    this.capacity = capacity;
    this.rand = rand;
  }

  push(sample: Sample): void {
    if (this.buf.length < this.capacity) this.buf.push(sample);
    else this.buf[this.pos] = sample;
    this.pos = (this.pos + 1) % this.capacity;
  }

  get size(): number {
    return this.buf.length;
  }

  // Uniform sample with replacement. Caller ensures size >= n (or n is small).
  sample(n: number): Sample[] {
    const out: Sample[] = [];
    const len = this.buf.length;
    for (let i = 0; i < n; i++) {
      out.push(this.buf[Math.floor(this.rand() * len)]);
    }
    return out;
  }
}
