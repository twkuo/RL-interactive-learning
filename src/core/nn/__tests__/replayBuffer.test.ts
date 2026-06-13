import { describe, it, expect } from 'vitest';
import { ReplayBuffer, type Sample } from '../replayBuffer';

const mk = (i: number): Sample => ({ s: [i], a: 0, r: i, s2: [i + 1], done: 0 });

describe('ReplayBuffer', () => {
  it('grows up to capacity then overwrites the oldest (ring buffer)', () => {
    const buf = new ReplayBuffer(3, () => 0); // rand=0 → always index 0
    expect(buf.size).toBe(0);
    buf.push(mk(0));
    buf.push(mk(1));
    buf.push(mk(2));
    expect(buf.size).toBe(3);
    buf.push(mk(3)); // overwrites slot 0 (the oldest)
    expect(buf.size).toBe(3);
    expect(buf.sample(1)[0].r).toBe(3); // slot 0 now holds mk(3)
  });

  it('returns exactly the requested number of samples', () => {
    const buf = new ReplayBuffer(10, () => 0.5);
    for (let i = 0; i < 5; i++) buf.push(mk(i));
    expect(buf.sample(8)).toHaveLength(8);
  });
});
