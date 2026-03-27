import { describe, it, expect } from 'vitest';
import { BitStream, bitsToSmallestBigInt } from '../lib/arithmetic';
import type { Bit } from '../lib/types';

function take(stream: BitStream, k: number): Bit[] {
  const out: Bit[] = [];
  for (let i = 0; i < k; i++) out.push(stream.readBit());
  return out;
}

describe('VdC ⊕ √2−1 bijection', () => {
  it('round-trips: bitsToSmallestBigInt inverts BitStream', () => {
    for (let n = 0n; n < 200n; n++) {
      const bits = take(new BitStream(n), 20);
      const recovered = bitsToSmallestBigInt(bits);
      const rebits = take(new BitStream(recovered), 20);
      expect(rebits).toEqual(bits);
    }
  });

  it('bitsToSmallestBigInt returns the smallest matching n', () => {
    for (let n = 0n; n < 100n; n++) {
      const bits = take(new BitStream(n), 12);
      const smallest = bitsToSmallestBigInt(bits);
      // The smallest n whose first 12 bits match should be <= n
      // (because we only look at the first 12 bits and set higher bits to 0)
      const rebits = take(new BitStream(smallest), 12);
      expect(rebits).toEqual(bits);
      expect(smallest).toBeLessThanOrEqual(n);
    }
  });

  it('is dense: n=0..2^k-1 covers all k-bit prefixes', () => {
    const k = 6;
    const seen = new Set<string>();
    for (let n = 0n; n < (1n << BigInt(k)); n++) {
      const bits = take(new BitStream(n), k);
      seen.add(bits.join(''));
    }
    expect(seen.size).toBe(1 << k); // all 2^k prefixes covered
  });

  it('never degenerates: small seeds still produce varied bits', () => {
    // The old bijection would go to all-0s after a few bits for small n.
    // The new one should produce a mix of 0s and 1s throughout.
    for (let n = 0n; n <= 7n; n++) {
      const bits = take(new BitStream(n), 100);
      const ones = bits.filter((b) => b === 1).length;
      // √2−1 has density ~0.5; expect between 20% and 80% ones
      expect(ones).toBeGreaterThan(20);
      expect(ones).toBeLessThan(80);
    }
  });

  it('n=0 output is exactly √2−1 bits', () => {
    // VdC₂(0) = 0.000..., so f(0) = 0 ⊕ √2−1 = √2−1
    // √2−1 = 0.01101010000010011110...₂
    const bits = take(new BitStream(0n), 20);
    // First 20 bits of √2−1 (known):
    const expected = [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0];
    expect(bits).toEqual(expected);
  });

  it('consecutive seeds produce different first bits', () => {
    let diffCount = 0;
    for (let base = 0n; base < 20n; base++) {
      const b1 = take(new BitStream(base), 4);
      const b2 = take(new BitStream(base + 1n), 4);
      if (b1.join('') !== b2.join('')) diffCount++;
    }
    // VdC flips bit 0 every step, so consecutive seeds always differ in bit 0
    expect(diffCount).toBe(20);
  });

  it('large seeds do not degenerate', () => {
    const big = (1n << 256n) + 42n;
    const bits = take(new BitStream(big), 500);
    const ones = bits.filter((b) => b === 1).length;
    expect(ones).toBeGreaterThan(100);
    expect(ones).toBeLessThan(400);
  });
});
