import { describe, it, expect } from 'vitest';
import { BitStream, scramble, unscramble } from '../lib/arithmetic';

describe('seed scrambling', () => {
  it('scramble is a bijection: unscramble inverts scramble for 0..999', () => {
    for (let n = 0n; n < 1000n; n++) {
      expect(unscramble(scramble(n))).toBe(n);
      expect(scramble(unscramble(n))).toBe(n);
    }
  });

  it('scramble is a bijection for large seeds', () => {
    const seeds = [
      437592987456924n,
      437592987456925n,
      1n << 100n,
      (1n << 100n) - 1n,
      (1n << 50n) + 7n,
    ];
    for (const n of seeds) {
      expect(unscramble(scramble(n))).toBe(n);
      expect(scramble(unscramble(n))).toBe(n);
    }
  });

  it('scramble preserves bit-length class', () => {
    function bitLen(n: bigint): number {
      if (n <= 0n) return 0;
      let len = 0;
      let x = n;
      while (x > 0n) { x >>= 1n; len++; }
      return len;
    }
    for (let n = 0n; n < 1000n; n++) {
      expect(bitLen(scramble(n) + 1n)).toBe(bitLen(n + 1n));
    }
  });

  it('consecutive seeds produce different BitStream prefixes', () => {
    const n1 = 437592987456924n;
    const n2 = 437592987456925n;
    const s1 = new BitStream(n1);
    const s2 = new BitStream(n2);
    const bits1 = Array.from({ length: 8 }, () => s1.readBit());
    const bits2 = Array.from({ length: 8 }, () => s2.readBit());
    expect(bits1).not.toEqual(bits2);
  });

  it('small consecutive seeds produce different BitStream prefixes', () => {
    // Check several consecutive pairs
    for (let base = 100n; base < 120n; base++) {
      const s1 = new BitStream(base);
      const s2 = new BitStream(base + 1n);
      const bits1 = Array.from({ length: 4 }, () => s1.readBit());
      const bits2 = Array.from({ length: 4 }, () => s2.readBit());
      // Not ALL consecutive pairs need to differ in 4 bits,
      // but scrambling should make most of them different
      // We'll check that at least 80% of pairs differ
    }
    // Stronger check: out of 20 consecutive pairs, at least 16 differ in first 4 bits
    let diffCount = 0;
    for (let base = 100n; base < 120n; base++) {
      const s1 = new BitStream(base);
      const s2 = new BitStream(base + 1n);
      const bits1 = Array.from({ length: 4 }, () => s1.readBit());
      const bits2 = Array.from({ length: 4 }, () => s2.readBit());
      if (bits1.join('') !== bits2.join('')) diffCount++;
    }
    expect(diffCount).toBeGreaterThanOrEqual(16);
  });
});
