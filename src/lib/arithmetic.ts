import type { Seed, Bit } from './types';

// --- Constants ---
export const PRECISION = 52n;
export const WHOLE = 1n << PRECISION;
export const HALF = WHOLE >> 1n;
export const QUARTER = HALF >> 1n;
export const THREE_Q = HALF + QUARTER;
export const MAX_TOKENS = 300;

// ============================================================
// √2 − 1 bit generator — exact integer arithmetic, no floats.
//
// Maintains the invariant x = a√2 + b where a,b ∈ ℤ.
// Each step doubles x and emits 1 (subtracting 1) if x ≥ 1.
// The comparison a√2 ≥ c is decided by comparing 2a² to c²,
// which is exact because √2 is irrational (ties never occur).
// ============================================================
function* sqrt2minus1bits(): Generator<Bit, never, never> {
  let a = 1n; // x₀ = 1·√2 + (−1) = √2 − 1
  let b = -1n;
  while (true) {
    // x ← 2x
    a <<= 1n;
    b <<= 1n;
    // is x ≥ 1?  i.e. is a√2 + b ≥ 1?  i.e. is a√2 ≥ 1−b?
    const rhs = 1n - b;
    if (rhs <= 0n || 2n * a * a >= rhs * rhs) {
      yield 1;
      b -= 1n; // x ← x − 1
    } else {
      yield 0;
    }
  }
}

// ============================================================
// BitStream — bijection ℕ → (0,1) via Van der Corput ⊕ (√2−1)
//
// f(n) bit i = (bit i of n, LSB first) ⊕ (bit i of √2−1)
//
// Properties:
//   • Dense: n = 0..2ᵏ−1 covers all k-bit prefixes
//   • Never degenerates: once n's bits are exhausted, the tail
//     is √2−1 (irrational) — no all-0s or all-1s
//   • Consecutive seeds always differ in bit 0 (VdC flips LSB)
//   • No floating-point arithmetic anywhere
//   • Simple inverse: XOR target with √2−1, read bits → n
// ============================================================
export class BitStream {
  private alpha: Generator<Bit, never, never>;
  private remaining: bigint;
  private seed: Seed;

  constructor(n: Seed) {
    this.seed = n;
    this.remaining = n;
    this.alpha = sqrt2minus1bits();
  }

  readBit(): Bit {
    const vdcBit = Number(this.remaining & 1n) as Bit;
    this.remaining >>= 1n;
    return (vdcBit ^ this.alpha.next().value) as Bit;
  }

  describe(): string {
    // Peek at first 64 bits via a fresh generator (doesn't consume main stream)
    const peek = sqrt2minus1bits();
    let tmp = this.seed;
    const show: Bit[] = [];
    for (let i = 0; i < 64; i++) {
      const vdc = Number(tmp & 1n) as Bit;
      tmp >>= 1n;
      show.push((vdc ^ peek.next().value) as Bit);
    }
    return `0.${show.join('')}...  [VdC₂⊕(√2−1)]`;
  }
}

// ============================================================
// ArithmeticDecoder — range coder, BigInt fixed-point
// ============================================================
export class ArithmeticDecoder {
  private s: BitStream;
  private lo: bigint;
  private hi: bigint;
  private val: bigint;

  constructor(stream: BitStream) {
    this.s = stream;
    this.lo = 0n;
    this.hi = WHOLE - 1n;
    this.val = 0n;
    for (let i = 0n; i < PRECISION; i++) {
      this.val = (this.val << 1n) | BigInt(this.s.readBit());
    }
  }

  /**
   * Decode a symbol given cumulative probabilities in (0, 1].
   * cum[i] is the cumulative probability up to and including token i.
   * cum[cum.length-1] must be exactly 1.
   */
  decode(cum: number[]): number {
    const range = this.hi - this.lo + 1n;
    // Map val to a position within the cumulative distribution
    const pos = this.val - this.lo;
    // Binary search: find smallest a such that cum[a] * range > pos
    // (equivalent to: val falls in the bin for token a)
    let a = 0;
    let b = cum.length - 1;
    while (a < b) {
      const m = (a + b) >>> 1;
      // Convert cum[m] to fixed-point boundary
      const boundary = BigInt(Math.floor(cum[m] * Number(range)));
      if (boundary <= pos) a = m + 1;
      else b = m;
    }
    // Update interval: lo_new = lo + cum[a-1] * range, hi_new = lo + cum[a] * range - 1
    const cumHi = a < cum.length ? cum[a] : 1;
    const cumLo = a > 0 ? cum[a - 1] : 0;
    this.hi = this.lo + BigInt(Math.floor(cumHi * Number(range))) - 1n;
    this.lo = this.lo + BigInt(Math.floor(cumLo * Number(range)));
    this._renorm();
    return a;
  }

  private _renorm(): void {
    for (;;) {
      if (this.hi < HALF) {
        // both in lower half — no-op before shift
      } else if (this.lo >= HALF) {
        this.lo -= HALF;
        this.hi -= HALF;
        this.val -= HALF;
      } else if (this.lo >= QUARTER && this.hi < THREE_Q) {
        this.lo -= QUARTER;
        this.hi -= QUARTER;
        this.val -= QUARTER;
      } else break;
      this.lo = this.lo << 1n;
      this.hi = (this.hi << 1n) | 1n;
      this.val = (this.val << 1n) | BigInt(this.s.readBit());
    }
  }
}

// ============================================================
// ArithmeticEncoder — exact mirror of the decoder
// ============================================================
export class ArithmeticEncoder {
  private lo: bigint;
  private hi: bigint;
  private pending: number;
  private bits: Bit[];

  constructor() {
    this.lo = 0n;
    this.hi = WHOLE - 1n;
    this.pending = 0;
    this.bits = [];
  }

  private _emit(b: Bit): void {
    this.bits.push(b);
    const opp = (1 - b) as Bit;
    for (let i = 0; i < this.pending; i++) this.bits.push(opp);
    this.pending = 0;
  }

  /**
   * Encode a symbol given cumulative probabilities in (0, 1].
   * cum[i] is the cumulative probability up to and including token i.
   */
  encode(cum: number[], sym: number): void {
    const range = this.hi - this.lo + 1n;
    const cumHi = cum[sym];
    const cumLo = sym > 0 ? cum[sym - 1] : 0;
    this.hi = this.lo + BigInt(Math.floor(cumHi * Number(range))) - 1n;
    this.lo = this.lo + BigInt(Math.floor(cumLo * Number(range)));
    this._renorm();
  }

  private _renorm(): void {
    for (;;) {
      if (this.hi < HALF) {
        this._emit(0);
      } else if (this.lo >= HALF) {
        this._emit(1);
        this.lo -= HALF;
        this.hi -= HALF;
      } else if (this.lo >= QUARTER && this.hi < THREE_Q) {
        this.pending++;
        this.lo -= QUARTER;
        this.hi -= QUARTER;
      } else break;
      this.lo = this.lo << 1n;
      this.hi = (this.hi << 1n) | 1n;
    }
  }

  finalize(): Bit[] {
    this.pending++;
    if (this.lo < QUARTER) this._emit(0);
    else this._emit(1);
    return this.bits;
  }
}

// ============================================================
// Convert encoder output bits → smallest seed whose BitStream
// starts with those bits.
//
// Since f(n) bit i = (n's bit i) ⊕ (√2−1 bit i),
// we recover n's bits by XORing target with √2−1.
// Setting all higher bits of n to 0 gives the smallest solution.
// ============================================================
export function bitsToSmallestBigInt(B: Bit[]): Seed {
  const alpha = sqrt2minus1bits();
  let n = 0n;
  for (let i = 0; i < B.length; i++) {
    const alphaBit = alpha.next().value;
    if ((B[i] ^ alphaBit) as Bit) {
      n |= 1n << BigInt(i);
    }
  }
  return n;
}
