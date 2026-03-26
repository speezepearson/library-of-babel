import type { Seed, Bit } from './types';

// --- Constants ---
export const PRECISION = 52n;
export const WHOLE = 1n << PRECISION;
export const HALF = WHOLE >> 1n;
export const QUARTER = HALF >> 1n;
export const THREE_Q = HALF + QUARTER;
export const MAX_TOKENS = 300;

// ============================================================
// BitStream — bijection N -> (0,1)
// n -> 0.[binary(n)]1000...
// ============================================================
export class BitStream {
  private bits: Bit[];
  private pos: number;

  constructor(n: Seed) {
    this.bits = [];
    if (n === 0n) {
      this.bits = [0];
    } else {
      let tmp = n;
      while (tmp > 0n) {
        this.bits.unshift(Number(tmp & 1n) as Bit);
        tmp >>= 1n;
      }
    }
    this.bits.push(1); // bijection marker
    this.pos = 0;
  }

  readBit(): Bit {
    if (this.pos < this.bits.length) return this.bits[this.pos++];
    this.pos++;
    return 0;
  }

  describe(): string {
    const show = this.bits.slice(0, 64).join('');
    const suffix = this.bits.length > 64 ? '...' : '';
    return `0.${show}${suffix}  [${this.bits.length} real bits + inf zeros]`;
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

  decode(cum: BigInt64Array, total: bigint): number {
    const range = this.hi - this.lo + 1n;
    const scaled = ((this.val - this.lo) * total) / range;
    let a = 0;
    let b = cum.length - 2;
    while (a < b) {
      const m = (a + b) >>> 1;
      if (cum[m + 1] <= scaled) a = m + 1;
      else b = m;
    }
    this.hi = this.lo + (range * cum[a + 1]) / total - 1n;
    this.lo = this.lo + (range * cum[a]) / total;
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

  encode(cum: BigInt64Array, total: bigint, sym: number): void {
    const range = this.hi - this.lo + 1n;
    this.hi = this.lo + (range * cum[sym + 1]) / total - 1n;
    this.lo = this.lo + (range * cum[sym]) / total;
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
// Convert encoder output bits -> smallest BigInt whose
// BitStream starts with those bits.
//
// BitStream(n) = [binary(n), 1, 0, 0, ...]
// We find the smallest n such that this stream starts with B.
// ============================================================
export function bitsToSmallestBigInt(B: Bit[]): Seed {
  const m = B.length;
  if (m === 0) return 0n;

  // Try marker positions mp = 1, 2, ..., m.
  // At position mp, the bijection places a '1'.
  // After mp, the stream is all zeros.
  // binary(n) = B[0..mp-1], n = int(B[0..mp-1]).
  for (let mp = 1; mp <= m; mp++) {
    if (mp < m && B[mp] !== 1) continue;
    let ok = true;
    for (let i = mp + 1; i < m; i++) {
      if (B[i] !== 0) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    let n = 0n;
    for (let i = 0; i < mp; i++) n = (n << 1n) | BigInt(B[i]);
    return n;
  }

  // Fallback: marker after all bits
  let n = 0n;
  for (let i = 0; i < m; i++) n = (n << 1n) | BigInt(B[i]);
  return n;
}
