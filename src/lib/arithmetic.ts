import type { Seed, Bit } from './types';

// --- Constants ---
export const PRECISION = 52n;
export const WHOLE = 1n << PRECISION;
export const HALF = WHOLE >> 1n;
export const QUARTER = HALF >> 1n;
export const THREE_Q = HALF + QUARTER;
export const MAX_TOKENS = 300;

// ============================================================
// Seed scrambling — reversible permutation so that consecutive
// seeds map to distant points in (0,1).
//
// We operate on the "payload" bits of m = n+1 (everything after
// the leading 1). The scramble is a bijection within each
// bit-length class:
//   1. XOR with golden-ratio–derived constant
//   2. Bit-reverse the payload
//   3. XOR with a second constant
//
// This ensures that consecutive seeds (which differ only in low
// bits) produce very different high bits after scrambling, so the
// arithmetic decoder's first tokens diverge.
// ============================================================
const PHI_CONST_1 = 0x9E3779B97F4A7C15n; // floor(2^64 * (sqrt(5)-1)/2)
const PHI_CONST_2 = 0x517CC1B727220A95n; // floor(2^64 * (sqrt(5)-1)/2 * (sqrt(3)-1)/2)

function payloadLength(n: Seed): number {
  const m = n + 1n;
  let len = 0;
  let x = m;
  while (x > 0n) { x >>= 1n; len++; }
  return len - 1; // subtract 1 for the leading 1
}

function reverseBits(x: bigint, k: number): bigint {
  let result = 0n;
  let val = x;
  for (let i = 0; i < k; i++) {
    result = (result << 1n) | (val & 1n);
    val >>= 1n;
  }
  return result;
}

export function scramble(n: Seed): Seed {
  const k = payloadLength(n);
  if (k <= 0) return n;

  const mask = (1n << BigInt(k)) - 1n;
  let payload = (n + 1n) & mask;

  payload ^= PHI_CONST_1 & mask;
  payload = reverseBits(payload, k);
  payload ^= PHI_CONST_2 & mask;

  return ((1n << BigInt(k)) | payload) - 1n;
}

export function unscramble(n: Seed): Seed {
  const k = payloadLength(n);
  if (k <= 0) return n;

  const mask = (1n << BigInt(k)) - 1n;
  let payload = (n + 1n) & mask;

  // Reverse the three steps in opposite order
  payload ^= PHI_CONST_2 & mask;
  payload = reverseBits(payload, k); // self-inverse
  payload ^= PHI_CONST_1 & mask;

  return ((1n << BigInt(k)) | payload) - 1n;
}

// ============================================================
// BitStream — bijection N -> (0,1)
//
// n -> 0.[drop_leading_1(binary(n+1))]1000...
//
// This maps N onto all dyadic rationals with odd numerator,
// which are dense in (0,1):
//   n=0 -> 0.1    = 1/2
//   n=1 -> 0.01   = 1/4
//   n=2 -> 0.11   = 3/4
//   n=3 -> 0.001  = 1/8
//   n=4 -> 0.011  = 3/8
//   n=5 -> 0.101  = 5/8
//   n=6 -> 0.111  = 7/8
//   ...
// ============================================================
export class BitStream {
  private bits: Bit[];
  private pos: number;

  constructor(n: Seed) {
    // Scramble the seed so consecutive inputs map to distant reals
    const m = scramble(n) + 1n;
    const raw: Bit[] = [];
    let tmp = m;
    while (tmp > 0n) {
      raw.unshift(Number(tmp & 1n) as Bit);
      tmp >>= 1n;
    }
    raw.shift(); // drop leading 1 (always present since m >= 1)
    raw.push(1); // bijection marker
    this.bits = raw;
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
// BitStream(n) produces: [drop_leading_1(binary(scramble(n)+1)), 1, 0, 0, ...]
// We call the part before the marker the "payload" (length k).
//
// For payload length k, the stream is [B[0..k-1], 1, 0, 0, ...].
// This matches B iff B[k]==1 (or k>=len) and B[k+1..len-1] are all 0.
// The corresponding m = scramble(n)+1 = 1 followed by the payload in binary.
//
// We try k = 0, 1, ..., len to find the smallest internal seed,
// then unscramble to get the user-facing seed.
// ============================================================
export function bitsToSmallestBigInt(B: Bit[]): Seed {
  const len = B.length;
  if (len === 0) return unscramble(0n);

  for (let k = 0; k <= len; k++) {
    // The marker falls at position k in the stream.
    // If k < len, the encoder's bit at that position must be 1.
    if (k < len && B[k] !== 1) continue;
    // All encoder bits after the marker must be 0.
    let ok = true;
    for (let i = k + 1; i < len; i++) {
      if (B[i] !== 0) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;

    // m = binary "1" followed by payload B[0..k-1]
    let m = 1n << BigInt(k);
    for (let i = 0; i < k; i++) {
      m |= BigInt(B[i]) << BigInt(k - 1 - i);
    }
    return unscramble(m - 1n);
  }

  // Fallback (k = len always satisfies the constraints above,
  // so this is unreachable, but kept for safety)
  let m = 1n << BigInt(len);
  for (let i = 0; i < len; i++) {
    m |= BigInt(B[i]) << BigInt(len - 1 - i);
  }
  return unscramble(m - 1n);
}
