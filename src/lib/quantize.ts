import type { VocabSize, SamplerConfig } from './types';
import { DEFAULT_SAMPLER_CONFIG } from './types';

// ---------------------------------------------------------------------------
// Float64 bit-level helpers
// ---------------------------------------------------------------------------

const _f64 = new Float64Array(1);
const _u64 = new BigUint64Array(_f64.buffer);

/** Next representable Float64 above x (for positive x). */
function nextFloat64(x: number): number {
  if (x === 0) return Number.MIN_VALUE;
  _f64[0] = x;
  _u64[0] += 1n;
  return _f64[0];
}

/** Previous representable Float64 below x (for x > 0). */
function prevFloat64(x: number): number {
  _f64[0] = x;
  _u64[0] -= 1n;
  return _f64[0];
}

// ---------------------------------------------------------------------------
// ensureAscending
// ---------------------------------------------------------------------------

/**
 * Nudge values in-place so they are strictly increasing, first > 0, last = 1.
 *
 * Assumes input values are non-strictly in [0, 1] and non-strictly increasing.
 * Uses the smallest possible Float64 nudges to preserve the distribution.
 */
export function ensureAscending(xs: number[]): void {
  const n = xs.length;
  if (n === 0) return;

  // Forward pass: ensure first > 0 and strictly increasing
  if (xs[0] <= 0) xs[0] = Number.MIN_VALUE;
  for (let i = 1; i < n; i++) {
    if (xs[i] <= xs[i - 1]) xs[i] = nextFloat64(xs[i - 1]);
  }

  // Backward pass: ensure last = 1 and strictly increasing
  xs[n - 1] = 1;
  for (let i = n - 2; i >= 0; i--) {
    if (xs[i] >= xs[i + 1]) xs[i] = prevFloat64(xs[i + 1]);
  }
}

// ---------------------------------------------------------------------------
// quantizeLogits
// ---------------------------------------------------------------------------

/**
 * Convert raw logits to cumulative Float64 probabilities in (0, 1].
 *
 * Returns an array of length vocabSize where entry i is the cumulative
 * probability up to and including token i. The values are strictly
 * increasing, the first is > 0, and the last is exactly 1.
 *
 * Every token gets a positive-width bin, preserving the surjection property
 * (every token sequence is reachable from some seed).
 */
export function quantizeLogits(
  logits: ArrayLike<number>,
  vocabSize: VocabSize,
  config: SamplerConfig = DEFAULT_SAMPLER_CONFIG,
): number[] {
  // Apply temperature scaling
  const temp = config.temperature;
  const scaled = new Float64Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = logits[i] / temp;
  }

  // Apply top-K: set non-top-K logits to -Infinity
  if (config.topK > 0 && config.topK < vocabSize) {
    const vals = Array.from(scaled);
    vals.sort((a, b) => b - a);
    const threshold = vals[config.topK - 1];
    for (let i = 0; i < vocabSize; i++) {
      if (scaled[i] < threshold) scaled[i] = -Infinity;
    }
  }

  // Stable softmax
  let maxL = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (scaled[i] > maxL) maxL = scaled[i];
  }
  const probs = new Float64Array(vocabSize);
  let sumE = 0;
  for (let i = 0; i < vocabSize; i++) {
    probs[i] = Math.exp(scaled[i] - maxL);
    sumE += probs[i];
  }
  for (let i = 0; i < vocabSize; i++) probs[i] /= sumE;

  // Build cumulative distribution
  const cum: number[] = new Array(vocabSize);
  let running = 0;
  for (let i = 0; i < vocabSize; i++) {
    running += probs[i];
    cum[i] = running;
  }

  ensureAscending(cum);
  return cum;
}
