import type { VocabSize, ProbTotal } from './types';

/**
 * Convert raw logits to a cumulative BigInt64Array of integer weights.
 * Every token gets weight >= 1, preserving the surjection property
 * (every token sequence is reachable from some seed).
 */
export function quantizeLogits(
  logits: ArrayLike<number>,
  vocabSize: VocabSize,
  probTotal: ProbTotal,
): BigInt64Array {
  // Stable softmax
  let maxL = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (logits[i] > maxL) maxL = logits[i];
  }
  const probs = new Float64Array(vocabSize);
  let sumE = 0;
  for (let i = 0; i < vocabSize; i++) {
    probs[i] = Math.exp(logits[i] - maxL);
    sumE += probs[i];
  }
  for (let i = 0; i < vocabSize; i++) probs[i] /= sumE;

  // Quantize: each token gets at least weight 1
  const pool = probTotal - vocabSize;
  const wt = new Uint32Array(vocabSize);
  let used = 0;
  for (let i = 0; i < vocabSize; i++) {
    const extra = Math.floor(probs[i] * pool);
    wt[i] = 1 + extra;
    used += extra;
  }

  // Distribute leftover by largest fractional part
  let leftover = pool - used;
  if (leftover > 0) {
    const frac = new Float64Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      frac[i] = probs[i] * pool - Math.floor(probs[i] * pool);
    }
    const idx = new Uint32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) idx[i] = i;
    idx.sort((a, b) => frac[b] - frac[a]);
    for (let k = 0; k < leftover; k++) wt[idx[k]]++;
  }

  // Build cumulative distribution
  const cum = new BigInt64Array(vocabSize + 1);
  cum[0] = 0n;
  for (let i = 0; i < vocabSize; i++) cum[i + 1] = cum[i] + BigInt(wt[i]);
  return cum;
}
