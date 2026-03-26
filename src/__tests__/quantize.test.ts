import { describe, it, expect } from 'vitest';
import { quantizeLogits } from '../lib/quantize';
import type { SamplerConfig } from '../lib/types';

const VOCAB = 8;
const PROB_TOTAL = 16; // small power of 2 >= VOCAB
// Logits: token 0 is clearly dominant
const LOGITS = new Float32Array([10, 2, 1, 0, -1, -2, -3, -4]);

function cumToWeights(cum: BigInt64Array): number[] {
  const w: number[] = [];
  for (let i = 0; i < cum.length - 1; i++) {
    w.push(Number(cum[i + 1] - cum[i]));
  }
  return w;
}

describe('quantizeLogits with SamplerConfig', () => {
  it('default config (temp=1, topK=0) preserves existing behavior', () => {
    const config: SamplerConfig = { temperature: 1.0, topK: 0 };
    const cum = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, config);
    const weights = cumToWeights(cum);
    // Total should equal PROB_TOTAL
    expect(weights.reduce((a, b) => a + b, 0)).toBe(PROB_TOTAL);
    // Every token gets at least weight 1
    for (const w of weights) expect(w).toBeGreaterThanOrEqual(1);
    // Token 0 (logit=10) should have the most weight
    expect(weights[0]).toBe(Math.max(...weights));
  });

  it('low temperature sharpens the distribution', () => {
    const normal = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 1.0, topK: 0 });
    const sharp = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 0.5, topK: 0 });
    const normalW = cumToWeights(normal);
    const sharpW = cumToWeights(sharp);
    // Token 0 should get MORE weight with lower temperature
    expect(sharpW[0]).toBeGreaterThanOrEqual(normalW[0]);
  });

  it('high temperature flattens the distribution', () => {
    const normal = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 1.0, topK: 0 });
    const flat = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 5.0, topK: 0 });
    const normalW = cumToWeights(normal);
    const flatW = cumToWeights(flat);
    // Token 0 should get LESS weight with higher temperature
    expect(flatW[0]).toBeLessThanOrEqual(normalW[0]);
  });

  it('topK concentrates mass on top tokens', () => {
    const full = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 1.0, topK: 0 });
    const top2 = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 1.0, topK: 2 });
    const fullW = cumToWeights(full);
    const top2W = cumToWeights(top2);
    // Total still equals PROB_TOTAL
    expect(top2W.reduce((a, b) => a + b, 0)).toBe(PROB_TOTAL);
    // Non-top-2 tokens should have minimum weight (1)
    for (let i = 2; i < VOCAB; i++) {
      expect(top2W[i]).toBe(1);
    }
    // Top tokens should get at least as much weight as before
    expect(top2W[0] + top2W[1]).toBeGreaterThanOrEqual(fullW[0] + fullW[1]);
  });

  it('topK=1 gives almost all mass to the top token', () => {
    const cum = quantizeLogits(LOGITS, VOCAB, PROB_TOTAL, { temperature: 1.0, topK: 1 });
    const weights = cumToWeights(cum);
    // Token 0 gets PROB_TOTAL - (VOCAB-1)*1 = 16 - 7 = 9
    expect(weights[0]).toBe(PROB_TOTAL - (VOCAB - 1));
    for (let i = 1; i < VOCAB; i++) {
      expect(weights[i]).toBe(1);
    }
  });
});
