import { describe, it, expect } from 'vitest';
import { quantizeLogits } from '../lib/quantize';
import type { SamplerConfig } from '../lib/types';

const VOCAB = 8;
// Logits: token 0 is clearly dominant
const LOGITS = new Float32Array([10, 2, 1, 0, -1, -2, -3, -4]);

/** Extract per-token probability widths from cumulative array. */
function cumToWidths(cum: number[]): number[] {
  const w: number[] = [cum[0]];
  for (let i = 1; i < cum.length; i++) {
    w.push(cum[i] - cum[i - 1]);
  }
  return w;
}

describe('quantizeLogits with SamplerConfig', () => {
  it('default config (temp=1, topK=0) produces valid cumulative distribution', () => {
    const config: SamplerConfig = { temperature: 1.0, topK: 0 };
    const cum = quantizeLogits(LOGITS, VOCAB, config);
    // Last element is exactly 1
    expect(cum[cum.length - 1]).toBe(1);
    // Strictly increasing
    for (let i = 1; i < cum.length; i++) {
      expect(cum[i]).toBeGreaterThan(cum[i - 1]);
    }
    // First > 0
    expect(cum[0]).toBeGreaterThan(0);
    // Token 0 (logit=10) should have the widest bin
    const widths = cumToWidths(cum);
    expect(widths[0]).toBe(Math.max(...widths));
  });

  it('low temperature sharpens the distribution', () => {
    const normal = quantizeLogits(LOGITS, VOCAB, { temperature: 1.0, topK: 0 });
    const sharp = quantizeLogits(LOGITS, VOCAB, { temperature: 0.5, topK: 0 });
    const normalW = cumToWidths(normal);
    const sharpW = cumToWidths(sharp);
    // Token 0 should get MORE mass with lower temperature
    expect(sharpW[0]).toBeGreaterThanOrEqual(normalW[0]);
  });

  it('high temperature flattens the distribution', () => {
    const normal = quantizeLogits(LOGITS, VOCAB, { temperature: 1.0, topK: 0 });
    const flat = quantizeLogits(LOGITS, VOCAB, { temperature: 5.0, topK: 0 });
    const normalW = cumToWidths(normal);
    const flatW = cumToWidths(flat);
    // Token 0 should get LESS mass with higher temperature
    expect(flatW[0]).toBeLessThanOrEqual(normalW[0]);
  });

  it('topK concentrates mass on top tokens', () => {
    const full = quantizeLogits(LOGITS, VOCAB, { temperature: 1.0, topK: 0 });
    const top2 = quantizeLogits(LOGITS, VOCAB, { temperature: 1.0, topK: 2 });
    const fullW = cumToWidths(full);
    const top2W = cumToWidths(top2);
    // Top tokens should get at least as much combined mass as before
    expect(top2W[0] + top2W[1]).toBeGreaterThanOrEqual(fullW[0] + fullW[1]);
  });

  it('topK=1 gives almost all mass to the top token', () => {
    const cum = quantizeLogits(LOGITS, VOCAB, { temperature: 1.0, topK: 1 });
    const widths = cumToWidths(cum);
    // Token 0 should have nearly all the mass
    expect(widths[0]).toBeGreaterThan(0.99);
    // Other tokens still have positive width (surjection)
    for (let i = 1; i < VOCAB; i++) {
      expect(widths[i]).toBeGreaterThan(0);
    }
  });
});
