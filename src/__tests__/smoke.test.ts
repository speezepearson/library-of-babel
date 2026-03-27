import { describe, it, expect, beforeAll } from 'vitest';
import {
  configureForNode,
  loadModel,
  getPromptIds,
  getStopIds,
  getModelParams,
  getLastLogits,
} from '../lib/model';
import { BitStream, ArithmeticDecoder, ArithmeticEncoder, bitsToSmallestBigInt } from '../lib/arithmetic';
import { quantizeLogits } from '../lib/quantize';
import type { Seed, TokenId } from '../lib/types';
import type { PreTrainedTokenizer, PreTrainedModel } from '@huggingface/transformers';
import type { ModelParams } from '../lib/model';

const SYSTEM_PROMPT = 'You are a helpful, concise assistant.';
const USER_MESSAGE = 'Tell me something interesting.';
const TEST_SEED: Seed = 42n;
const NUM_TOKENS = 5;

async function decodeNTokens(
  model: PreTrainedModel,
  promptIds: TokenId[],
  stopIds: Set<TokenId>,
  params: ModelParams,
  seed: Seed,
  maxTokens: number,
): Promise<TokenId[]> {
  const stream = new BitStream(seed);
  const decoder = new ArithmeticDecoder(stream);
  const allIds = [...promptIds];
  const decoded: TokenId[] = [];

  for (let n = 0; n < maxTokens; n++) {
    const logits = await getLastLogits(model, allIds);
    const cum = quantizeLogits(logits, params.vocabSize);
    const tokId = decoder.decode(cum);
    if (stopIds.has(tokId)) break;
    allIds.push(tokId);
    decoded.push(tokId);
  }
  return decoded;
}

describe('arithmetic decoding smoke test', () => {
  let tokenizer: PreTrainedTokenizer;
  let model: PreTrainedModel;
  let promptIds: TokenId[];
  let stopIds: Set<TokenId>;
  let params: ModelParams;

  beforeAll(async () => {
    configureForNode('./models');
    const bundle = await loadModel((msg) => console.log(`[model] ${msg}`));
    tokenizer = bundle.tokenizer;
    model = bundle.model;
    promptIds = getPromptIds(tokenizer, SYSTEM_PROMPT, USER_MESSAGE);
    stopIds = getStopIds(tokenizer);
    params = getModelParams(model);
  });

  it('decodes deterministically from a known seed', async () => {
    const tokens1 = await decodeNTokens(
      model, promptIds, stopIds, params, TEST_SEED, NUM_TOKENS,
    );
    expect(tokens1.length).toBeGreaterThan(0);
    expect(tokens1.length).toBeLessThanOrEqual(NUM_TOKENS);

    // Same seed must produce identical tokens
    const tokens2 = await decodeNTokens(
      model, promptIds, stopIds, params, TEST_SEED, NUM_TOKENS,
    );
    expect(tokens2).toEqual(tokens1);
  });

  it('produces different output for different seeds', async () => {
    const tokensA = await decodeNTokens(
      model, promptIds, stopIds, params, 0n, NUM_TOKENS,
    );
    const tokensB = await decodeNTokens(
      model, promptIds, stopIds, params, 9999n, NUM_TOKENS,
    );
    expect(tokensA.length).toBeGreaterThan(0);
    expect(tokensB.length).toBeGreaterThan(0);
    // Extremely unlikely that two arbitrary seeds produce the same 5 tokens
    expect(tokensA).not.toEqual(tokensB);
  });

  it('decoded tokens are valid strings', async () => {
    const tokens = await decodeNTokens(
      model, promptIds, stopIds, params, TEST_SEED, NUM_TOKENS,
    );
    for (const tokId of tokens) {
      const piece = tokenizer.decode([tokId], { skip_special_tokens: false });
      expect(typeof piece).toBe('string');
      expect(piece.length).toBeGreaterThan(0);
    }
  });

  it('prefix search roundtrips: encode then decode recovers the prefix', async () => {
    const TARGET = 'The true meaning of life is';

    // Tokenize the target
    const enc = tokenizer(TARGET, { add_special_tokens: false });
    const targetIds: TokenId[] = (
      Array.isArray(enc.input_ids)
        ? enc.input_ids
        : Array.from(
            (enc.input_ids as unknown as { data?: ArrayLike<number> }).data ??
              (enc.input_ids as unknown as ArrayLike<number>),
          )
    ).map(Number);

    // --- ENCODE: target tokens → seed ---
    const encoder = new ArithmeticEncoder();
    const ctx = [...promptIds];
    for (let i = 0; i < targetIds.length; i++) {
      const logits = await getLastLogits(model, ctx);
      const cum = quantizeLogits(logits, params.vocabSize);
      encoder.encode(cum, targetIds[i]);
      ctx.push(targetIds[i]);
    }
    const bits = encoder.finalize();
    const seed = bitsToSmallestBigInt(bits);

    // --- DECODE: seed → tokens ---
    const decoded = await decodeNTokens(
      model, promptIds, stopIds, params, seed, 50,
    );
    const text = tokenizer.decode(decoded, { skip_special_tokens: false });

    expect(text).toMatch(new RegExp('^' + TARGET.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  });
});
