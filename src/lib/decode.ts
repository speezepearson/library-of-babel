import type { Seed, TokenId } from './types';
import {
  BitStream,
  ArithmeticDecoder,
  ArithmeticEncoder,
  bitsToSmallestBigInt,
  MAX_TOKENS,
} from './arithmetic';
import { quantizeLogits } from './quantize';
import {
  loadModel,
  getPromptIds,
  getStopIds,
  getModelParams,
  getLastLogits,
  type StatusCallback,
} from './model';

export interface DecodeCallbacks {
  onToken: (piece: string, tokenIndex: number) => void;
  onStatus: StatusCallback;
  shouldStop: () => boolean;
}

export interface SearchCallbacks {
  onStatus: StatusCallback;
  shouldStop: () => boolean;
}

export interface PrefixSearchResult {
  seed: Seed;
  bitLen: number;
  tokenCount: number;
  decodedTarget: string;
}

export interface SubstringSearchResult {
  seed: Seed;
  bitLen: number;
  info: string;
}

// ============================================================
// DECODE — read bits from BigInt, produce tokens
// ============================================================
export async function decode(
  systemPrompt: string,
  userMessage: string,
  seed: Seed,
  callbacks: DecodeCallbacks,
): Promise<void> {
  const { tokenizer, model } = await loadModel(callbacks.onStatus);
  const promptIds = getPromptIds(tokenizer, systemPrompt, userMessage);
  const stopIds = getStopIds(tokenizer);
  const { vocabSize, probTotal, probTotalBig } = getModelParams(model);

  const stream = new BitStream(seed);
  const decoder = new ArithmeticDecoder(stream);
  callbacks.onStatus(
    `Decoding: vocab=${vocabSize.toLocaleString()}, P=2^${Math.log2(probTotal)}, prec=${52}`,
  );

  const allIds = [...promptIds];
  let n = 0;
  while (n < MAX_TOKENS) {
    if (callbacks.shouldStop()) break;
    const logits = await getLastLogits(model, allIds);
    const cum = quantizeLogits(logits, vocabSize, probTotal);
    const tokId = decoder.decode(cum, probTotalBig);
    if (stopIds.has(tokId)) {
      callbacks.onStatus(`Done: ${n} tokens, hit EOS.`);
      break;
    }
    const piece = tokenizer.decode([tokId], { skip_special_tokens: false });
    allIds.push(tokId);
    n++;
    callbacks.onToken(piece, n);
    await new Promise((r) => setTimeout(r, 0));
  }
  if (n >= MAX_TOKENS)
    callbacks.onStatus(`Stopped at ${MAX_TOKENS} token limit.`);
}

// ============================================================
// PREFIX SEARCH — encode target tokens -> find smallest seed
// ============================================================
export async function prefixSearch(
  systemPrompt: string,
  userMessage: string,
  targetText: string,
  callbacks: SearchCallbacks,
): Promise<PrefixSearchResult | null> {
  const { tokenizer, model } = await loadModel(callbacks.onStatus);
  const promptIds = getPromptIds(tokenizer, systemPrompt, userMessage);
  const { vocabSize, probTotal, probTotalBig } = getModelParams(model);

  let targetIds: TokenId[];
  try {
    const enc = tokenizer(targetText, { add_special_tokens: false });
    targetIds = Array.isArray(enc.input_ids)
      ? enc.input_ids
      : Array.from(
          (enc.input_ids as { data?: ArrayLike<number> }).data ??
            (enc.input_ids as ArrayLike<number>),
        );
  } catch {
    // SWALLOW_EXCEPTION: tokenizer call syntax can vary; fall back to encode()
    const enc = tokenizer.encode(targetText);
    targetIds = Array.isArray(enc) ? enc : Array.from(enc);
  }
  targetIds = targetIds.map(Number);

  callbacks.onStatus(`Encoding ${targetIds.length} target tokens...`);
  const encoder = new ArithmeticEncoder();
  const ctx = [...promptIds];

  for (let i = 0; i < targetIds.length; i++) {
    if (callbacks.shouldStop()) return null;
    callbacks.onStatus(
      `Encoding token ${i + 1}/${targetIds.length}: "${tokenizer.decode([targetIds[i]])}"`,
    );
    const logits = await getLastLogits(model, ctx);
    const cum = quantizeLogits(logits, vocabSize, probTotal);
    encoder.encode(cum, probTotalBig, targetIds[i]);
    ctx.push(targetIds[i]);
    await new Promise((r) => setTimeout(r, 0));
  }

  const bits = encoder.finalize();
  const seed = bitsToSmallestBigInt(bits);
  const decoded = tokenizer.decode(targetIds, { skip_special_tokens: false });

  callbacks.onStatus(
    `Done: ${bits.length} bits -> ${seed.toString().length}-digit seed.`,
  );
  return {
    seed,
    bitLen: bits.length,
    tokenCount: targetIds.length,
    decodedTarget: decoded,
  };
}

// ============================================================
// SUBSTRING SEARCH — try random prefixes, encode prefix+target
// ============================================================
export async function substringSearch(
  systemPrompt: string,
  userMessage: string,
  targetText: string,
  numTrials: number,
  maxPrefixLen: number,
  callbacks: SearchCallbacks,
): Promise<SubstringSearchResult | null> {
  const { tokenizer, model } = await loadModel(callbacks.onStatus);
  const promptIds = getPromptIds(tokenizer, systemPrompt, userMessage);
  const stopIds = getStopIds(tokenizer);
  const { vocabSize, probTotal, probTotalBig } = getModelParams(model);

  let targetIds: TokenId[];
  try {
    const enc = tokenizer(targetText, { add_special_tokens: false });
    targetIds = Array.isArray(enc.input_ids)
      ? enc.input_ids
      : Array.from(
          (enc.input_ids as { data?: ArrayLike<number> }).data ??
            (enc.input_ids as ArrayLike<number>),
        );
  } catch {
    // SWALLOW_EXCEPTION: tokenizer call syntax can vary; fall back to encode()
    const enc = tokenizer.encode(targetText);
    targetIds = Array.isArray(enc) ? enc : Array.from(enc);
  }
  targetIds = targetIds.map(Number);

  let bestSeed: Seed | null = null;
  let bestBitLen = Infinity;
  let bestInfo = '';

  for (let trial = 0; trial < numTrials; trial++) {
    if (callbacks.shouldStop()) break;

    const prefixLen =
      3 + Math.floor(Math.random() * Math.max(1, maxPrefixLen - 2));
    callbacks.onStatus(
      `Trial ${trial + 1}/${numTrials}: sampling ${prefixLen}-token prefix...`,
    );

    // Phase 1: sample prefix tokens, cache cumulative weights
    const prefixIds: TokenId[] = [];
    const prefixCums: BigInt64Array[] = [];
    const ctx1 = [...promptIds];
    let aborted = false;

    for (let i = 0; i < prefixLen; i++) {
      const logits = await getLastLogits(model, ctx1);
      const cum = quantizeLogits(logits, vocabSize, probTotal);
      // Weighted random sample
      const r = BigInt(Math.floor(Math.random() * probTotal));
      let a = 0;
      let b = vocabSize - 1;
      while (a < b) {
        const m = (a + b) >>> 1;
        if (cum[m + 1] <= r) a = m + 1;
        else b = m;
      }
      if (stopIds.has(a)) {
        aborted = true;
        break;
      }
      prefixIds.push(a);
      prefixCums.push(cum);
      ctx1.push(a);
      await new Promise((r2) => setTimeout(r2, 0));
    }

    if (aborted || prefixIds.length < 2) continue;

    // Phase 2: encode prefix (cached) + target (fresh forward passes)
    callbacks.onStatus(
      `Trial ${trial + 1}/${numTrials}: encoding ${prefixIds.length}+${targetIds.length} tokens...`,
    );
    const encoder = new ArithmeticEncoder();

    for (let i = 0; i < prefixIds.length; i++) {
      encoder.encode(prefixCums[i], probTotalBig, prefixIds[i]);
    }

    const ctx2 = [...promptIds, ...prefixIds];
    for (let i = 0; i < targetIds.length; i++) {
      if (callbacks.shouldStop()) break;
      const logits = await getLastLogits(model, ctx2);
      const cum = quantizeLogits(logits, vocabSize, probTotal);
      encoder.encode(cum, probTotalBig, targetIds[i]);
      ctx2.push(targetIds[i]);
      await new Promise((r2) => setTimeout(r2, 0));
    }

    const bits = encoder.finalize();
    const seed = bitsToSmallestBigInt(bits);

    if (bits.length < bestBitLen) {
      bestBitLen = bits.length;
      bestSeed = seed;
      const prefixText = tokenizer.decode(prefixIds, {
        skip_special_tokens: true,
      });
      const targetDecoded = tokenizer.decode(targetIds, {
        skip_special_tokens: false,
      });
      bestInfo = `${bits.length} bits | prefix: "${prefixText.slice(0, 80)}" -> "${targetDecoded}"`;
      callbacks.onStatus(
        `Trial ${trial + 1}: new best — ${bits.length} bits (${seed.toString().length} digits)`,
      );
    }

    await new Promise((r2) => setTimeout(r2, 0));
  }

  if (bestSeed === null) return null;
  callbacks.onStatus(
    `Best: ${bestBitLen} bits -> ${bestSeed.toString().length}-digit seed.`,
  );
  return { seed: bestSeed, bitLen: bestBitLen, info: bestInfo };
}
