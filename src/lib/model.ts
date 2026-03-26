import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
  env,
  type PreTrainedTokenizer,
  type PreTrainedModel,
} from '@huggingface/transformers';
import type { TokenId, VocabSize, ProbTotal } from './types';

export const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct';

export interface ModelParams {
  vocabSize: VocabSize;
  probTotal: ProbTotal;
  probTotalBig: bigint;
}

export type StatusCallback = (message: string) => void;

export interface ModelBundle {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}

let _tok: PreTrainedTokenizer | null = null;
let _mod: PreTrainedModel | null = null;

/**
 * Configure the transformers.js environment for Node.js (tests / download script).
 * Must be called before loadModel().
 */
export function configureForNode(cacheDir: string): void {
  env.cacheDir = cacheDir;
  env.allowRemoteModels = false;
}

/** Reset cached model/tokenizer (for test isolation). */
export function resetModel(): void {
  _tok = null;
  _mod = null;
}

export async function loadModel(onStatus: StatusCallback): Promise<ModelBundle> {
  if (_tok && _mod) return { tokenizer: _tok, model: _mod };

  onStatus('Loading tokenizer...');
  _tok = await AutoTokenizer.from_pretrained(MODEL_ID);

  onStatus('Loading model weights...');
  const pcb = (p: { status: string; loaded?: number; total?: number }) => {
    if (p.status === 'progress' && p.total) {
      const pct = Math.round(((p.loaded ?? 0) / p.total) * 100);
      onStatus(`Downloading: ${((p.loaded ?? 0) / 1e6).toFixed(1)} MB (${pct}%)`);
    }
  };

  try {
    _mod = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: 'fp16',
      device: 'webgpu',
      progress_callback: pcb,
    });
    onStatus('Model loaded (WebGPU).');
  } catch {
    // SWALLOW_EXCEPTION: WebGPU not available (e.g. Node.js, older browsers) — fall through to WASM/default
    onStatus('WebGPU unavailable — trying WASM...');
    try {
      _mod = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        dtype: 'fp32',
        device: 'wasm',
        progress_callback: pcb,
      });
    } catch {
      // SWALLOW_EXCEPTION: WASM not available — try default device
      _mod = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        progress_callback: pcb,
      });
    }
    onStatus('Model loaded (WASM).');
  }

  return { tokenizer: _tok, model: _mod };
}

export function getPromptIds(
  tokenizer: PreTrainedTokenizer,
  systemPrompt: string,
  userMessage: string,
): TokenId[] {
  const msgs = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userMessage },
  ];
  let ids: number[];
  try {
    const result = tokenizer.apply_chat_template(msgs, {
      add_generation_prompt: true,
    });
    if (!Array.isArray(result)) {
      ids = Array.from(
        (result as unknown as { data?: ArrayLike<number> }).data ??
          (result as unknown as ArrayLike<number>),
      );
    } else {
      ids = result.flat();
    }
  } catch {
    // SWALLOW_EXCEPTION: apply_chat_template can fail for some tokenizer configs; fall back to manual template
    const raw =
      `<|im_start|>system\n${systemPrompt}<|im_end|>\n` +
      `<|im_start|>user\n${userMessage}<|im_end|>\n` +
      `<|im_start|>assistant\n`;
    const enc = tokenizer(raw);
    const inputIds = enc.input_ids;
    ids = Array.isArray(inputIds)
      ? inputIds
      : Array.from(
          (inputIds as { data?: ArrayLike<number> }).data ??
            (inputIds as ArrayLike<number>),
        );
  }
  return ids.map(Number);
}

export function getStopIds(tokenizer: PreTrainedTokenizer): Set<TokenId> {
  const s = new Set<TokenId>();
  if (tokenizer.eos_token_id != null) s.add(Number(tokenizer.eos_token_id));
  try {
    const e = tokenizer.encode('<|im_end|>');
    if (e?.length) s.add(e[e.length - 1]);
  } catch {
    // SWALLOW_EXCEPTION: token may not exist in this model's vocabulary
  }
  try {
    const e = tokenizer.encode('<|endoftext|>');
    if (e?.length) s.add(e[e.length - 1]);
  } catch {
    // SWALLOW_EXCEPTION: token may not exist in this model's vocabulary
  }
  return s;
}

export function getModelParams(model: PreTrainedModel): ModelParams {
  // DEFAULT_VALUE: SmolLM2 vocab size is 49152; config.vocab_size should always be present
  const vocabSize: VocabSize =
    (model.config as { vocab_size?: number })?.vocab_size ?? 49152;
  let probBits = 18;
  while ((1 << probBits) <= vocabSize) probBits++;
  const probTotal: ProbTotal = 1 << probBits;
  return { vocabSize, probTotal, probTotalBig: BigInt(probTotal) };
}

export async function getLastLogits(
  model: PreTrainedModel,
  ids: TokenId[],
): Promise<Float32Array> {
  const seqLength = ids.length;
  const inp = new Tensor(
    'int64',
    new BigInt64Array(ids.map((id) => BigInt(id))),
    [1, seqLength],
  );
  const maskData = new BigInt64Array(seqLength).fill(1n);
  const attention_mask = new Tensor('int64', maskData, [1, seqLength]);
  const posData = new BigInt64Array(seqLength);
  for (let i = 0; i < seqLength; i++) posData[i] = BigInt(i);
  const position_ids = new Tensor('int64', posData, [1, seqLength]);

  const out = await model({ input_ids: inp, attention_mask, position_ids });
  const logits = out.logits as Tensor;
  const [, seqLen, vSz] = logits.dims;
  return (logits.data as Float32Array).subarray(
    (seqLen - 1) * vSz,
    seqLen * vSz,
  );
}
