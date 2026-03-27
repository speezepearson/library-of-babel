import type { TokenId, VocabSize } from './types';

/**
 * Client for llama.cpp's llama-server HTTP API.
 *
 * The Vite dev proxy forwards /llamacpp/* to localhost:8080/* to avoid CORS.
 * Start llama-server with your model, e.g.:
 *   llama-server -m model.gguf -c 4096 --port 8080
 */

const BASE = '/llamacpp';

// ---------------------------------------------------------------------------
// Types for llama-server JSON responses
// ---------------------------------------------------------------------------

interface LlamaCppLogprobEntry {
  /** Token ID. */
  id: number;
  /** Token string. */
  token: string;
  /** Log-probability. */
  logprob: number;
}

interface LlamaCppCompletionProb {
  id: number;
  token: string;
  top_logprobs: LlamaCppLogprobEntry[];
}

interface LlamaCppCompletionResponse {
  content: string;
  stop: boolean;
  tokens_predicted: number;
  completion_probabilities?: LlamaCppCompletionProb[];
}

// ---------------------------------------------------------------------------
// Cached model info
// ---------------------------------------------------------------------------

export interface LlamaCppModelInfo {
  vocabSize: VocabSize;
}

let _cachedInfo: LlamaCppModelInfo | null = null;

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Cannot reach llama-server at ${BASE}${path}! Is it running? (${(err as Error).message})`,
    );
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`llama-server error ${resp.status}: ${text.slice(0, 300)}`);
  }
  return resp.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function llamaCppTokenize(text: string): Promise<TokenId[]> {
  const data = await post<{ tokens: number[] }>('/tokenize', { content: text });
  return data.tokens;
}

export async function llamaCppDetokenize(tokens: TokenId[]): Promise<string> {
  const data = await post<{ content: string }>('/detokenize', { tokens });
  return data.content;
}

/**
 * Discover the model's vocab size by making a probe completion request
 * with n_probs set very high (the server clamps to vocab size).
 */
export async function getLlamaCppModelInfo(
  promptTokens: TokenId[],
): Promise<LlamaCppModelInfo> {
  if (_cachedInfo) return _cachedInfo;

  // Request one token with maximum probs — server will clamp n_probs to vocab_size
  const data = await post<LlamaCppCompletionResponse>('/completion', {
    prompt: promptTokens,
    n_predict: 1,
    n_probs: 999999,
    temperature: 1.0,
    cache_prompt: false,
  });

  const topLogprobs = data.completion_probabilities?.[0]?.top_logprobs ?? [];
  if (topLogprobs.length === 0) {
    throw new Error(
      'llama-server returned no probabilities. Make sure n_probs is supported.',
    );
  }

  // The number of entries returned = vocab_size (server clamps n_probs)
  const vocabSize = topLogprobs.length as VocabSize;

  _cachedInfo = { vocabSize };
  return _cachedInfo;
}

export function resetLlamaCppCache(): void {
  _cachedInfo = null;
}

/**
 * Get the next-token logit vector from llama-server.
 *
 * Calls /completion with n_predict=1 and n_probs=vocabSize.
 * Converts the returned probabilities to log-probabilities (our "logits").
 *
 * With cache_prompt=true, llama-server reuses KV cache for the prefix,
 * so each call only processes the last new token — fast for sequential decoding.
 */
export async function getLlamaCppLogits(
  contextTokens: TokenId[],
  vocabSize: VocabSize,
): Promise<Float32Array> {
  const data = await post<LlamaCppCompletionResponse>('/completion', {
    prompt: contextTokens,
    n_predict: 1,
    n_probs: vocabSize,
    temperature: 1.0, // No temperature — we apply our own in quantizeLogits
    top_k: 0, // No filtering — we need the full distribution
    top_p: 1.0,
    min_p: 0.0, // Disable min_p filtering (server default is 0.05)
    cache_prompt: false, // KV cache reuse causes nondeterministic logprobs
  });

  const topLogprobs = data.completion_probabilities?.[0]?.top_logprobs ?? [];

  // Build dense logit array from log-probabilities, -Infinity for tokens not in top_logprobs
  const logits = new Float32Array(vocabSize);
  logits.fill(-Infinity);

  for (let i = 0; i < topLogprobs.length; i++) {
    const entry = topLogprobs[i];
    if (entry.id >= 0 && entry.id < vocabSize) {
      logits[entry.id] = entry.logprob;
    }
  }

  return logits;
}

/**
 * Build a set of stop token IDs by tokenizing common end-of-sequence markers.
 */
export async function getLlamaCppStopIds(): Promise<Set<TokenId>> {
  const markers = ['<|endoftext|>', '</s>', '<|im_end|>', '<|end|>', '<eos>', '<|eot_id|>'];
  const stopIds = new Set<TokenId>();
  for (const marker of markers) {
    try {
      const tokens = await llamaCppTokenize(marker);
      // Single-token markers are likely actual special tokens
      if (tokens.length === 1) stopIds.add(tokens[0]);
    } catch {
      // SWALLOW_EXCEPTION: marker may not tokenize cleanly in this model's vocab
    }
  }
  return stopIds;
}

/**
 * Format system+user messages into a ChatML prompt string.
 * llama-server will tokenize this using the model's vocabulary.
 *
 * Note: if the model uses a different chat template, the user should
 * start llama-server with --chat-template to match, or adjust prompts.
 */
export function formatChatPrompt(
  systemPrompt: string,
  userMessage: string,
): string {
  return (
    `<|im_start|>system\n${systemPrompt}<|im_end|>\n` +
    `<|im_start|>user\n${userMessage}<|im_end|>\n` +
    `<|im_start|>assistant\n`
  );
}
