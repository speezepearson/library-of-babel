/** A natural number used as the seed for arithmetic decoding. */
export type Seed = bigint;

/** A token ID in the model's vocabulary. */
export type TokenId = number;

/** A single bit: 0 or 1. */
export type Bit = 0 | 1;

/** The size of the model's vocabulary. */
export type VocabSize = number;

/** HuggingFace model identifier. */
export type ModelId = string;

/** Controls how the probability distribution is shaped before arithmetic coding. */
export interface SamplerConfig {
  /** Softmax temperature. >1 = flatter/more diverse, <1 = sharper/more deterministic. */
  temperature: number;
  /** Keep only the top-K most probable tokens. 0 = disabled (use full vocab). */
  topK: number;
}

export const DEFAULT_SAMPLER_CONFIG: SamplerConfig = {
  temperature: 1.0,
  topK: 0,
};
