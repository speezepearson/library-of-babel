import { AutoTokenizer, AutoModelForCausalLM, env } from '@huggingface/transformers';
import fs from 'node:fs';

const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const CACHE_DIR = './models';

// Skip if model is already cached (check for config.json as sentinel)
const sentinelDir = `${CACHE_DIR}/${MODEL_ID}`;
if (fs.existsSync(sentinelDir)) {
  console.log(`Model already cached in ${sentinelDir}, skipping download.`);
  process.exit(0);
}

env.cacheDir = CACHE_DIR;
env.allowRemoteModels = true;

console.log(`Downloading ${MODEL_ID} to ${CACHE_DIR}/ ...`);

await AutoTokenizer.from_pretrained(MODEL_ID);
console.log('Tokenizer downloaded.');

await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  progress_callback: (p) => {
    if (p.status === 'progress' && p.total) {
      const pct = Math.round((p.loaded / p.total) * 100);
      process.stdout.write(
        `\rDownloading model: ${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB (${pct}%)`,
      );
    }
    if (p.status === 'done') {
      process.stdout.write('\n');
    }
  },
});
console.log('Model downloaded.');

// ONNX runtime sessions can keep the event loop alive
process.exit(0);
