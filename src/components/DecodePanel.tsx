import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { decode } from '../lib/decode';
import { BitStream } from '../lib/arithmetic';
import { isLlamaCppModel } from '../lib/model';
import type { Seed, ModelId, SamplerConfig } from '../lib/types';

const EXAMPLE_SEEDS = [
  { label: '0', value: '0' },
  { label: '1', value: '1' },
  { label: '42', value: '42' },
  { label: '10^12 - 1', value: '999999999999' },
  {
    label: '2^256 - 1',
    value:
      '115792089237316195423570985008687907853269984665640564039457584007913129639935',
  },
];

interface Props {
  systemPrompt: string;
  userMessage: string;
  initialSeed: string | null;
  onSeedConsumed: () => void;
  modelId: ModelId;
  samplerConfig: SamplerConfig;
}

export function DecodePanel({
  systemPrompt,
  userMessage,
  initialSeed,
  onSeedConsumed,
  modelId,
  samplerConfig,
}: Props) {
  const [seedInput, setSeedInput] = useState(
    '31415926535897932384626433832795028841971',
  );
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState(
    'Ready. Click Generate to load the model (~270 MB on first visit).',
  );
  const [spinning, setSpinning] = useState(false);
  const [output, setOutput] = useState<string[]>([]);
  const [tokenCount, setTokenCount] = useState(0);
  const [rational, setRational] = useState('');
  const stopRef = useRef(false);

  const liveBits = useMemo(() => {
    try {
      const n = BigInt(seedInput.trim());
      if (n < 0n) return null;
      const bs = new BitStream(n);
      const bits: number[] = [];
      for (let i = 0; i < 100; i++) bits.push(bs.readBit());
      return bits.join('');
    } catch {
      return null;
    }
  }, [seedInput]);

  useEffect(() => {
    if (initialSeed !== null) {
      setSeedInput(initialSeed);
      onSeedConsumed();
    }
  }, [initialSeed, onSeedConsumed]);

  const handleGenerate = useCallback(async () => {
    let seed: Seed;
    try {
      seed = BigInt(seedInput.trim());
      if (seed < 0n) throw new Error('negative');
    } catch {
      setStatus('Invalid seed.');
      setSpinning(false);
      return;
    }

    setIsRunning(true);
    stopRef.current = false;
    setOutput([]);
    setTokenCount(0);

    if (isLlamaCppModel(modelId)) {
      setRational('llama.cpp server (arithmetic coding via HTTP)');
    } else {
      const bs = new BitStream(seed);
      setRational(`rational = ${bs.describe()}`);
    }

    try {
      await decode(systemPrompt, userMessage, seed, {
        onToken(piece, n) {
          setOutput((prev) => [...prev, piece]);
          setTokenCount(n);
        },
        onStatus(msg) {
          setStatus(msg);
          setSpinning(true);
        },
        shouldStop() {
          return stopRef.current;
        },
      }, modelId, samplerConfig);
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
      console.error(err);
    }

    setSpinning(false);
    setIsRunning(false);
  }, [seedInput, systemPrompt, userMessage, modelId, samplerConfig]);

  return (
    <>
      <div className="field">
        <label htmlFor="bigintInput">Seed (natural number)</label>
        <input
          type="text"
          id="bigintInput"
          value={seedInput}
          onChange={(e) => setSeedInput(e.target.value)}
        />
        <div className="hint">
          Any non-negative integer. Different seeds produce different
          completions.
        </div>
        {liveBits && (
          <div className="bit-preview">0.{liveBits}</div>
        )}
      </div>

      <div className="controls">
        <button
          className="btn-primary"
          disabled={isRunning}
          onClick={handleGenerate}
        >
          Generate
        </button>
        {isRunning && (
          <button
            className="btn-danger"
            onClick={() => {
              stopRef.current = true;
              setStatus('Stopping...');
            }}
          >
            Stop
          </button>
        )}
      </div>

      <div className="status-line">
        {spinning && <span className="spinner" />}
        {status}
      </div>

      <div className="output-area">
        <div className="output-header">
          <span>decoded output</span>
          <span>{tokenCount > 0 ? `${tokenCount} tok` : ''}</span>
        </div>
        <div className="output-body">
          {output.length === 0 ? (
            <span className="placeholder">Output will appear here...</span>
          ) : (
            <>
              {output.join('')}
              {isRunning && <span className="cursor" />}
            </>
          )}
        </div>
        {rational && <div className="rational-display">{rational}</div>}
      </div>

      <div className="examples">
        <h3>Try a seed</h3>
        <div className="example-chips">
          {EXAMPLE_SEEDS.map((ex) => (
            <span
              key={ex.value}
              className="chip"
              onClick={() => setSeedInput(ex.value)}
            >
              {ex.label}
            </span>
          ))}
        </div>
      </div>
    </>
  );
}
