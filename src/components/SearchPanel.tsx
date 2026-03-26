import { useState, useRef, useCallback } from 'react';
import { prefixSearch, substringSearch } from '../lib/decode';
import type { Seed, ModelId, SamplerConfig } from '../lib/types';

type SearchMode = 'prefix' | 'substring';

interface Props {
  systemPrompt: string;
  userMessage: string;
  onUseSeed: (seed: string) => void;
  modelId: ModelId;
  samplerConfig: SamplerConfig;
}

export function SearchPanel({ systemPrompt, userMessage, onUseSeed, modelId, samplerConfig }: Props) {
  const [target, setTarget] = useState('The answer is 42.');
  const [mode, setMode] = useState<SearchMode>('prefix');
  const [maxPrefixLen, setMaxPrefixLen] = useState(15);
  const [numTrials, setNumTrials] = useState(12);
  const [isSearching, setIsSearching] = useState(false);
  const [status, setStatus] = useState('Ready.');
  const [spinning, setSpinning] = useState(false);
  const [result, setResult] = useState<{
    seed: Seed;
    info: string;
  } | null>(null);
  const stopRef = useRef(false);

  const handleSearch = useCallback(async () => {
    if (!target.trim()) {
      setStatus('Enter target text.');
      setSpinning(false);
      return;
    }

    setIsSearching(true);
    stopRef.current = false;
    setResult(null);

    const callbacks = {
      onStatus(msg: string) {
        setStatus(msg);
        setSpinning(true);
      },
      shouldStop() {
        return stopRef.current;
      },
    };

    try {
      if (mode === 'prefix') {
        const r = await prefixSearch(
          systemPrompt,
          userMessage,
          target,
          callbacks,
          modelId,
          samplerConfig,
        );
        if (r) {
          setResult({
            seed: r.seed,
            info: `${r.bitLen} bits, ${r.tokenCount} tokens`,
          });
          setStatus('Search complete.');
        } else {
          setStatus('Search stopped or no result found.');
        }
      } else {
        const r = await substringSearch(
          systemPrompt,
          userMessage,
          target,
          numTrials,
          maxPrefixLen,
          callbacks,
          modelId,
          samplerConfig,
        );
        if (r) {
          setResult({ seed: r.seed, info: r.info });
          setStatus('Search complete.');
        } else {
          setStatus('Search stopped or no result found.');
        }
      }
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
      console.error(err);
    }

    setSpinning(false);
    setIsSearching(false);
  }, [target, mode, numTrials, maxPrefixLen, systemPrompt, userMessage, modelId, samplerConfig]);

  return (
    <>
      <p className="tagline" style={{ marginBottom: '1.5rem' }}>
        Find a seed whose decoded output contains your target text. Prefix
        search is exact and fast. Substring search tries random prefixes to find
        a natural lead-in.
      </p>

      <div className="field">
        <label htmlFor="searchTarget">Target Text</label>
        <textarea
          id="searchTarget"
          style={{ height: '2.6rem' }}
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <div className="hint">
          The text you want to appear in the decoded output.
        </div>
      </div>

      <div className="tabs" style={{ marginTop: '1rem' }}>
        <button
          className={`tab ${mode === 'prefix' ? 'active' : ''}`}
          onClick={() => setMode('prefix')}
        >
          Prefix Search
        </button>
        <button
          className={`tab ${mode === 'substring' ? 'active' : ''}`}
          onClick={() => setMode('substring')}
        >
          Substring Search
        </button>
      </div>

      {mode === 'substring' && (
        <>
          <div className="inline-fields">
            <div className="field">
              <label htmlFor="maxPrefixLen">Max prefix tokens</label>
              <input
                type="number"
                id="maxPrefixLen"
                value={maxPrefixLen}
                min={3}
                max={40}
                onChange={(e) => setMaxPrefixLen(parseInt(e.target.value) || 15)}
              />
            </div>
            <div className="field">
              <label htmlFor="numTrials">Trials</label>
              <input
                type="number"
                id="numTrials"
                value={numTrials}
                min={1}
                max={100}
                onChange={(e) => setNumTrials(parseInt(e.target.value) || 12)}
              />
            </div>
          </div>
          <div className="hint" style={{ marginBottom: '1rem' }}>
            More trials &amp; longer prefixes = smoother lead-in, but slower
            search.
          </div>
        </>
      )}

      <div className="controls">
        <button
          className="btn-primary"
          disabled={isSearching}
          onClick={handleSearch}
        >
          Search
        </button>
        {isSearching && (
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

      {result && (
        <div className="result-box">
          <div className="result-label">Seed found</div>
          <div className="result-seed">{result.seed.toString()}</div>
          <div className="result-info">{result.info}</div>
          <div className="result-actions">
            <button
              className="btn-green"
              onClick={() => onUseSeed(result.seed.toString())}
            >
              Decode this seed &rarr;
            </button>
          </div>
        </div>
      )}
    </>
  );
}
