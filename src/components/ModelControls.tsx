import { useCallback } from 'react';
import { AVAILABLE_MODELS } from '../lib/model';
import type { ModelId, SamplerConfig } from '../lib/types';

interface Props {
  modelId: ModelId;
  onModelIdChange: (id: ModelId) => void;
  samplerConfig: SamplerConfig;
  onSamplerConfigChange: (config: SamplerConfig) => void;
}

export function ModelControls({
  modelId,
  onModelIdChange,
  samplerConfig,
  onSamplerConfigChange,
}: Props) {
  const handleTemperatureChange = useCallback(
    (val: number) => {
      onSamplerConfigChange({ ...samplerConfig, temperature: val });
    },
    [samplerConfig, onSamplerConfigChange],
  );

  const handleTopKChange = useCallback(
    (val: number) => {
      onSamplerConfigChange({ ...samplerConfig, topK: val });
    },
    [samplerConfig, onSamplerConfigChange],
  );

  return (
    <div className="model-controls">
      <div className="model-controls-row">
        <div className="field model-field">
          <label htmlFor="modelSelect">Model</label>
          <select
            id="modelSelect"
            value={modelId}
            onChange={(e) => onModelIdChange(e.target.value)}
          >
            {AVAILABLE_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} ({m.size})
              </option>
            ))}
          </select>
        </div>

        <div className="field slider-field">
          <label htmlFor="temperature">
            Temperature{' '}
            <span className="label-value">{samplerConfig.temperature.toFixed(2)}</span>
          </label>
          <input
            type="range"
            id="temperature"
            min="0.1"
            max="5.0"
            step="0.05"
            value={samplerConfig.temperature}
            onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
          />
          <div className="slider-labels">
            <span>sharp</span>
            <span>flat</span>
          </div>
        </div>

        <div className="field slider-field">
          <label htmlFor="topK">
            Top-K{' '}
            <span className="label-value">
              {samplerConfig.topK === 0 ? 'off' : samplerConfig.topK}
            </span>
          </label>
          <input
            type="range"
            id="topK"
            min="0"
            max="200"
            step="1"
            value={samplerConfig.topK}
            onChange={(e) => handleTopKChange(parseInt(e.target.value))}
          />
          <div className="slider-labels">
            <span>full vocab</span>
            <span>restrictive</span>
          </div>
        </div>
      </div>
    </div>
  );
}
