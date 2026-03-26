import { useState, useCallback } from 'react';
import { DecodePanel } from './components/DecodePanel';
import { SearchPanel } from './components/SearchPanel';
import './App.css';

type Tab = 'decode' | 'search';

function App() {
  const [systemPrompt, setSystemPrompt] = useState(
    'You are a helpful, concise assistant.',
  );
  const [userMessage, setUserMessage] = useState(
    'Tell me something interesting.',
  );
  const [activeTab, setActiveTab] = useState<Tab>('decode');
  const [seedFromSearch, setSeedFromSearch] = useState<string | null>(null);

  const handleUseSeed = useCallback((seed: string) => {
    setSeedFromSearch(seed);
    setActiveTab('decode');
  }, []);

  const handleSeedConsumed = useCallback(() => {
    setSeedFromSearch(null);
  }, []);

  return (
    <div className="container">
      <header>
        <h1>arithmetic_llm_decoder</h1>
        <p className="tagline">
          A natural number deterministically selects text from a language model
          via arithmetic coding. Same seed, same prompt &rarr; same output.
        </p>
      </header>

      <div className="field">
        <label htmlFor="systemPrompt">System Prompt</label>
        <textarea
          id="systemPrompt"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
      </div>
      <div className="field">
        <label htmlFor="userMessage">User Message</label>
        <textarea
          id="userMessage"
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
        />
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'decode' ? 'active' : ''}`}
          onClick={() => setActiveTab('decode')}
        >
          Decode
        </button>
        <button
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Search
        </button>
      </div>

      {activeTab === 'decode' && (
        <DecodePanel
          systemPrompt={systemPrompt}
          userMessage={userMessage}
          initialSeed={seedFromSearch}
          onSeedConsumed={handleSeedConsumed}
        />
      )}
      {activeTab === 'search' && (
        <SearchPanel
          systemPrompt={systemPrompt}
          userMessage={userMessage}
          onUseSeed={handleUseSeed}
        />
      )}
    </div>
  );
}

export default App;
