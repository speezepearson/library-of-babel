import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { FancyUI } from './components/FancyUI';
import './index.css';

const isFancy = window.location.hash === '#fancy';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isFancy ? <FancyUI /> : <App />}
  </StrictMode>,
);
