import { useEffect, useState } from 'react';
import { fetchModels } from './api.js';
import { TASKS } from './tasks.js';
import TaskMenu from './components/TaskMenu.jsx';
import PredictPanel from './components/PredictPanel.jsx';
import ModelsOverview from './components/ModelsOverview.jsx';
import HowItWorks from './components/HowItWorks.jsx';
import FAQ from './components/FAQ.jsx';
import HelpCard from './components/HelpCard.jsx';
import ThemeToggle from './components/ThemeToggle.jsx';

const THEME_KEY = 'aldimi-theme';

function getPreferredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  const [activeTaskKey, setActiveTaskKey] = useState(null);
  const [theme, setTheme] = useState(getPreferredTheme);

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data.models))
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'));

  const task = TASKS.find((t) => t.key === activeTaskKey);
  const taskModels =
    task && models
      ? task.models.map((name) => models.find((m) => m.name === name)).filter(Boolean)
      : [];

  return (
    <div className="app">
      <header className="app-header">
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
        <div className="brand">
          <svg
            className="brand-mark"
            width="30"
            height="30"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M12 20.2s-7.4-4.4-7.4-9.7A4.1 4.1 0 0 1 12 7.9a4.1 4.1 0 0 1 7.4 2.6c0 5.3-7.4 9.7-7.4 9.7z" />
            <path d="M8 11.7h2l1-2 2 4 1-2h2" />
          </svg>
          <h1>ALDIMI Predict</h1>
        </div>
        <p>
          Herramienta de apoyo para el personal del albergue: evalúa pacientes, revisa el
          inventario y consulta la proyección de donaciones.
        </p>
      </header>

      {error && (
        <p className="status error">
          {error} — verifica que el sistema esté disponible e inténtalo de nuevo.
        </p>
      )}
      {!models && !error && <p className="status">Cargando…</p>}

      {models && !task && (
        <>
          <HowItWorks />
          <TaskMenu tasks={TASKS} onSelect={setActiveTaskKey} />
          <FAQ />
          <HelpCard />
          <details className="technical-details">
            <summary>Detalles técnicos (equipo de ML)</summary>
            <ModelsOverview models={models} />
          </details>
        </>
      )}

      {models && task && (
        <>
          <button type="button" className="back-link" onClick={() => setActiveTaskKey(null)}>
            ‹ Volver al inicio
          </button>
          <PredictPanel models={taskModels} taskTitle={task.title} />
        </>
      )}

      <footer className="app-footer">
        <p>ALDIMI Predict — Herramienta de apoyo para el albergue, desarrollada por el equipo de Machine Learning.</p>
      </footer>
    </div>
  );
}
