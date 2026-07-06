import { useEffect, useState } from 'react';
import { fetchModels } from './api.js';
import ModelsOverview from './components/ModelsOverview.jsx';
import PredictPanel from './components/PredictPanel.jsx';

export default function App() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data.models))
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>ALDIMI Predict</h1>
        <p>
          Dashboard de los modelos de Machine Learning del albergue: prioridad de
          atención de pacientes, alertas de stock, demanda de consumo, estadía y
          proyección de donaciones.
        </p>
      </header>

      {error && (
        <p className="status error">
          {error} — verifica que el backend esté corriendo y que VITE_API_URL apunte a él.
        </p>
      )}
      {!models && !error && <p className="status">Cargando modelos…</p>}
      {models && (
        <>
          <ModelsOverview models={models} />
          <PredictPanel models={models} />
        </>
      )}
    </div>
  );
}
