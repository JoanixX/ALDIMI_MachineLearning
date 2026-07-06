import { useMemo, useState } from 'react';
import { predict } from '../api.js';
import HBarChart from './HBarChart.jsx';

function emptyForm(model) {
  const values = {};
  for (const f of model.numeric_features) values[f] = '';
  for (const f of model.categorical_features) {
    const options = model.categorical_values?.[f] || [];
    values[f] = options[0] || '';
  }
  return values;
}

// Formulario dinámico: los campos salen del esquema de features publicado
// por el backend en /models, así el frontend nunca se desincroniza del modelo.
export default function PredictPanel({ models }) {
  const [modelName, setModelName] = useState(models[0]?.name || '');
  const model = useMemo(
    () => models.find((m) => m.name === modelName),
    [models, modelName],
  );
  const [form, setForm] = useState(() => (model ? emptyForm(model) : {}));
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  if (!model) return null;

  const selectModel = (name) => {
    const next = models.find((m) => m.name === name);
    setModelName(name);
    setForm(next ? emptyForm(next) : {});
    setResult(null);
    setError(null);
  };

  const setField = (name, value) => setForm((prev) => ({ ...prev, [name]: value }));

  const submit = async (event) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    const payload = {};
    for (const f of model.numeric_features) {
      const parsed = Number(form[f]);
      if (form[f] === '' || Number.isNaN(parsed)) {
        setError(`El campo "${f}" debe ser un número.`);
        return;
      }
      payload[f] = parsed;
    }
    for (const f of model.categorical_features) payload[f] = form[f];

    setLoading(true);
    try {
      const response = await predict(model.name, payload);
      setResult(response.predictions[0]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isClassification = model.task === 'classification';

  return (
    <section className="card">
      <h2>Predicción interactiva</h2>
      <p className="subtitle">
        La inferencia corre en el backend Rust sobre el modelo ONNX exportado desde Python.
      </p>

      <div className="model-picker">
        <label htmlFor="model-select">Modelo</label>
        <select
          id="model-select"
          value={modelName}
          onChange={(e) => selectModel(e.target.value)}
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.description}
            </option>
          ))}
        </select>
      </div>

      <form onSubmit={submit}>
        <div className="form-grid">
          {model.numeric_features.map((f) => (
            <div className="field" key={f}>
              <label htmlFor={`f-${f}`}>{f}</label>
              <input
                id={`f-${f}`}
                type="number"
                step="any"
                value={form[f]}
                onChange={(e) => setField(f, e.target.value)}
                required
              />
            </div>
          ))}
          {model.categorical_features.map((f) => (
            <div className="field" key={f}>
              <label htmlFor={`f-${f}`}>{f}</label>
              <select
                id={`f-${f}`}
                value={form[f]}
                onChange={(e) => setField(f, e.target.value)}
              >
                {(model.categorical_values?.[f] || []).map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
        <button type="submit" disabled={loading}>
          {loading ? 'Prediciendo…' : 'Predecir'}
        </button>
      </form>

      {error && <p className="status error">{error}</p>}

      {result && (
        <div className="result">
          {isClassification ? (
            <>
              <div className="result-headline">
                {result.prediccion}
                <span className="badge">{model.selected_algorithm}</span>
              </div>
              <p className="result-caption">Probabilidad por clase</p>
              <HBarChart
                items={(model.class_labels || []).map((label) => ({
                  label,
                  value: result.probabilidades[label] ?? 0,
                }))}
                max={1}
                format={(v) => `${(v * 100).toFixed(1)}%`}
              />
            </>
          ) : (
            <>
              <div className="result-headline">
                {result.prediccion.toFixed(2)}
                <span className="badge">{model.selected_algorithm}</span>
              </div>
              <p className="result-caption">{model.description}</p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
