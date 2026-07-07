import { useEffect, useMemo, useRef, useState } from 'react';
import { predict } from '../api.js';
import { fieldMeta, categoryValueLabel } from '../fieldMeta.js';
import { getFieldGroups } from '../formLayout.js';
import { getModelLabel } from '../tasks.js';
import ResultCard from './ResultCard.jsx';

function emptyForm(model) {
  const values = {};
  for (const f of model.numeric_features) values[f] = '';
  for (const f of model.categorical_features) {
    const options = model.categorical_values?.[f] || [];
    values[f] = options[0] || '';
  }
  return values;
}

// Formulario dinamico: los campos salen del esquema de features publicado
// por el backend en /models, agrupados y traducidos a espanol llano para
// que el personal del albergue (sin conocimientos de ML) pueda usarlo.
export default function PredictPanel({ models, taskTitle }) {
  const [modelName, setModelName] = useState(models[0]?.name || '');
  const model = useMemo(
    () => models.find((m) => m.name === modelName) || models[0],
    [models, modelName],
  );
  const [form, setForm] = useState(() => (model ? emptyForm(model) : {}));
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const resultRef = useRef(null);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [result]);

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
        setError(`El campo "${fieldMeta(f).label}" debe ser un número.`);
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

  const allFields = [...model.numeric_features, ...model.categorical_features];
  const groups = getFieldGroups(model, allFields);

  return (
    <section className="card">
      <h2>{taskTitle}</h2>
      <p className="subtitle">Ingresa los datos solicitados y presiona «Evaluar» para ver el resultado.</p>

      {models.length > 1 && (
        <div className="model-picker">
          <label htmlFor="model-select">¿Qué quieres calcular?</label>
          <select
            id="model-select"
            value={modelName}
            onChange={(e) => selectModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {getModelLabel(m)}
              </option>
            ))}
          </select>
        </div>
      )}

      <form onSubmit={submit}>
        {groups.map((group) => (
          <fieldset className="form-section" key={group.title}>
            <legend>{group.title}</legend>
            <div className="form-grid">
              {group.fields.map((f) => {
                const isNumeric = model.numeric_features.includes(f);
                const meta = fieldMeta(f);
                return (
                  <div className="field" key={f}>
                    <label htmlFor={`f-${f}`}>{meta.label}</label>
                    {isNumeric ? (
                      <input
                        id={`f-${f}`}
                        type="number"
                        step="any"
                        placeholder={meta.helper}
                        value={form[f]}
                        onChange={(e) => setField(f, e.target.value)}
                        required
                      />
                    ) : (
                      <select
                        id={`f-${f}`}
                        value={form[f]}
                        onChange={(e) => setField(f, e.target.value)}
                      >
                        {(model.categorical_values?.[f] || []).map((option) => (
                          <option key={option} value={option}>
                            {categoryValueLabel(f, option)}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                );
              })}
            </div>
          </fieldset>
        ))}
        <button type="submit" disabled={loading}>
          {loading && <span className="spinner" aria-hidden="true" />}
          {loading ? 'Evaluando…' : 'Evaluar'}
        </button>
      </form>

      {error && <p className="status error">{error}</p>}

      {result && (
        <div ref={resultRef}>
          <ResultCard model={model} result={result} />
        </div>
      )}
    </section>
  );
}
