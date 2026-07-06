import HBarChart from './HBarChart.jsx';

const TASK_LABELS = {
  classification: 'Clasificación',
  regression: 'Regresión',
};

// Tabla de métricas reales (test) por modelo + comparación visual del
// F1-macro de los clasificadores. Cada valor sale de model_registry.json,
// generado por una ejecución real de entrenamiento.
export default function ModelsOverview({ models }) {
  const classifiers = models.filter((m) => m.task === 'classification');
  const regressors = models.filter((m) => m.task === 'regression');

  return (
    <>
      <section className="card">
        <h2>Modelos desplegados</h2>
        <p className="subtitle">
          Algoritmo ganador por frente (Decision Tree vs Random Forest vs XGBoost) y su
          métrica de selección en el conjunto de test.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Tarea</th>
                <th>Algoritmo</th>
                <th className="num">Métrica de selección</th>
                <th className="num">Valor (test)</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td>{m.description}</td>
                  <td>{TASK_LABELS[m.task] || m.task}</td>
                  <td>{m.selected_algorithm}</td>
                  <td className="num">{m.selection_metric}</td>
                  <td className="num">
                    {Number(m.metrics[m.selection_metric]).toFixed(
                      m.task === 'classification' ? 3 : 2,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {classifiers.length > 0 && (
        <section className="card">
          <h2>F1-macro en test — modelos de clasificación</h2>
          <p className="subtitle">Escala 0 a 1; mayor es mejor.</p>
          <HBarChart
            items={classifiers.map((m) => ({
              label: m.name,
              value: Number(m.metrics.f1_macro),
            }))}
            max={1}
          />
        </section>
      )}

      {regressors.length > 0 && (
        <section className="card">
          <h2>Error absoluto medio (MAE) en test — modelos de regresión</h2>
          <p className="subtitle">
            En la unidad de cada target (unidades consumidas, días, soles); menor es mejor.
            Se muestra en escala propia por modelo.
          </p>
          <HBarChart
            items={regressors.map((m) => ({
              label: m.name,
              value: Number(m.metrics.mae),
            }))}
            max={Math.max(...regressors.map((m) => Number(m.metrics.mae)))}
            format={(v) => v.toFixed(2)}
          />
        </section>
      )}
    </>
  );
}
