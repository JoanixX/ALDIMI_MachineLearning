// Traduce la prediccion cruda del backend (clase o numero) a un mensaje en
// lenguaje humano, con color semantico, para personal sin conocimientos de ML.
const CLASS_COPY = {
  prioridad_atencion: {
    Bajo: {
      color: 'green',
      eyebrow: 'Prioridad',
      headline: 'Prioridad BAJA',
      message: 'Este paciente puede continuar con el seguimiento habitual.',
    },
    Medio: {
      color: 'amber',
      eyebrow: 'Prioridad',
      headline: 'Prioridad MEDIA',
      message: 'Este paciente requiere seguimiento cercano en los próximos días.',
    },
    Alto: {
      color: 'red',
      eyebrow: 'Prioridad',
      headline: 'Prioridad ALTA — este paciente necesita atención inmediata',
      message: 'Coordina cuanto antes con el personal médico del albergue.',
    },
  },
  stock_critico_7d: {
    normal: {
      color: 'green',
      eyebrow: 'Stock',
      headline: 'Stock normal',
      message: 'No se espera desabastecimiento de este ítem en los próximos 7 días.',
    },
    critico: {
      color: 'red',
      eyebrow: 'Stock',
      headline: 'Stock crítico',
      message: 'Este ítem podría agotarse en los próximos 7 días. Considera reponerlo pronto.',
    },
  },
  stock_critico_14d: {
    normal: {
      color: 'green',
      eyebrow: 'Stock',
      headline: 'Stock normal',
      message: 'No se espera desabastecimiento de este ítem en los próximos 14 días.',
    },
    critico: {
      color: 'red',
      eyebrow: 'Stock',
      headline: 'Stock crítico',
      message: 'Este ítem podría agotarse en los próximos 14 días. Considera reponerlo pronto.',
    },
  },
};

const REGRESSION_COPY = {
  duracion_estadia: (value) => ({
    color: 'teal',
    eyebrow: 'Estimado',
    headline: `${value.toFixed(0)} días de hospedaje`,
    message: `Se estima que este paciente necesitará aproximadamente ${value.toFixed(0)} días de hospedaje.`,
  }),
  demanda_consumo: (value) => ({
    color: 'teal',
    eyebrow: 'Estimado',
    headline: `${value.toFixed(1)} unidades`,
    message: `Se estima un consumo de aproximadamente ${value.toFixed(1)} unidades para este ítem.`,
  }),
  proyeccion_donaciones: (value) => ({
    color: 'teal',
    eyebrow: 'Proyección',
    headline: `S/ ${value.toFixed(2)}`,
    message: `Se proyecta un valor de donaciones de aproximadamente S/ ${value.toFixed(2)} para esta categoría el próximo mes.`,
  }),
};

export default function ResultCard({ model, result }) {
  if (model.task === 'classification') {
    const label = result.prediccion;
    const copy = CLASS_COPY[model.name]?.[label] || {
      color: 'teal',
      eyebrow: 'Resultado',
      headline: label,
      message: 'Resultado de la evaluación.',
    };
    const probs = result.probabilidades || {};
    const confidence = probs[label] != null ? Math.round(probs[label] * 100) : null;

    return (
      <div className={`result-card result-${copy.color}`}>
        <p className="result-eyebrow">{copy.eyebrow}</p>
        <h3>{copy.headline}</h3>
        <p>{copy.message}</p>
        {confidence != null && (
          <p className="result-confidence">Confianza estimada: {confidence}%</p>
        )}
      </div>
    );
  }

  const value = Number(result.prediccion);
  const build = REGRESSION_COPY[model.name];
  const copy = build
    ? build(value)
    : { color: 'teal', eyebrow: 'Resultado', headline: value.toFixed(2), message: model.description };

  return (
    <div className={`result-card result-${copy.color}`}>
      <p className="result-eyebrow">{copy.eyebrow}</p>
      <h3>{copy.headline}</h3>
      <p>{copy.message}</p>
    </div>
  );
}
