// Nombres en lenguaje natural para el personal del albergue, en vez del
// nombre tecnico interno del modelo. Si aparece un modelo nuevo que no
// conocemos, usamos su "description" (ya redactada en espanol llano).
export const DISPLAY_NAMES = {
  prioridad_atencion: 'Prioridad de atención del paciente',
  duracion_estadia: 'Días de hospedaje estimados',
  stock_critico_7d: 'Riesgo de quiebre de stock (7 días)',
  stock_critico_14d: 'Riesgo de quiebre de stock (14 días)',
  demanda_consumo: 'Consumo esperado por ítem',
  proyeccion_donaciones: 'Proyección de donaciones por categoría',
};

export function getModelLabel(model) {
  return DISPLAY_NAMES[model.name] || model.description;
}

// La pantalla principal se organiza por tarea del personal del albergue,
// no por modelo tecnico. Cada tarea agrupa uno o mas modelos relacionados.
export const TASKS = [
  {
    key: 'paciente',
    icon: 'patient',
    accent: 'teal',
    title: 'Evaluar un paciente nuevo',
    description:
      'Calcula la prioridad de atención y los días de hospedaje estimados para un paciente que ingresa al albergue.',
    models: ['prioridad_atencion', 'duracion_estadia'],
  },
  {
    key: 'inventario',
    icon: 'inventory',
    accent: 'gold',
    title: 'Consultar el estado del inventario',
    description:
      'Revisa si un ítem del almacén está en riesgo de agotarse y estima su consumo esperado.',
    models: ['stock_critico_7d', 'stock_critico_14d', 'demanda_consumo'],
  },
  {
    key: 'donaciones',
    icon: 'donations',
    accent: 'coral',
    title: 'Ver proyección de donaciones',
    description:
      'Estima el valor de donaciones esperado para el próximo mes, por categoría.',
    models: ['proyeccion_donaciones'],
  },
];
