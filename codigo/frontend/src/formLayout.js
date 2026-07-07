// Agrupacion de campos por seccion, en espanol llano, para cada modelo
// conocido. Cualquier campo del esquema real (numeric_features +
// categorical_features) que no aparezca en estos grupos se agrega
// automaticamente a una seccion de respaldo "Otros datos", para que el
// formulario no pierda campos si el backend cambia el esquema.
const GROUPS_BY_MODEL = {
  prioridad_atencion: [
    {
      title: 'Datos del paciente',
      fields: [
        'edad',
        'sexo',
        'region_origen',
        'diagnostico_general',
        'estado_tratamiento',
        'seguro_salud',
        'alfabetizacion_digital',
      ],
    },
    {
      title: 'Signos vitales y clínicos',
      fields: ['temperatura_c', 'peso_kg', 'imc', 'hemoglobina_g_dl', 'neutrofilos', 'plaquetas'],
    },
    {
      title: 'Tratamiento y seguimiento',
      fields: [
        'dias_hospedaje',
        'num_controles_mes',
        'num_quimios_mes',
        'acompanante_presente',
        'requiere_apoyo_psicosocial',
      ],
    },
    {
      title: 'Situación familiar',
      fields: ['distancia_origen_km', 'ingreso_familiar_mensual'],
    },
  ],
  duracion_estadia: [
    {
      title: 'Datos del paciente',
      fields: [
        'edad',
        'sexo',
        'region_origen',
        'diagnostico_general',
        'estado_tratamiento',
        'seguro_salud',
        'alfabetizacion_digital',
      ],
    },
    {
      title: 'Signos vitales y clínicos',
      fields: ['temperatura_c', 'peso_kg', 'imc', 'hemoglobina_g_dl', 'neutrofilos', 'plaquetas'],
    },
    {
      title: 'Tratamiento y seguimiento',
      fields: ['num_controles_mes', 'num_quimios_mes', 'acompanante_presente', 'requiere_apoyo_psicosocial'],
    },
    {
      title: 'Situación familiar',
      fields: ['distancia_origen_km', 'ingreso_familiar_mensual'],
    },
  ],
  stock_critico_7d: [
    { title: 'Datos del ítem', fields: ['categoria', 'unidad_medida', 'item_id'] },
    {
      title: 'Movimiento de stock',
      fields: ['ocupacion_familias', 'stock_inicio', 'ingreso_stock', 'consumo_real'],
    },
  ],
  stock_critico_14d: [
    { title: 'Datos del ítem', fields: ['categoria', 'unidad_medida', 'item_id'] },
    {
      title: 'Movimiento de stock',
      fields: ['ocupacion_familias', 'stock_inicio', 'ingreso_stock', 'consumo_real'],
    },
  ],
  demanda_consumo: [
    { title: 'Datos del ítem', fields: ['categoria', 'item_id'] },
    {
      title: 'Historial de consumo',
      fields: ['consumo_lag_1', 'consumo_lag_7', 'consumo_lag_14', 'consumo_media_7', 'ocupacion_lag_1'],
    },
    { title: 'Fecha', fields: ['dia_semana', 'mes'] },
  ],
  proyeccion_donaciones: [
    { title: 'Categoría', fields: ['categoria'] },
    {
      title: 'Historial de donaciones',
      fields: ['valor_lag_1', 'valor_lag_2', 'valor_lag_3', 'donaciones_lag_1'],
    },
    { title: 'Fecha', fields: ['mes'] },
  ],
};

export function getFieldGroups(model, allFields) {
  const predefined = GROUPS_BY_MODEL[model.name] || [];
  const covered = new Set();

  const groups = predefined
    .map((g) => ({ title: g.title, fields: g.fields.filter((f) => allFields.includes(f)) }))
    .filter((g) => g.fields.length > 0);

  groups.forEach((g) => g.fields.forEach((f) => covered.add(f)));

  const rest = allFields.filter((f) => !covered.has(f));
  if (rest.length > 0) groups.push({ title: 'Otros datos', fields: rest });

  return groups;
}
