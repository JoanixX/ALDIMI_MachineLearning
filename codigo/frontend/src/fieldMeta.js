// Traduccion de los nombres crudos de columnas (los que exponen /models) a
// etiquetas y ayudas en espanol llano, con unidades y rango esperado cuando
// aplica. Si aparece un campo nuevo que no esta aqui, se muestra su nombre
// crudo como respaldo (el formulario nunca deja de funcionar).
export const FIELD_LABELS = {
  edad: { label: 'Edad del paciente', helper: 'años — ej. 8' },
  dias_hospedaje: { label: 'Días de hospedaje hasta ahora', helper: 'días — ej. 5' },
  num_controles_mes: { label: 'Controles médicos en el último mes', helper: 'ej. 2' },
  num_quimios_mes: { label: 'Sesiones de quimioterapia en el último mes', helper: 'ej. 1' },
  hemoglobina_g_dl: { label: 'Hemoglobina', helper: 'g/dL — ej. 11.5 (normal: 12-16)' },
  neutrofilos: { label: 'Neutrófilos', helper: 'células/mm³ — ej. 1800' },
  plaquetas: { label: 'Plaquetas', helper: 'miles/mm³ — ej. 250' },
  temperatura_c: { label: 'Temperatura corporal', helper: '°C — ej. 36.5' },
  peso_kg: { label: 'Peso', helper: 'kg — ej. 24' },
  imc: { label: 'Índice de masa corporal (IMC)', helper: 'ej. 16.5' },
  distancia_origen_km: { label: 'Distancia desde su lugar de origen', helper: 'km — ej. 120' },
  ingreso_familiar_mensual: { label: 'Ingreso familiar mensual', helper: 'S/ — ej. 1200' },
  acompanante_presente: { label: '¿Tiene acompañante presente?', helper: '1 = sí, 0 = no' },
  requiere_apoyo_psicosocial: { label: '¿Requiere apoyo psicosocial?', helper: '1 = sí, 0 = no' },
  sexo: { label: 'Sexo' },
  region_origen: { label: 'Región de origen' },
  diagnostico_general: { label: 'Diagnóstico general' },
  estado_tratamiento: { label: 'Estado del tratamiento' },
  seguro_salud: { label: 'Seguro de salud' },
  alfabetizacion_digital: { label: 'Nivel de alfabetización digital' },
  ocupacion_familias: { label: 'Familias hospedadas actualmente', helper: 'ej. 18' },
  stock_inicio: { label: 'Stock al inicio del periodo', helper: 'unidades — ej. 100' },
  ingreso_stock: { label: 'Ingreso de stock en el periodo', helper: 'unidades — ej. 30' },
  consumo_real: { label: 'Consumo real en el periodo', helper: 'unidades — ej. 40' },
  categoria: { label: 'Categoría del ítem' },
  unidad_medida: { label: 'Unidad de medida' },
  item_id: { label: 'Ítem' },
  consumo_lag_1: { label: 'Consumo de ayer', helper: 'unidades' },
  consumo_lag_7: { label: 'Consumo hace 7 días', helper: 'unidades' },
  consumo_lag_14: { label: 'Consumo hace 14 días', helper: 'unidades' },
  consumo_media_7: { label: 'Consumo promedio (últimos 7 días)', helper: 'unidades' },
  dia_semana: { label: 'Día de la semana', helper: '1 = lunes … 7 = domingo' },
  mes: { label: 'Mes', helper: '1 a 12' },
  ocupacion_lag_1: { label: 'Familias hospedadas ayer', helper: 'ej. 18' },
  valor_lag_1: { label: 'Valor de donaciones hace 1 mes', helper: 'S/' },
  valor_lag_2: { label: 'Valor de donaciones hace 2 meses', helper: 'S/' },
  valor_lag_3: { label: 'Valor de donaciones hace 3 meses', helper: 'S/' },
  donaciones_lag_1: { label: 'Número de donaciones el mes pasado', helper: 'ej. 6' },
};

export function fieldMeta(name) {
  return FIELD_LABELS[name] || { label: name, helper: '' };
}

// Traduccion de valores crudos de categorias que no vienen ya en espanol
// natural (la mayoria de region_origen/diagnostico_general/etc. ya lo estan).
const CATEGORY_VALUE_LABELS = {
  sexo: { F: 'Femenino', M: 'Masculino' },
  categoria: {
    alimentos: 'Alimentos',
    higiene: 'Higiene',
    insumos_medicos: 'Insumos médicos',
    medicamentos_soporte: 'Medicamentos de soporte',
  },
};

export function categoryValueLabel(field, value) {
  return CATEGORY_VALUE_LABELS[field]?.[value] || value;
}
