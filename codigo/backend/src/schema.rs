//! Validacion del payload de prediccion contra el esquema de features del
//! modelo (numeric_features / categorical_features del registro).

use serde_json::{Map, Value};

use crate::error::ApiError;
use crate::registry::ModelEntry;

/// Entrada ya validada y tipada para construir los tensores ONNX.
#[derive(Debug)]
pub struct ValidatedInput {
    pub numeric: Vec<(String, Vec<f32>)>,
    pub categorical: Vec<(String, Vec<String>)>,
    pub n_rows: usize,
}

/// Acepta un objeto (un caso) o una lista de objetos (batch).
pub fn validate_payload(entry: &ModelEntry, payload: &Value) -> Result<ValidatedInput, ApiError> {
    let records: Vec<&Map<String, Value>> = match payload {
        Value::Object(map) => vec![map],
        Value::Array(items) => {
            if items.is_empty() {
                return Err(ApiError::InvalidInput("la lista de casos esta vacia".into()));
            }
            items
                .iter()
                .map(|item| {
                    item.as_object().ok_or_else(|| {
                        ApiError::InvalidInput("cada caso debe ser un objeto JSON".into())
                    })
                })
                .collect::<Result<_, _>>()?
        }
        _ => {
            return Err(ApiError::InvalidInput(
                "el cuerpo debe ser un objeto JSON o una lista de objetos".into(),
            ))
        }
    };

    let mut numeric = Vec::with_capacity(entry.numeric_features.len());
    for feature in &entry.numeric_features {
        let mut column = Vec::with_capacity(records.len());
        for record in &records {
            column.push(extract_number(record, feature)?);
        }
        numeric.push((feature.clone(), column));
    }

    let mut categorical = Vec::with_capacity(entry.categorical_features.len());
    for feature in &entry.categorical_features {
        let mut column = Vec::with_capacity(records.len());
        for record in &records {
            column.push(extract_string(record, feature)?);
        }
        categorical.push((feature.clone(), column));
    }

    Ok(ValidatedInput {
        numeric,
        categorical,
        n_rows: records.len(),
    })
}

fn extract_number(record: &Map<String, Value>, feature: &str) -> Result<f32, ApiError> {
    match record.get(feature) {
        Some(Value::Number(n)) => n
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| ApiError::InvalidInput(format!("'{feature}' no es un numero valido"))),
        Some(Value::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Some(other) => Err(ApiError::InvalidInput(format!(
            "'{feature}' debe ser numerico, se recibio: {other}"
        ))),
        None => Err(ApiError::InvalidInput(format!("falta la feature numerica '{feature}'"))),
    }
}

fn extract_string(record: &Map<String, Value>, feature: &str) -> Result<String, ApiError> {
    match record.get(feature) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(ApiError::InvalidInput(format!(
            "'{feature}' debe ser texto, se recibio: {other}"
        ))),
        None => Err(ApiError::InvalidInput(format!(
            "falta la feature categorica '{feature}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Task;
    use serde_json::json;

    fn entry() -> ModelEntry {
        ModelEntry {
            name: "test".into(),
            task: Task::Classification,
            description: String::new(),
            selection_metric: "f1_macro".into(),
            selected_algorithm: "xgboost".into(),
            metrics: Map::new(),
            numeric_features: vec!["edad".into()],
            categorical_features: vec!["sexo".into()],
            class_labels: Some(vec!["Bajo".into(), "Alto".into()]),
            onnx_file: Some("test.onnx".into()),
            trained_at: String::new(),
        }
    }

    #[test]
    fn acepta_objeto_valido() {
        let input = validate_payload(&entry(), &json!({"edad": 10, "sexo": "F"})).unwrap();
        assert_eq!(input.n_rows, 1);
        assert_eq!(input.numeric[0].1, vec![10.0]);
        assert_eq!(input.categorical[0].1, vec!["F".to_string()]);
    }

    #[test]
    fn acepta_lista_de_objetos() {
        let payload = json!([{"edad": 1, "sexo": "F"}, {"edad": 2, "sexo": "M"}]);
        let input = validate_payload(&entry(), &payload).unwrap();
        assert_eq!(input.n_rows, 2);
        assert_eq!(input.numeric[0].1, vec![1.0, 2.0]);
    }

    #[test]
    fn rechaza_feature_faltante() {
        let err = validate_payload(&entry(), &json!({"edad": 10})).unwrap_err();
        assert!(err.to_string().contains("sexo"));
    }

    #[test]
    fn rechaza_tipo_incorrecto() {
        let err = validate_payload(&entry(), &json!({"edad": "diez", "sexo": "F"})).unwrap_err();
        assert!(err.to_string().contains("edad"));
    }

    #[test]
    fn rechaza_lista_vacia() {
        assert!(validate_payload(&entry(), &json!([])).is_err());
    }
}
