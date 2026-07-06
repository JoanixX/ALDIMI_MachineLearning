//! Lectura de `model_registry.json`, el contrato generado por
//! `python -m aldimi_models.train` con el esquema de entrada y las metricas
//! reales de cada modelo.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ApiError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub task: Task,
    pub description: String,
    pub selection_metric: String,
    pub selected_algorithm: String,
    pub metrics: serde_json::Map<String, serde_json::Value>,
    pub numeric_features: Vec<String>,
    pub categorical_features: Vec<String>,
    pub class_labels: Option<Vec<String>>,
    pub onnx_file: Option<String>,
    pub trained_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Classification,
    Regression,
}

pub fn load_registry(artifacts_dir: &Path) -> Result<Vec<ModelEntry>, ApiError> {
    let path = artifacts_dir.join("model_registry.json");
    let raw = fs::read_to_string(&path).map_err(|e| {
        ApiError::Startup(format!("no se pudo leer {}: {e}", path.display()))
    })?;
    let entries: Vec<ModelEntry> = serde_json::from_str(&raw).map_err(|e| {
        ApiError::Startup(format!("model_registry.json invalido: {e}"))
    })?;
    if entries.is_empty() {
        return Err(ApiError::Startup(
            "model_registry.json no contiene modelos; ejecuta el entrenamiento".into(),
        ));
    }
    Ok(entries)
}
