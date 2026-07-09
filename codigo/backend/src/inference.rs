//! Carga de sesiones ONNX Runtime y ejecucion de predicciones.
//!
//! Los modelos se entrenan en Python (scikit-learn/XGBoost) y se exportan a
//! ONNX; aqui solo se hace inferencia. Cada columna del modelo es un input
//! ONNX independiente que se alimenta por nombre.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::{Tensor, Value as OrtValue};
use serde::Serialize;

use crate::error::ApiError;
use crate::registry::{ModelEntry, Task};
use crate::schema::ValidatedInput;

pub struct LoadedModel {
    pub entry: ModelEntry,
    session: Mutex<Session>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Prediction {
    Classification {
        prediccion: String,
        clase_indice: i64,
        probabilidades: HashMap<String, f32>,
    },
    Regression {
        prediccion: f32,
    },
}

impl LoadedModel {
    pub fn load(artifacts_dir: &Path, entry: ModelEntry) -> Result<Self, ApiError> {
        let onnx_file = entry.onnx_file.as_ref().ok_or_else(|| {
            ApiError::Startup(format!("el modelo {} no tiene ONNX exportado", entry.name))
        })?;
        let path = artifacts_dir.join(onnx_file);
        if !path.exists() {
            return Err(ApiError::Startup(format!(
                "no existe el artefacto ONNX: {}",
                path.display()
            )));
        }
        let session = Session::builder()?.commit_from_file(&path)?;
        Ok(Self {
            entry,
            session: Mutex::new(session),
        })
    }

    pub fn predict(&self, input: &ValidatedInput) -> Result<Vec<Prediction>, ApiError> {
        let mut feeds: Vec<(String, OrtValue)> = Vec::new();
        for (name, column) in &input.numeric {
            let tensor = Tensor::from_array(([input.n_rows, 1], column.clone()))?;
            feeds.push((name.clone(), tensor.into_dyn()));
        }
        for (name, column) in &input.categorical {
            let tensor = Tensor::from_string_array(([input.n_rows, 1], column.as_slice()))?;
            feeds.push((name.clone(), tensor.into_dyn()));
        }

        let mut session = self.session.lock().map_err(|_| {
            ApiError::Inference("sesion ONNX envenenada por un panic previo".into())
        })?;
        let outputs = session.run(feeds)?;

        match self.entry.task {
            Task::Classification => self.classification_results(&outputs, input.n_rows),
            Task::Regression => regression_results(&outputs, input.n_rows),
        }
    }

    fn classification_results(
        &self,
        outputs: &ort::session::SessionOutputs,
        n_rows: usize,
    ) -> Result<Vec<Prediction>, ApiError> {
        let class_labels = self.entry.class_labels.as_ref().ok_or_else(|| {
            ApiError::Inference(format!("el modelo {} no declara class_labels", self.entry.name))
        })?;
        let (_, labels) = outputs[0].try_extract_tensor::<i64>()?;
        let (_, probabilities) = outputs[1].try_extract_tensor::<f32>()?;
        let n_classes = class_labels.len();

        let mut results = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let idx = labels[row];
            let label = class_labels
                .get(idx as usize)
                .ok_or_else(|| ApiError::Inference(format!("indice de clase fuera de rango: {idx}")))?;
            let probs = class_labels
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), probabilities[row * n_classes + i]))
                .collect();
            results.push(Prediction::Classification {
                prediccion: label.clone(),
                clase_indice: idx,
                probabilidades: probs,
            });
        }
        Ok(results)
    }
}

fn regression_results(
    outputs: &ort::session::SessionOutputs,
    n_rows: usize,
) -> Result<Vec<Prediction>, ApiError> {
    let (_, values) = outputs[0].try_extract_tensor::<f32>()?;
    if values.len() < n_rows {
        return Err(ApiError::Inference(format!(
            "la salida del modelo tiene {} valores para {} filas",
            values.len(),
            n_rows
        )));
    }
    Ok(values[..n_rows]
        .iter()
        .map(|v| Prediction::Regression { prediccion: *v })
        .collect())
}
