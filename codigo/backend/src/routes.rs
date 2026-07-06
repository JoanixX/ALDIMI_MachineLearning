//! Endpoints HTTP del servicio de inferencia.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::inference::LoadedModel;
use crate::schema::validate_payload;

pub struct AppState {
    pub models: HashMap<String, LoadedModel>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/predict/{model}", post(predict))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Expone el registro completo (esquema de entrada + metricas reales de test)
/// para que el frontend construya formularios y tablas dinamicamente.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let entries: Vec<_> = state.models.values().map(|m| &m.entry).collect();
    Json(json!({ "models": entries }))
}

async fn predict(
    State(state): State<Arc<AppState>>,
    Path(model_name): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let model = state
        .models
        .get(&model_name)
        .ok_or_else(|| ApiError::UnknownModel(model_name.clone()))?;
    let input = validate_payload(&model.entry, &payload)?;
    let predictions = model.predict(&input)?;
    Ok(Json(json!({
        "model": model_name,
        "predictions": predictions,
    })))
}
