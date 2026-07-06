//! Errores de la API con conversion a respuestas HTTP.
//!
//! Regla del proyecto: los handlers devuelven `Result<T, ApiError>`;
//! `unwrap()`/`expect()` solo se permiten en tests.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("modelo desconocido: {0}")]
    UnknownModel(String),

    #[error("entrada invalida: {0}")]
    InvalidInput(String),

    #[error("error de inferencia: {0}")]
    Inference(String),

    #[error("error de configuracion: {0}")]
    Startup(String),
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            ApiError::UnknownModel(_) => StatusCode::NOT_FOUND,
            ApiError::InvalidInput(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::Inference(_) | ApiError::Startup(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({ "error": self.to_string() }));
        (self.status(), body).into_response()
    }
}

impl From<ort::Error> for ApiError {
    fn from(err: ort::Error) -> Self {
        ApiError::Inference(err.to_string())
    }
}
