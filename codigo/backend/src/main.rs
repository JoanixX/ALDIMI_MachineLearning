//! ALDIMI Predict - API de inferencia.
//!
//! Servicio Rust (Axum) que sirve los modelos ONNX entrenados por el paquete
//! Python `aldimi_models`. Variables de entorno:
//!
//! - `ALDIMI_ARTIFACTS_DIR`: carpeta con los .onnx y model_registry.json
//!   (default: ../ia_models/artifacts, pensado para desarrollo local)
//! - `PORT`: puerto de escucha (default 8000; Render lo inyecta)

mod error;
mod inference;
mod registry;
mod routes;
mod schema;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tower_http::cors::CorsLayer;

use crate::error::ApiError;
use crate::inference::LoadedModel;
use crate::routes::AppState;

fn artifacts_dir() -> PathBuf {
    std::env::var("ALDIMI_ARTIFACTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../ia_models/artifacts"))
}

fn build_state() -> Result<AppState, ApiError> {
    let dir = artifacts_dir();
    let entries = registry::load_registry(&dir)?;
    let mut models = HashMap::new();
    for entry in entries {
        let name = entry.name.clone();
        match LoadedModel::load(&dir, entry) {
            Ok(model) => {
                println!("modelo cargado: {name}");
                models.insert(name, model);
            }
            Err(err) => {
                // Un modelo sin ONNX no tumba el servicio: se sirve el resto.
                eprintln!("ADVERTENCIA: {name} no disponible: {err}");
            }
        }
    }
    if models.is_empty() {
        return Err(ApiError::Startup(
            "ningun modelo ONNX pudo cargarse; revisa ALDIMI_ARTIFACTS_DIR".into(),
        ));
    }
    Ok(AppState { models })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(build_state()?);
    let app = routes::router(state).layer(CorsLayer::permissive());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("ALDIMI backend escuchando en http://0.0.0.0:{port}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        eprintln!("no se pudo instalar el manejador de ctrl-c: {err}");
    }
}
