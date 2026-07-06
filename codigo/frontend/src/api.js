// Cliente unico del backend Rust. La URL se configura por variable de
// entorno en Vercel (VITE_API_URL); en desarrollo apunta al backend local.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch (err) {
    throw new Error(`No se pudo conectar con el backend (${API_URL}): ${err.message}`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Error HTTP ${response.status}`);
  }
  return body;
}

export function fetchModels() {
  return request('/models');
}

export function predict(modelName, payload) {
  return request(`/predict/${modelName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
