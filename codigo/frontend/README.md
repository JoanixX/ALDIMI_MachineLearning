# ALDIMI Predict - Dashboard Frontend (React)

Panel interactivo web desarrollado en **React** y empaquetado con **Vite**. Permite al personal de la ONG visualizar el rendimiento real de los modelos de Machine Learning y realizar predicciones interactivas en tiempo real.

---

## 🎨 Características principales

1. **Diseño dinámico**: Los formularios de entrada se generan dinámicamente según el esquema de variables numéricas y categóricas que publica el backend en el endpoint `/models`. De esta forma, si se añaden variables o se actualiza un modelo, la interfaz se adapta automáticamente.
2. **Visualización de métricas**: Gráficos horizontales interactivos de las métricas F1-macro y MAE del conjunto de test.
3. **Predicción en tiempo real**: Formulario dinámico para ingresar datos clínicos o de inventario y obtener de inmediato la inferencia servida por el backend Rust.

---

## 🛠️ Ejecución Local

1. Instala las dependencias de Node:
   ```bash
   npm install
   ```
2. Inicia el servidor de desarrollo de Vite:
   ```bash
   npm run dev
   ```
3. El frontend se abrirá en `http://localhost:5173`. Por defecto, apuntará al backend de desarrollo local (`http://localhost:8000`).

---

## ☁️ Guía de Despliegue en Vercel (Hobby Plan)

Vercel permite alojar aplicaciones de React de forma gratuita:

1. Crea una cuenta gratuita en [Vercel](https://vercel.com/).
2. Instala la CLI de Vercel (opcional) o conecta tu cuenta de GitHub.
3. Crea un **Nuevo Proyecto** y selecciona el repositorio de ALDIMI.
4. En **Framework Preset**, selecciona `Vite`.
5. En la sección **Environment Variables**, añade la siguiente variable:
   - `VITE_API_URL`: La URL pública de tu API de Rust desplegada en Render (por ejemplo, `https://aldimi-backend.onrender.com`).
6. Presiona **Deploy**. 

Vercel compilará la aplicación estática y te entregará una URL pública gratuita (por ejemplo, `https://aldimi-frontend.vercel.app`).
