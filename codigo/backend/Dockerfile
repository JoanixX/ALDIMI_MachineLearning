# Imagen del backend de inferencia para Render (Web Service via Docker).
#
# IMPORTANTE: construir con el contexto en la RAIZ del repo, porque la imagen
# copia tambien los artefactos ONNX de codigo/ia_models/artifacts:
#
#   docker build -f codigo/backend/Dockerfile -t aldimi-backend .
#
# En Render: "Dockerfile Path" = codigo/backend/Dockerfile y
# "Docker Build Context Directory" = . (raiz del repo).

# Stage 1: Build the Rust application
FROM rust:1-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files from the backend directory
COPY codigo/backend/Cargo.toml codigo/backend/Cargo.lock ./

# Create a dummy src/main.rs to build dependencies and cache them
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src target/release/deps/aldimi_backend*

# Copy the real source code
COPY codigo/backend/src ./src

# Build the real application
RUN cargo build --release

# Stage 2: Create a minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary and the onnxruntime shared library
COPY --from=builder /app/target/release/aldimi-backend /app/aldimi-backend
# Copy ONNX Runtime shared library downloaded by the builder
COPY --from=builder /app/target/release/*.so* /usr/local/lib/

# Configure dynamic linker search path
ENV LD_LIBRARY_PATH=/usr/local/lib

# Setup environment variables for production
ENV ALDIMI_ARTIFACTS_DIR=/app/artifacts
ENV PORT=8000
ENV RUST_LOG=info

# Copy the trained models and registry into the container
COPY codigo/ia_models/artifacts /app/artifacts

# Expose port
EXPOSE 8000

# Run the backend binary
CMD ["/app/aldimi-backend"]
