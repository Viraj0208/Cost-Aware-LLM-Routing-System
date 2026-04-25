# Cost-Aware LLM Routing System

A Software Engineering in AI course project that routes prompts to either a
low-cost "small" model or a higher-quality "large" model based on estimated
prompt complexity. The working demo runs in simulation mode with mock model
backends for local development, and in production mode with a Triton/TensorRT
router when the TensorRT engine file is available.

## What Works

- FastAPI backend with completion, routing, health, metrics, and analytics APIs
- Streamlit dashboard for trying prompts and viewing cost savings
- Rule-based prompt complexity router
- Triton/TensorRT router path for production-mode API routing
- Mock small/large model backends with different latency and response quality
- Request cost estimation and in-memory analytics
- Prometheus metrics endpoint
- Unit and integration tests

## Architecture

```text
Client / Streamlit
    -> FastAPI API
    -> Preprocessor
    -> Complexity Router (Python rules locally, Triton/TensorRT in production)
    -> Small Mock Model or Large Mock Model
    -> Cost Tracker
    -> Analytics + Prometheus Metrics
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn src.api.app:app --reload
```

The API runs at:

```text
http://localhost:8000
```

Useful endpoints:

```text
GET  /health
GET  /metrics
GET  /v1/models
GET  /v1/analytics/costs
POST /v1/route
POST /v1/completions
```

Example request:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Explain what Python is\", \"max_tokens\": 128}"
```

### 3. Run the Streamlit demo

In a second terminal:

```bash
streamlit run demo/streamlit_app.py
```

The dashboard runs at:

```text
http://localhost:8501
```

## Running Tests

```bash
python -m pytest tests -q
```

## Configuration

The default local mode is `simulation`, defined in:

```text
config/default.yaml
```

For local development without GPU, keep:

```text
APP_MODE=simulation
```

For the Triton/TensorRT router demo, use:

```text
APP_MODE=production
```

Production mode expects Triton to serve the router TensorRT engine at
`TRITON_URL`.

## Triton + TensorRT Router

The course-required Triton/TensorRT path uses TensorRT for the prompt router,
not for full LLM generation. Export the ONNX router, convert it to
`model.plan` on an NVIDIA GPU machine, and place it in:

```text
src/triton/model_repository/router/1/model.plan
```

Detailed steps are in:

```text
docs/TRITON_TENSORRT_ROUTER.md
```

## Docker

Docker is configured for the Triton-backed router demo. It requires the
TensorRT engine file to exist at
`src/triton/model_repository/router/1/model.plan`.

```bash
cd docker
docker-compose up --build api demo prometheus
```

Then open:

```text
API:       http://localhost:8000
Streamlit: http://localhost:8501
Prometheus: http://localhost:9090
```

The API runs in `production` mode and calls Triton for router inference.

## Project Structure

```text
config/        YAML configuration
demo/          Streamlit dashboard and sample prompts
docker/        Docker setup for Triton router demo
scripts/       Utility scripts
src/api/       FastAPI application and routes
src/config/    Settings and model registry
src/cost/      Cost calculation and tracking
src/inference/ Inference pipeline and preprocessing
src/models/    Backend abstraction and mock backend
src/router/    Feature extraction and routing logic
src/triton/    Triton model repository for TensorRT router
tests/         Unit, integration, and benchmark tests
training/      Experimental router training scripts
```

## Notes for Evaluation

This project demonstrates the software engineering design of a cost-aware LLM
router. The main implemented idea is that simple prompts are routed to a cheaper
small model, while complex prompts are routed to a higher-quality large model.

The current version focuses on a working Triton/TensorRT router demo. Full LLM
serving through Triton is intentionally out of scope for this course project.
