# Cost-Aware LLM Routing System

**Using TensorRT-Optimized Models on Triton Inference Server**

An intelligent system that routes LLM queries to the most cost-effective model based on prompt complexity, achieving **50-85% cost savings** while maintaining quality thresholds.

## Architecture

```
Client → FastAPI API → Router (DistilBERT, <10ms) → Model Selection
                                                      ├── Small Model (Phi-2, $0.002/1K tokens)
                                                      └── Large Model (Llama-2-70B, $0.06/1K tokens)
                     → Cost Tracker → SQLite
                     → Prometheus Metrics
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Router | DistilBERT + Rule-based features | Classify prompt complexity in <10ms |
| Optimization | TensorRT (FP16/INT8) | 2-4x inference speedup |
| Serving | Triton Inference Server | Production model serving with dynamic batching |
| API | FastAPI | REST endpoints with cost/routing metadata |
| Monitoring | Prometheus | Latency, cost, and routing metrics |
| Demo | Streamlit | Interactive dashboard |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Quick CLI demo
python -m scripts.run_demo

# Interactive Streamlit dashboard
streamlit run demo/streamlit_app.py

# Start the API server
uvicorn src.api.app:app --reload
```

### 3. API Usage

```bash
# Generate with routing
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?", "max_tokens": 256}'

# Route only (no generation)
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Implement a binary search tree"}'

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Cost analytics
curl http://localhost:8000/v1/analytics/costs
```

## Project Structure

```
├── config/                  # YAML configuration (default, dev, production)
├── src/
│   ├── config/             # Settings loader, model registry
│   ├── router/             # Complexity classifier, feature extractor, routing engine
│   ├── models/             # Backend abstraction (mock, HuggingFace, Triton)
│   ├── inference/          # Pipeline orchestration
│   ├── optimization/       # ONNX export, TensorRT conversion, benchmarks
│   ├── triton/             # Model repository, ensemble configs, BLS scripts
│   ├── api/                # FastAPI endpoints
│   ├── cost/               # Calculator, tracker, SQLite storage
│   └── monitoring/         # Prometheus metrics, logging
├── training/               # Router training pipeline (dataset, train, evaluate, export)
├── demo/                   # Streamlit dashboard + sample prompts
├── docker/                 # Dockerfiles for API, Triton, docker-compose
├── tests/                  # Unit, integration, benchmark tests
└── scripts/                # Utility scripts
```

## Training the Router

```bash
# 1. Generate dataset
python -m training.prepare_dataset

# 2. Train DistilBERT classifier (requires torch, transformers)
pip install -r requirements-ml.txt
python -m training.train_router --data training/data/router_dataset.csv --epochs 3

# 3. Export to ONNX
python -m training.export_model

# 4. Evaluate
python -m training.evaluate_router
```

## Production Deployment (Docker)

```bash
cd docker
docker-compose up --build
```

This starts:
- **API server** on port 8000
- **Triton Inference Server** on ports 8001 (gRPC) / 8002 (metrics)
- **Prometheus** on port 9090
- **Streamlit demo** on port 8501

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Benchmark tests
python -m pytest tests/benchmarks/ -v -s
```

## Configuration

The system uses YAML configuration with environment overlay:

- `config/default.yaml` — Base configuration
- `config/development.yaml` — Simulation mode (mock backends)
- `config/production.yaml` — Triton mode

Set `APP_MODE=simulation|local|production` to switch modes.

## Key Design Decisions

1. **Simulation-first**: Mock backends produce quality-differentiated responses for demos without GPU
2. **Hybrid router**: ML (DistilBERT) + rule-based features for accuracy and interpretability
3. **ONNX Runtime on Windows**: TensorRT only in Docker/production; ONNX provides cross-platform optimization
4. **YAML config layering**: Same code works across simulation/local/production modes

## References

1. Chen et al. (2023). FrugalGPT: How to Use LLMs While Reducing Cost. TMLR 2024.
2. Ong et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. ICLR 2025.
3. Ding et al. (2024). Hybrid-LLM: Cost-Efficient Quality-Aware Query Routing. ICLR 2024.
4. NVIDIA. TensorRT Documentation. developer.nvidia.com/tensorrt
5. NVIDIA. Triton Inference Server Documentation. docs.nvidia.com/triton-inference-server
