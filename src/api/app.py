"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.middleware import TimingMiddleware
from src.api.routes import inference, health, metrics, analytics
from src.config.settings import get_settings, PROJECT_ROOT
from src.monitoring.logger import setup_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    setup_logging(
        level=settings.monitoring.log_level,
        format_type=settings.monitoring.log_format,
    )

    app = FastAPI(
        title="Cost-Aware LLM Routing System",
        description=(
            "Intelligent LLM query routing system that minimizes inference cost "
            "while maintaining quality thresholds. Uses TensorRT-optimized models "
            "served via Triton Inference Server."
        ),
        version="0.1.0",
    )

    # Allow all origins for demo — frontend served from same host
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimingMiddleware)

    # API routes (registered before static files so they take precedence)
    app.include_router(inference.router, tags=["Inference"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(metrics.router, tags=["Monitoring"])
    app.include_router(analytics.router, tags=["Analytics"])

    # Serve the frontend dashboard at root
    @app.get("/")
    async def serve_frontend() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "index.html")

    return app


# For uvicorn: uvicorn src.api.app:app --reload
app = create_app()
