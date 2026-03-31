"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import TimingMiddleware
from src.api.routes import inference, health, metrics, analytics
from src.config.settings import get_settings
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

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimingMiddleware)

    # Routes
    app.include_router(inference.router, tags=["Inference"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(metrics.router, tags=["Monitoring"])
    app.include_router(analytics.router, tags=["Analytics"])

    @app.get("/")
    async def root():
        return {
            "name": "Cost-Aware LLM Routing System",
            "version": "0.1.0",
            "mode": settings.mode,
            "docs": "/docs",
        }

    return app


# For uvicorn: uvicorn src.api.app:app --reload
app = create_app()
