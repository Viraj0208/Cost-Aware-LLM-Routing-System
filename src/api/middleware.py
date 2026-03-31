"""FastAPI middleware for request timing and error handling."""

from __future__ import annotations

import time
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger("llm_router")


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Response-Time header to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response
