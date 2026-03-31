"""Timing utilities for performance measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generator


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@contextmanager
def timer(name: str = "operation") -> Generator[TimingResult, None, None]:
    """Context manager for timing code blocks.

    Usage:
        with timer("routing") as t:
            result = router.route(prompt)
        print(f"Routing took {t.elapsed_ms:.2f}ms")
    """
    result = TimingResult(name=name, elapsed_ms=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


def timed(name: str | None = None) -> Callable:
    """Decorator for timing function execution.

    Adds `_last_timing` attribute to the function with the latest TimingResult.
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                wrapper._last_timing = TimingResult(name=label, elapsed_ms=elapsed)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                async_wrapper._last_timing = TimingResult(name=label, elapsed_ms=elapsed)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper._last_timing = None
            return async_wrapper

        wrapper._last_timing = None
        return wrapper

    return decorator
