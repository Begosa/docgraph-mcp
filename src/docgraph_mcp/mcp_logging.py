from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def logged_tool(fn: Callable[..., Any], get_backend: Callable[[], Any]) -> Callable[..., Any]:
    sig = inspect.signature(fn)
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        backend = get_backend()
        started = time.perf_counter()
        try:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
        except Exception:
            params = {"args_count": len(args), "kwargs_keys": sorted(kwargs)}
        if hasattr(backend, "log_tool_start"):
            backend.log_tool_start(fn.__name__, params)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            if hasattr(backend, "log_tool_error"):
                backend.log_tool_error(fn.__name__, exc, (time.perf_counter() - started) * 1000)
            raise
        if hasattr(backend, "log_tool_done"):
            backend.log_tool_done(fn.__name__, result, (time.perf_counter() - started) * 1000)
        return result
    return wrapper
