from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .backend import DocGraphBackend
from .mcp_logging import logged_tool

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - allows tests without mcp installed
    FastMCP = None

MCP_MISSING_ERROR = "mcp package is not installed. Install with: python -m pip install -e docgraph-mcp"


@dataclass
class ServerRuntime:
    mcp: Any | None
    get_backend: Callable[[], DocGraphBackend]
    tool: Callable[[Callable[..., Any]], Callable[..., Any]]
    run: Callable[[], None]


def build_server_runtime(server_name: str, description: str) -> ServerRuntime:
    """Create shared MCP server bootstrap helpers for an entrypoint module."""
    backend: DocGraphBackend | None = None
    mcp = FastMCP(server_name) if FastMCP else None

    def get_backend() -> DocGraphBackend:
        nonlocal backend
        if backend is None:
            backend = DocGraphBackend.from_env()
        return backend

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        wrapped = logged_tool(fn, get_backend)
        if mcp is not None:
            return mcp.tool()(wrapped)
        return wrapped

    def run() -> None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--self-test", action="store_true", help="print backend validation and exit")
        args = parser.parse_args()
        if args.self_test:
            print(json.dumps(get_backend().validate(), indent=2))
            return
        if mcp is None:
            print(MCP_MISSING_ERROR, file=sys.stderr)
            raise SystemExit(2)
        mcp.run()

    return ServerRuntime(mcp=mcp, get_backend=get_backend, tool=tool, run=run)
