#!/usr/bin/env python3
"""Execute one real DocGraph context retrieval for GUI/live inspection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("DOCGRAPH_ROOT", Path.cwd()))
DEFAULT_DB = Path(os.environ.get("DOCGRAPH_DB", "docs/docgraph.sqlite"))
sys.path.insert(0, str(BUNDLE_ROOT / "src"))

from docgraph_mcp import DocGraphBackend  # noqa: E402

RESULT_PREFIX = "DOCGRAPH_QUERY_RESULT="


def parse_anchors(value: str | None) -> list[str] | None:
    if not value:
        return None
    anchors = [part.strip() for part in value.split(",") if part.strip()]
    return anchors or None


def default_config_path(root: Path) -> Path | None:
    env_config = os.environ.get("DOCGRAPH_CONFIG")
    if env_config:
        config = Path(env_config).expanduser()
        return config if config.is_absolute() else root / config
    for candidate in (
        root / ".opencode" / "docgraph" / "docgraph.config.yaml",
        root / "docgraph.config.yaml",
        BUNDLE_ROOT / "docgraph.config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def run_context(
    root: str | Path,
    *,
    db: str | Path,
    config: str | Path | None,
    query: str | None,
    anchors: list[str] | None,
    role: str | None,
    intent: str | None,
    mode: str,
    budget: str,
) -> dict[str, Any]:
    project_root = Path(root).expanduser().resolve()
    db_path = Path(db).expanduser()
    if not db_path.is_absolute():
        db_path = project_root / db_path
    db_path = db_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DocGraph database not found: {db_path}")
    config_path = Path(config).expanduser() if config else default_config_path(project_root)
    if config_path is not None and not config_path.is_absolute():
        config_path = project_root / config_path
    backend = DocGraphBackend(
        db_path=db_path,
        root=project_root,
        config_path=config_path if config_path.exists() else None,
    )
    try:
        packet = backend.context(
            anchors=anchors,
            query=query or None,
            role=role or None,
            intent=intent or None,
            mode=mode,
            budget=budget,
        )
        return {
            "ok": True,
            "root": str(project_root),
            "database": str(db_path),
            "run_id": backend.last_retrieval_run_id,
            "query": query,
            "anchors": anchors or [],
            "mode": packet["mode"],
            "role": packet.get("role"),
            "intent": packet.get("intent"),
            "budget": packet["budget"],
            "selected_anchor_ids": [node.get("node_id") for node in packet["selected_anchors"]],
            "claim_ids": [claim.get("claim_id") for claim in packet["active_claims"]],
            "edge_ids": [edge.get("edge_id") for edge in packet["related_edges"]],
            "semantic_available": packet.get("semantic_candidates", {}).get("available", False),
            "semantic_reason": packet.get("semantic_candidates", {}).get("reason"),
            "markdown": packet["markdown"],
        }
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a DocGraph context query through the real backend.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Project root used as DOCGRAPH_ROOT")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path; relative paths are resolved from --root")
    parser.add_argument("--config", default="", help="Config path; defaults to bundled .opencode/docgraph config when present")
    parser.add_argument("--query", default="", help="Natural-language retrieval query")
    parser.add_argument("--anchors", default="", help="Optional comma-separated node names/aliases/IDs")
    parser.add_argument("--role", default="", help="Requesting role")
    parser.add_argument("--intent", default="", help="Request intent")
    parser.add_argument("--mode", default="local", choices=["local", "global", "bridge", "hybrid", "mix"])
    parser.add_argument("--budget", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--json-line", action="store_true", help="Emit one machine-readable result line for the GUI")
    args = parser.parse_args(argv)
    if not args.query.strip() and not parse_anchors(args.anchors):
        parser.error("provide --query, --anchors, or both")
    try:
        result = run_context(
            args.root,
            db=args.db,
            config=args.config.strip() or None,
            query=args.query.strip() or None,
            anchors=parse_anchors(args.anchors),
            role=args.role.strip() or None,
            intent=args.intent.strip() or None,
            mode=args.mode,
            budget=args.budget,
        )
    except Exception as exc:
        result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        if args.json_line:
            print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    if args.json_line:
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    else:
        print(result["markdown"])
        print(f"Stored retrieval trace: {result['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
