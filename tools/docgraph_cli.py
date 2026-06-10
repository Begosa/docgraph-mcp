#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("DOCGRAPH_ROOT", Path.cwd()))
DEFAULT_DB = Path(os.environ.get("DOCGRAPH_DB", "docs/docgraph.sqlite"))

sys.path.insert(0, str(BUNDLE_ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--config", default="", help="Config path; defaults to bundled .opencode/docgraph config when present")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("search")
    q.add_argument("query")
    v = sub.add_parser("validate")
    r = sub.add_parser("render")
    i = sub.add_parser("ingest")
    i.add_argument("source_type")
    i.add_argument("uri")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    db = Path(args.db).expanduser()
    if not db.is_absolute():
        db = root / db
    config = Path(args.config).expanduser() if args.config else default_config_path(root)
    if config is not None and not config.is_absolute():
        config = root / config
    b = DocGraphBackend(db, root, config_path=config)
    if args.cmd == "search":
        out = b.search(args.query)
    elif args.cmd == "validate":
        out = b.validate()
    elif args.cmd == "render":
        out = b.render_docs()
    elif args.cmd == "ingest":
        out = b.ingest_source(args.source_type, args.uri)
    else:
        raise AssertionError
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
