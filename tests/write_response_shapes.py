#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_write_shapes_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)

        compact_schema = b.mutation_schema()
        full_schema = b.mutation_schema(detail="full")
        assert compact_schema["detail"] == "compact", compact_schema
        assert "example" not in compact_schema["ops"]["upsert_node"], compact_schema["ops"]["upsert_node"]
        assert "example" in full_schema["ops"]["upsert_node"], full_schema["ops"]["upsert_node"]

        for idx in range(5):
            b.conn.execute(
                "INSERT INTO claims(claim_id, target_node_id, target_edge_id, claim_text, classification, confidence, status, visibility, finder_role, audience_roles_json, interface_tags_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"claim.broken.{idx}", None, None, "Broken claim", "Fact", "high", "active", "local", None, "[]", "[]", "now", "now"),
            )
        compact_validation = b.validate(limit=2)
        assert compact_validation["ok"] is False, compact_validation
        assert compact_validation["error_count"] >= 5, compact_validation
        assert len(compact_validation["errors"]) == 2, compact_validation
        assert compact_validation["truncated"] is True, compact_validation
        full_validation = b.validate(detail="full")
        assert len(full_validation["errors"]) == full_validation["error_count"], full_validation

        # Use a clean backend for stale-scan response shaping.
        repo2 = tmp / "repo2"
        repo2.mkdir()
        (repo2 / "src").mkdir()
        b2 = DocGraphBackend(repo2 / "docs" / "docgraph.sqlite", root=repo2)
        for idx in range(3):
            path = repo2 / "src" / f"f{idx}.c"
            path.write_text(f"int old_{idx};\n", encoding="utf-8")
            b2.ingest_source("code_file", f"src/f{idx}.c")
            path.write_text(f"int new_{idx};\n", encoding="utf-8")
        stale = b2.stale_scan(result_limit=1)
        assert stale["changed_count"] == 3, stale
        assert len(stale["changed"]) == 1, stale
        assert stale["truncated"] is True, stale
        stale_full = b2.stale_scan(detail="full")
        assert len(stale_full["changed"]) == 3, stale_full

        print("WRITE_RESPONSE_SHAPES_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
