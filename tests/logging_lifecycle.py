#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_logging_lifecycle_"))
    old_env = {k: os.environ.get(k) for k in ["DOCGRAPH_ROOT", "DOCGRAPH_DB", "DOCGRAPH_CONFIG", "DOCGRAPH_LOG_LEVEL"]}
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
logging:
  enabled: true
  level: debug
  file: docs/logs/lifecycle.log
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )
        (repo / "source.md").write_text("Lifecycle evidence about pixel_gap.\n", encoding="utf-8")
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "source.md")
        chunk_id = b.conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (ingest["source_id"],)).fetchone()["chunk_id"]
        prop = b.propose_update(
            "logging lifecycle proposal",
            [
                {"op": "upsert_node", "node_id": "register.pixel_gap", "node_type": "register", "canonical_name": "pixel_gap"},
                {"op": "upsert_claim", "claim_id": "claim.pixel_gap.lifecycle", "target_node_id": "register.pixel_gap", "claim_text": "pixel_gap has lifecycle evidence.", "classification": "Fact", "chunk_ids": [chunk_id]},
            ],
            created_by="logging_lifecycle_test",
        )
        b.commit_update(prop["proposal_id"])
        b.resolve("pixel_gap")
        b.search("pixel gap", role="firmware")
        b.context(anchors=["pixel_gap"], query="pixel gap", role="firmware", mode="mix")
        b.related_context_check("pixel_gap register config at frame timing", finder_role="firmware", interface_tags=["register", "timing"])
        b.render_docs()
        b.stale_scan(auto_ingest=False)
        b.validate()
        b.close()

        # Exercise MCP boundary wrapper without starting a server.
        os.environ["DOCGRAPH_ROOT"] = str(repo)
        os.environ["DOCGRAPH_DB"] = str(repo / "docs" / "docgraph.sqlite")
        os.environ["DOCGRAPH_LOG_LEVEL"] = "debug"
        from docgraph_mcp import read_server  # noqa: WPS433,E402

        read_server.backend = None
        read_server.dg_validate()

        log_path = repo / "docs" / "logs" / "lifecycle.log"
        events = [json.loads(line).get("event") for line in log_path.read_text(encoding="utf-8").splitlines()]
        required = {
            "ingest.start",
            "ingest.done",
            "proposal.start",
            "proposal.done",
            "commit.start",
            "mutation.apply",
            "commit.done",
            "resolve.start",
            "resolve.done",
            "search.start",
            "search.done",
            "context.start",
            "context.done",
            "semantic.disabled_or_unavailable",
            "related_context_check.start",
            "related_context_check.done",
            "render_docs.start",
            "render_docs.done",
            "stale_scan.start",
            "stale_scan.done",
            "validate.start",
            "validate.done",
            "mcp.tool.start",
            "mcp.tool.done",
        }
        missing = sorted(required - set(events))
        assert not missing, missing
        print("LOGGING_LIFECYCLE_OK")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
