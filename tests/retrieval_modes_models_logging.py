#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


def commit(b: DocGraphBackend, reason: str, mutations: list[dict]) -> None:
    prop = b.propose_update(reason, mutations, created_by="retrieval_modes_test")
    b.commit_update(prop["proposal_id"])


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_modes_models_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
logging:
  enabled: true
  level: debug
  file: docs/logs/test-docgraph-mcp.log
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )
        (repo / "evidence.md").write_text(
            "Pixel gap registers belong to line timing. Line timing is part of the frame lifecycle.\n",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "evidence.md")
        chunk_id = b.conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (ingest["source_id"],)).fetchone()["chunk_id"]
        commit(
            b,
            "seed graph",
            [
                {"op": "upsert_node", "node_id": "register.pixel_gap", "node_type": "register", "canonical_name": "pixel_gap", "summary": "Pixel gap timing register.", "visibility": "shared", "audience_roles": ["firmware", "rtl", "architecture"], "interface_tags": ["register", "timing"]},
                {"op": "upsert_node", "node_id": "feature.line_timing", "node_type": "feature", "canonical_name": "line_timing", "summary": "Feature covering line timing and pixel spacing.", "visibility": "global", "audience_roles": ["firmware", "rtl", "architecture"]},
                {"op": "upsert_node", "node_id": "flow.frame_lifecycle", "node_type": "flow", "canonical_name": "frame_lifecycle", "summary": "High-level per-frame lifecycle and timing update flow.", "visibility": "global", "audience_roles": ["firmware", "rtl", "test_debug", "architecture"]},
                {"op": "add_alias", "node_id": "register.pixel_gap", "alias": "pixel gap"},
                {"op": "add_alias", "node_id": "flow.frame_lifecycle", "alias": "each frame"},
                {"op": "add_edge", "edge_id": "edge.pixel_gap.belongs.line_timing", "from_node_id": "register.pixel_gap", "relation": "belongs_to", "to_node_id": "feature.line_timing", "summary": "pixel_gap is grouped under line timing.", "visibility": "shared", "audience_roles": ["firmware", "rtl", "architecture"]},
                {"op": "add_edge", "edge_id": "edge.line_timing.part.frame", "from_node_id": "feature.line_timing", "relation": "part_of", "to_node_id": "flow.frame_lifecycle", "summary": "Line timing participates in the frame lifecycle.", "visibility": "global", "audience_roles": ["firmware", "rtl", "architecture"]},
                {"op": "upsert_claim", "claim_id": "claim.pixel_gap.line_timing", "target_edge_id": "edge.pixel_gap.belongs.line_timing", "claim_text": "pixel_gap belongs to line timing.", "classification": "Fact", "visibility": "shared", "audience_roles": ["firmware", "rtl", "architecture"], "chunk_ids": [chunk_id]},
            ],
        )
        local = b.context(anchors=["pixel gap"], mode="local", role="firmware", budget="small")
        assert local["mode"] == "local"
        assert not local["bridge_paths"]
        assert "trace" not in b.search("pixel gap", role="firmware"), "normal search responses must not expose GUI telemetry"
        global_ctx = b.context(anchors=["pixel gap"], query="pixel gap each frame", mode="global", role="firmware", budget="medium")
        assert any(f["node"]["node_id"] == "flow.frame_lifecycle" for f in global_ctx["global_frames"]), global_ctx["markdown"]
        bridge = b.context(anchors=["pixel gap", "each frame"], mode="bridge", role="firmware", budget="medium")
        assert bridge["bridge_paths"], bridge["markdown"]
        hybrid = b.context(anchors=["pixel gap", "each frame"], query="pixel gap each frame", mode="hybrid", role="firmware", budget="medium")
        assert hybrid["global_frames"] and hybrid["bridge_paths"]
        mix = b.context(anchors=["pixel gap", "each frame"], query="pixel gap each frame", mode="mix", role="firmware", budget="small")
        assert mix["semantic_candidates"]["available"] is False
        assert "embeddings disabled" in (mix["semantic_candidates"]["reason"] or "")
        assert "retrieval_trace" not in mix, "GUI telemetry must not inflate context packets returned to agents"
        latest_run = b.conn.execute("SELECT trace_json FROM retrieval_runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        trace = json.loads(latest_run["trace_json"])
        assert trace["final"]["anchor_ids"], trace
        assert "lexical_search" in trace["anchor_resolution"], trace
        assert trace["semantic_promotion"]["enabled"] is False, trace
        assert "timings_ms" in trace, trace
        log_path = repo / "docs" / "logs" / "test-docgraph-mcp.log"
        assert log_path.exists()
        events = [json.loads(line).get("event") for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert "context.start" in events and "context.done" in events
        print("RETRIEVAL_MODES_MODELS_LOGGING_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
