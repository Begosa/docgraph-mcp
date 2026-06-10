#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


def commit(b: DocGraphBackend, reason: str, mutations: list[dict]) -> None:
    prop = b.propose_update(reason, mutations, created_by="fts_render_validation_test")
    b.commit_update(prop["proposal_id"])


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_fts_render_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "evidence.md").write_text("A source chunk for graph validation evidence.\n", encoding="utf-8")
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "evidence.md")
        chunk_id = b.conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (ingest["source_id"],)).fetchone()["chunk_id"]
        commit(
            b,
            "seed fts validation graph",
            [
                {
                    "op": "upsert_node",
                    "node_id": "concept.rare_node_summary",
                    "node_type": "concept",
                    "canonical_name": "ordinary_concept",
                    "summary": "This node summary contains the raretermalpha discovery marker.",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                },
                {
                    "op": "upsert_node",
                    "node_id": "flow.source_flow",
                    "node_type": "flow",
                    "canonical_name": "source_flow",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                },
                {
                    "op": "add_edge",
                    "edge_id": "edge.rare_summary",
                    "from_node_id": "concept.rare_node_summary",
                    "relation": "depends_on",
                    "to_node_id": "flow.source_flow",
                    "summary": "Relationship profile contains rareedgebeta routing marker.",
                    "visibility": "shared",
                    "audience_roles": ["firmware", "architecture"],
                },
                {
                    "op": "upsert_claim",
                    "claim_id": "claim.rare_node.evidence",
                    "target_node_id": "concept.rare_node_summary",
                    "claim_text": "raretermalpha is backed by source evidence.",
                    "classification": "Fact",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                    "chunk_ids": [chunk_id],
                },
            ],
        )

        node_search = b.search("raretermalpha", role="firmware", limit=10)
        assert any(r["kind"] == "node" and r["id"] == "concept.rare_node_summary" for r in node_search["results"]), node_search
        edge_search = b.search("rareedgebeta", role="firmware", limit=10)
        assert any(r["kind"] == "edge" and r["id"] == "edge.rare_summary" for r in edge_search["results"]), edge_search

        render = b.render_docs("docs/custom-rendered")
        assert (Path(render["output_dir"]) / "index.md").exists()
        try:
            b.render_docs(tmp / "outside-render")
        except ValueError as exc:
            assert "outside DOCGRAPH_ROOT" in str(exc), exc
        else:
            raise AssertionError("render_docs outside root should fail")

        try:
            b.propose_update(
                "claim without target should fail validation",
                [
                    {
                        "op": "upsert_claim",
                        "claim_id": "claim.no_target.bad",
                        "claim_text": "This claim intentionally has no target.",
                        "classification": "Fact",
                        "chunk_ids": [chunk_id],
                    }
                ],
                created_by="fts_render_validation_test",
            )
        except ValueError as exc:
            assert "claim_without_target" in str(exc), exc
        else:
            raise AssertionError("non-open-question claim without target should fail during proposal preflight")

        assert b.conn.execute("SELECT COUNT(*) AS n FROM schema_migrations WHERE version='5'").fetchone()["n"] == 1
        assert b.conn.execute("SELECT COUNT(*) AS n FROM schema_migrations WHERE version='6'").fetchone()["n"] == 1
        assert "trace_json" in {r["name"] for r in b.conn.execute("PRAGMA table_info(retrieval_runs)").fetchall()}
        print("SEARCH_FTS_RENDER_VALIDATION_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
