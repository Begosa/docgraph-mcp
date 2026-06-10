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


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_anchor_rank_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "evidence.md").write_text(
            "rarehightoken indicates the vsync timeout register and not unrelated neighbor blocks.\n",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "evidence.md")
        chunk_id = b.conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_id=? ORDER BY rowid LIMIT 1",
            (ingest["source_id"],),
        ).fetchone()["chunk_id"]

        proposal = b.propose_update(
            "seed anchor ranking case",
            [
                {
                    "op": "upsert_node",
                    "node_id": "node.a_neighbor",
                    "node_type": "block",
                    "canonical_name": "a_neighbor",
                    "summary": "Neighbor block connected by graph relation only.",
                },
                {
                    "op": "upsert_node",
                    "node_id": "node.z_target",
                    "node_type": "register",
                    "canonical_name": "z_target",
                    "summary": "rarehightoken register controlling vsync timeout.",
                },
                {
                    "op": "add_edge",
                    "edge_id": "edge.z_target.depends_on.a_neighbor",
                    "from_node_id": "node.z_target",
                    "relation": "depends_on",
                    "to_node_id": "node.a_neighbor",
                    "summary": "Target register depends on neighbor wiring.",
                },
                {
                    "op": "upsert_claim",
                    "claim_id": "claim.z_target.rarehightoken",
                    "target_node_id": "node.z_target",
                    "claim_text": "rarehightoken points to z_target timeout logic.",
                    "classification": "Fact",
                    "chunk_ids": [chunk_id],
                },
            ],
            created_by="search_anchor_ranking_test",
        )
        b.commit_update(proposal["proposal_id"])

        search = b.search("rarehightoken", role="firmware", limit=5)
        anchor_ids = [a["node_id"] for a in search["anchors"]]
        assert "node.a_neighbor" in anchor_ids and "node.z_target" in anchor_ids, json.dumps(search, indent=2)
        assert anchor_ids[0] == "node.z_target", json.dumps(search, indent=2)

        print("SEARCH_ANCHOR_RANKING_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
