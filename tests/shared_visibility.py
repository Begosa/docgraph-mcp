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
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_shared_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        src = repo / "arch_notes.md"
        src.write_text(
            "PROCESS_X sits in the processing chain and has 8 input channels and 16 output channels.\n",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "arch_notes.md")
        chunk_id = b.conn.execute("SELECT chunk_id FROM chunks WHERE source_id=?", (ingest["source_id"],)).fetchone()["chunk_id"]
        mutations = [
            {"op": "upsert_node", "node_id": "flow.processing_chain", "node_type": "flow", "canonical_name": "processing_chain", "summary": "High-level processing chain."},
            {"op": "upsert_node", "node_id": "block.process_x", "node_type": "block", "canonical_name": "PROCESS_X", "summary": "Processing block discovered from RTL."},
            {
                "op": "upsert_node",
                "node_id": "block.restricted_tap",
                "node_type": "block",
                "canonical_name": "process_x_restricted_tap",
                "summary": "RTL-only internal tap around PROCESS_X.",
                "visibility": "local",
                "finder_role": "rtl",
                "audience_roles": ["rtl"],
            },
            {"op": "add_alias", "node_id": "block.restricted_tap", "alias": "PROCESS_X_RESTRICTED"},
            {
                "op": "add_edge",
                "edge_id": "edge.process_x.part_of.processing_chain",
                "from_node_id": "block.process_x",
                "relation": "belongs_to",
                "to_node_id": "flow.processing_chain",
                "summary": "PROCESS_X is part of the processing chain.",
                "visibility": "shared",
                "finder_role": "rtl",
                "audience_roles": ["rtl", "firmware", "test_debug", "architecture"],
                "interface_tags": ["data_path", "channel_count"],
            },
            {
                "op": "upsert_claim",
                "claim_id": "claim.process_x.channels.shared",
                "target_node_id": "block.process_x",
                "claim_text": "PROCESS_X has 8 input channels and 16 output channels.",
                "classification": "Fact",
                "confidence": "medium",
                "visibility": "shared",
                "finder_role": "rtl",
                "audience_roles": ["rtl", "firmware", "test_debug", "architecture"],
                "interface_tags": ["data_path", "channel_count"],
                "chunk_ids": [chunk_id],
            },
            {
                "op": "upsert_claim",
                "claim_id": "claim.process_x.internal.local",
                "target_node_id": "block.process_x",
                "claim_text": "PROCESS_X has an RTL-local internal staging signal detail.",
                "classification": "Fact",
                "confidence": "low",
                "visibility": "local",
                "finder_role": "rtl",
                "audience_roles": ["rtl"],
                "interface_tags": [],
                "chunk_ids": [chunk_id],
            },
            {
                "op": "add_edge",
                "edge_id": "edge.process_x.secret.local",
                "from_node_id": "block.process_x",
                "relation": "depends_on",
                "to_node_id": "block.restricted_tap",
                "summary": "RTL-only implementation edge.",
                "visibility": "local",
                "finder_role": "rtl",
                "audience_roles": ["rtl"],
            },
        ]
        prop = b.propose_update("shared visibility seed", mutations, created_by="shared_visibility_test")
        b.commit_update(prop["proposal_id"])

        fw_ctx = b.context(anchors=["block.process_x"], role="firmware", budget="medium")
        assert "PROCESS_X has 8 input channels" in fw_ctx["markdown"], fw_ctx["markdown"]
        assert "RTL-local internal staging" not in fw_ctx["markdown"], fw_ctx["markdown"]
        assert fw_ctx["cross_role_notes"], fw_ctx

        rtl_ctx = b.context(anchors=["block.process_x"], role="rtl", budget="medium")
        assert "RTL-local internal staging" in rtl_ctx["markdown"], rtl_ctx["markdown"]

        fw_search = b.search("PROCESS_X_RESTRICTED", role="firmware", limit=10)
        assert not any(r["id"] == "block.restricted_tap" for r in fw_search["results"]), fw_search
        assert not any(a["node_id"] == "block.restricted_tap" for a in fw_search["anchors"]), fw_search

        fw_ctx_secret = b.context(anchors=["PROCESS_X_RESTRICTED"], role="firmware", budget="small")
        assert not any(a["node_id"] == "block.restricted_tap" for a in fw_ctx_secret["selected_anchors"]), fw_ctx_secret

        fw_ctx_secret_direct = b.context(anchors=["block.restricted_tap"], role="firmware", budget="small")
        assert not any(a["node_id"] == "block.restricted_tap" for a in fw_ctx_secret_direct["selected_anchors"]), fw_ctx_secret_direct

        rtl_search = b.search("PROCESS_X_RESTRICTED", role="rtl", limit=10)
        assert any(r["id"] == "block.restricted_tap" for r in rtl_search["results"]), rtl_search

        check = b.related_context_check(
            "RTL found PROCESS_X has 8 input channels and 16 output channels",
            finder_role="rtl",
            interface_tags=["data_path", "channel_count"],
        )
        assert check["suggested_visibility"] in {"shared", "shared_candidate"}, check
        assert any(n["node_id"] == "flow.processing_chain" for n in check["high_level_candidates"]), check

        print("SHARED_VISIBILITY_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
