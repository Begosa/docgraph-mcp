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
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_smoke_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "Firmware" / "timing").mkdir(parents=True)
        src_file = repo / "Firmware" / "timing" / "pixel_gap.c"
        src_file.write_text(
            """
            void configure_pixel_gap(int value) {
              // PIX_GAP controls output pixel spacing and is relevant to line rate debug.
              write_reg(PIX_GAP, value);
            }
            """,
            encoding="utf-8",
        )
        (repo / "docgraph.config.yaml").write_text(
            """
roles:
  firmware:
    preferred_node_types: [function, file, register, field, flow, fw_symbol]
    preferred_relations: [configures, writes, reads, calls, affects, implements]
    suggested_checks:
      - CUSTOM_FW_CONFIG_CHECK from project config.
retrieval:
  budgets:
    small:
      nodes: 5
      claims: 4
      edges: 8
      evidence: 2
""",
            encoding="utf-8",
        )
        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)
        ingest = b.ingest_source("code_file", "Firmware/timing/pixel_gap.c")
        assert ingest["status"] == "ingested"
        assert ingest["chunk_ids"], ingest
        chunk_id = ingest["chunk_ids"][0]

        mutations = [
            {
                "op": "upsert_node",
                "node_id": "register.pixel_gap",
                "node_type": "register",
                "canonical_name": "pixel_gap",
                "summary": "Controls output pixel spacing configuration.",
            },
            {
                "op": "upsert_node",
                "node_id": "concept.line_rate",
                "node_type": "concept",
                "canonical_name": "line_rate",
                "summary": "Output line rate / timing stability concept.",
            },
            {"op": "add_alias", "node_id": "register.pixel_gap", "alias": "PIX_GAP", "alias_kind": "register_name"},
            {"op": "add_alias", "node_id": "register.pixel_gap", "alias": "PixelGap", "alias_kind": "doc_name"},
            {
                "op": "add_edge",
                "from_node_id": "register.pixel_gap",
                "relation": "affects",
                "to_node_id": "concept.line_rate",
                "confidence": "medium",
            },
            {
                "op": "upsert_claim",
                "claim_id": "claim.pixel_gap.line_rate_debug",
                "target_node_id": "register.pixel_gap",
                "claim_text": "pixel_gap configuration is relevant when debugging line-rate instability.",
                "classification": "Fact",
                "confidence": "medium",
            },
            {
                "op": "attach_evidence",
                "claim_id": "claim.pixel_gap.line_rate_debug",
                "chunk_id": chunk_id,
                "evidence_role": "supports",
                "strength": "medium",
            },
        ]
        prop = b.propose_update("seed pixel_gap timing knowledge", mutations, created_by="smoke_test")
        commit = b.commit_update(prop["proposal_id"])
        assert commit["after_revision"] == "1"
        validation = b.validate()
        assert validation["ok"], validation

        resolved = b.resolve("PIX_GAP")
        assert resolved["matches"] and resolved["matches"][0]["node_id"] == "register.pixel_gap"
        search = b.search("line rate unstable", role="firmware", intent="debug")
        assert search["anchors"], json.dumps(search, indent=2)
        ctx = b.context(query="line rate unstable", role="firmware", intent="debug", budget="small")
        assert "register.pixel_gap" in ctx["markdown"], ctx["markdown"]
        assert "CUSTOM_FW_CONFIG_CHECK" in ctx["markdown"], ctx["markdown"]
        render = b.render_docs()
        assert render["rendered_nodes"] >= 2

        # Re-ingest changed file; old evidence should become stale and claim needs review.
        src_file.write_text(
            """
            void configure_new_timing(int value) {
              // pixel gap moved elsewhere; this file no longer writes the old register.
              (void)value;
            }
            """,
            encoding="utf-8",
        )
        ingest2 = b.ingest_source("code_file", "Firmware/timing/pixel_gap.c")
        assert ingest2["superseded_episode_id"], ingest2
        claim_status = b.conn.execute("SELECT status FROM claims WHERE claim_id='claim.pixel_gap.line_rate_debug'").fetchone()["status"]
        assert claim_status == "needs_review", claim_status

        print("SMOKE_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
