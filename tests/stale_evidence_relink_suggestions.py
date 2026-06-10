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
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_relink_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        src_dir = repo / "Firmware" / "clock"
        src_dir.mkdir(parents=True)
        src = src_dir / "clock_mux.c"
        src.write_text(
            """
            void configure_dbr_mux(int source) {
              // Selects DBR clock source.
              write_reg(DBR_CLK_SEL, source);
            }
            """,
            encoding="utf-8",
        )
        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)

        first = b.ingest_source("code_file", "Firmware/clock/clock_mux.c")
        assert first["status"] == "ingested", first
        old_chunk_id = first["chunk_ids"][0]

        mutations = [
            {
                "op": "upsert_node",
                "node_id": "register.dbr_clk_sel",
                "node_type": "register",
                "canonical_name": "DBR_CLK_SEL",
                "summary": "DBR clock source selector.",
            },
            {
                "op": "upsert_claim",
                "claim_id": "claim.dbr_clk_sel.write",
                "target_node_id": "register.dbr_clk_sel",
                "claim_text": "Firmware writes DBR_CLK_SEL to select the DBR clock source.",
                "classification": "Fact",
                "confidence": "high",
            },
            {
                "op": "attach_evidence",
                "claim_id": "claim.dbr_clk_sel.write",
                "chunk_id": old_chunk_id,
                "evidence_role": "supports",
                "strength": "high",
            },
        ]
        prop = b.propose_update("seed DBR clock selector claim", mutations, created_by="test")
        b.commit_update(prop["proposal_id"])

        src.write_text(
            """
            // Local comment changed; behavior below is unchanged.
            void configure_dbr_mux(int source)
            {
              write_reg(DBR_CLK_SEL, source);
            }
            """,
            encoding="utf-8",
        )
        second = b.ingest_source("code_file", "Firmware/clock/clock_mux.c")
        assert second["status"] == "ingested", second
        assert old_chunk_id in second["stale_chunk_ids"], second
        assert second["affected_claim_ids"] == ["claim.dbr_clk_sel.write"], second
        assert second["claims_marked_needs_review"] == ["claim.dbr_clk_sel.write"], second

        status = b.conn.execute("SELECT status FROM claims WHERE claim_id='claim.dbr_clk_sel.write'").fetchone()["status"]
        assert status == "needs_review", status
        active_links_before = b.conn.execute(
            "SELECT COUNT(*) AS n FROM claim_evidence WHERE claim_id='claim.dbr_clk_sel.write' AND status='active'"
        ).fetchone()["n"]
        assert active_links_before == 0, active_links_before

        suggestion = b.suggest_evidence_relinks("claim.dbr_clk_sel.write", limit=5)
        assert suggestion["recommendation"] == "safe_equivalent_relink_candidate", suggestion
        assert suggestion["candidates"], suggestion
        best = suggestion["candidates"][0]
        assert best["support_level"] == "equivalent", suggestion
        assert any("normalized token hash matches" in reason for reason in best["reasons"]), suggestion
        assert best["chunk_id"] in second["chunk_ids"], suggestion
        assert suggestion["draft_mutations"] == [
            {
                "op": "attach_evidence",
                "claim_id": "claim.dbr_clk_sel.write",
                "chunk_id": best["chunk_id"],
                "evidence_role": "supports",
                "strength": "high",
            }
        ], suggestion
        batch = b.suggest_source_relinks(source_id=second["source_id"], limit_per_claim=5)
        assert batch["summary"]["affected_claim_count"] == 1, batch
        assert batch["summary"]["safe_relink_claim_count"] == 1, batch
        assert batch["summary"]["review_candidate_claim_count"] == 0, batch
        assert batch["summary"]["unresolved_claim_count"] == 0, batch
        assert batch["draft_mutations"] == suggestion["draft_mutations"], batch

        batch_by_uri = b.suggest_source_relinks(uri="Firmware/clock/clock_mux.c", limit_per_claim=5)
        assert batch_by_uri["draft_mutations"] == batch["draft_mutations"], batch_by_uri

        # Suggestions are read-only; curator must still propose/commit any repair.
        status_after = b.conn.execute("SELECT status FROM claims WHERE claim_id='claim.dbr_clk_sel.write'").fetchone()["status"]
        assert status_after == "needs_review", status_after
        active_links_after = b.conn.execute(
            "SELECT COUNT(*) AS n FROM claim_evidence WHERE claim_id='claim.dbr_clk_sel.write' AND status='active'"
        ).fetchone()["n"]
        assert active_links_after == 0, active_links_after

        src.write_text(
            """
            void configure_dbr_mux(int source)
            {
              write_reg(DBR_CLK_SEL, source + 1);
            }
            """,
            encoding="utf-8",
        )
        third = b.ingest_source("code_file", "Firmware/clock/clock_mux.c")
        assert third["status"] == "ingested", third
        changed_suggestion = b.suggest_evidence_relinks("claim.dbr_clk_sel.write", limit=5)
        assert changed_suggestion["recommendation"] == "review_candidates", changed_suggestion
        assert changed_suggestion["candidates"], changed_suggestion
        assert not changed_suggestion["draft_mutations"], changed_suggestion
        assert all(c["support_level"] != "equivalent" for c in changed_suggestion["candidates"]), changed_suggestion

        changed_batch = b.suggest_source_relinks(source_id=third["source_id"], limit_per_claim=5)
        assert changed_batch["summary"]["affected_claim_count"] == 1, changed_batch
        assert changed_batch["summary"]["safe_relink_claim_count"] == 0, changed_batch
        assert changed_batch["summary"]["review_candidate_claim_count"] == 1, changed_batch
        assert changed_batch["summary"]["unresolved_claim_count"] == 0, changed_batch
        assert not changed_batch["draft_mutations"], changed_batch

        print("STALE_EVIDENCE_RELINK_SUGGESTIONS_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
