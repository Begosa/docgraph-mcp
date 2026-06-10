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
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_mut_alias_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "old_build_doc.md").write_text("The old docs describe a build flow and test flow.\n", encoding="utf-8")
        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)
        ingest = b.ingest_source("doc_file", "old_build_doc.md")
        assert ingest["chunk_ids"], ingest
        assert ingest["chunk_refs"][0]["chunk_id"] == ingest["chunk_ids"][0], ingest
        chunk_id = ingest["chunk_ids"][0]

        schema = b.mutation_schema()
        assert "upsert_node" in schema["canonical_ops"]
        assert schema["op_aliases"]["upsert_edge"] == "add_edge"
        assert schema["op_aliases"]["node"] == "upsert_node"
        assert "flow" in schema["taxonomy_contract"]["node_types"]
        assert "depends_on" in schema["taxonomy_contract"]["relation_types"]
        assert "Fact" in schema["taxonomy_contract"]["claim_classes"]

        mutations = [
            {"op": "node", "type": "flow", "name": "build_flow", "summary": "Build flow from old docs."},
            {"op": "upsert_alias", "node_id": "flow.build_flow", "name": "build process"},
            {"op": "node", "node_type": "flow", "canonical_name": "test_flow"},
            {"op": "upsert_edge", "from": "flow.build_flow", "relation_type": "depends_on", "to": "flow.test_flow"},
            {
                "op": "claim",
                "claim_id": "claim.build_flow.old_doc",
                "target_node_id": "flow.build_flow",
                "text": "The old docs describe a build flow.",
                "classification": "Hypothesis",
                "confidence": "low",
            },
            {"op": "evidence", "claim_id": "claim.build_flow.old_doc", "chunk_id": chunk_id, "role": "supports"},
        ]
        prop = b.propose_update("mutation alias compatibility", mutations, created_by="test")
        b.commit_update(prop["proposal_id"])
        v = b.validate()
        assert v["ok"], v
        assert b._get_node("flow.build_flow") is not None
        assert b.conn.execute("SELECT COUNT(*) AS n FROM edges WHERE relation='depends_on'").fetchone()["n"] == 1

        # Updating an existing logical edge with a different edge_id should not fail
        # or leave orphan FTS rows.
        prop_edge_update = b.propose_update(
            "edge upsert keeps canonical edge_id",
            [
                {
                    "op": "add_edge",
                    "edge_id": "edge.override.depends",
                    "from_node_id": "flow.build_flow",
                    "relation": "depends_on",
                    "to_node_id": "flow.test_flow",
                    "summary": "Updated edge summary.",
                }
            ],
            created_by="test",
        )
        b.commit_update(prop_edge_update["proposal_id"])
        edge_row = b.conn.execute(
            "SELECT edge_id, summary FROM edges WHERE from_node_id='flow.build_flow' AND relation='depends_on' AND to_node_id='flow.test_flow'"
        ).fetchone()
        assert edge_row is not None
        assert edge_row["summary"] == "Updated edge summary."
        orphan_edge_fts = b.conn.execute(
            "SELECT COUNT(*) AS n FROM edges_fts ef LEFT JOIN edges e ON e.edge_id=ef.edge_id WHERE e.edge_id IS NULL"
        ).fetchone()["n"]
        assert orphan_edge_fts == 0

        # Updating an existing logical alias with a different alias_id should not
        # create stale aliases_fts rows.
        prop_alias_update = b.propose_update(
            "alias upsert keeps canonical alias_id",
            [
                {
                    "op": "add_alias",
                    "alias_id": "alias.override.build_process",
                    "node_id": "flow.build_flow",
                    "alias": "BuildProcess",
                    "alias_kind": "name",
                }
            ],
            created_by="test",
        )
        b.commit_update(prop_alias_update["proposal_id"])
        alias_row = b.conn.execute(
            "SELECT alias_id, alias FROM aliases WHERE node_id='flow.build_flow' AND normalized_alias=?",
            ("build_process",),
        ).fetchone()
        assert alias_row is not None
        assert alias_row["alias"] == "BuildProcess"
        orphan_alias_fts = b.conn.execute(
            "SELECT COUNT(*) AS n FROM aliases_fts af LEFT JOIN aliases a ON a.alias_id=af.alias_id WHERE a.alias_id IS NULL"
        ).fetchone()["n"]
        assert orphan_alias_fts == 0

        # Compact claim evidence shortcut should expand into attach_evidence.
        shortcut = [
            {"op": "node", "type": "runbook", "name": "debug_runbook"},
            {
                "op": "claim",
                "claim_id": "claim.debug_runbook.shortcut",
                "target_node_id": "runbook.debug_runbook",
                "text": "Old docs mention a debug runbook.",
                "classification": "Hypothesis",
                "confidence": "low",
                "chunk_ids": [chunk_id],
            },
        ]
        prop2 = b.propose_update("claim chunk_ids shortcut", shortcut, created_by="test")
        b.commit_update(prop2["proposal_id"])
        ev = b.conn.execute(
            "SELECT COUNT(*) AS n FROM claim_evidence WHERE claim_id='claim.debug_runbook.shortcut' AND chunk_id=?",
            (chunk_id,),
        ).fetchone()["n"]
        assert ev == 1

        # Generated rendered docs must not be ingested as source evidence.
        rendered = repo / "docs" / "rendered" / "architecture.md"
        rendered.parent.mkdir(parents=True, exist_ok=True)
        rendered.write_text("Generated by render_docs\n\n# Architecture\n", encoding="utf-8")
        try:
            b.ingest_source("doc_file", "docs/rendered/architecture.md")
        except ValueError as exc:
            assert "refusing to ingest generated DocGraph output" in str(exc), exc
        else:
            raise AssertionError("generated rendered docs should be rejected")

        # Helpful missing-field error should be explicit.
        try:
            b.propose_update("bad node", [{"op": "upsert_node", "node_type": "flow"}], created_by="test")
        except ValueError as exc:
            assert "upsert_node requires canonical_name" in str(exc), exc
        else:
            raise AssertionError("missing canonical_name should fail")

        # Curator errors should be typed before SQLite can raise opaque FK errors.
        try:
            b.propose_update(
                "bad target node",
                [
                    {
                        "op": "upsert_claim",
                        "claim_id": "claim.bad_target",
                        "target_node_id": "flow.not_created",
                        "claim_text": "This open question points at a missing node.",
                        "classification": "OpenQuestion",
                    }
                ],
                created_by="test",
            )
        except ValueError as exc:
            assert "missing_node_reference" in str(exc), exc
        else:
            raise AssertionError("missing target_node_id should fail during proposal preflight")

        try:
            b.propose_update(
                "bad guessed chunk",
                [
                    {"op": "upsert_node", "node_id": "flow.bad_evidence", "node_type": "flow", "canonical_name": "bad_evidence"},
                    {
                        "op": "upsert_claim",
                        "claim_id": "claim.bad_guessed_chunk",
                        "target_node_id": "flow.bad_evidence",
                        "claim_text": "This claim uses a guessed chunk ID.",
                        "classification": "Fact",
                        "chunk_ids": ["chunk_src_fake_ep_fake_0"],
                    },
                ],
                created_by="test",
            )
        except ValueError as exc:
            assert "missing_chunk_reference" in str(exc), exc
            assert "must not be inferred or constructed" in str(exc), exc
        else:
            raise AssertionError("guessed chunk_id should fail during proposal preflight")

        try:
            b.propose_update(
                "unsupported fact",
                [
                    {"op": "upsert_node", "node_id": "flow.no_evidence", "node_type": "flow", "canonical_name": "no_evidence"},
                    {
                        "op": "upsert_claim",
                        "claim_id": "claim.no_evidence",
                        "target_node_id": "flow.no_evidence",
                        "claim_text": "This fact has no evidence.",
                        "classification": "Fact",
                    },
                ],
                created_by="test",
            )
        except ValueError as exc:
            assert "active_claim_without_active_support" in str(exc), exc
        else:
            raise AssertionError("active Fact without support should fail during proposal preflight")

        try:
            b.propose_update(
                "untargeted fact",
                [
                    {
                        "op": "upsert_claim",
                        "claim_id": "claim.no_target",
                        "claim_text": "This fact has no target.",
                        "classification": "Fact",
                        "chunk_ids": [chunk_id],
                    }
                ],
                created_by="test",
            )
        except ValueError as exc:
            assert "claim_without_target" in str(exc), exc
        else:
            raise AssertionError("non-OpenQuestion claim without target should fail during proposal preflight")

        print("MUTATION_SCHEMA_ALIASES_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
