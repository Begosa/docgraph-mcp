#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


BLOCK_COUNT = 80
REGISTER_COUNT = 220

SEARCH_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_SEARCH_MAX_MS", "5000"))
CONTEXT_LOCAL_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CONTEXT_LOCAL_MAX_MS", "10000"))
CONTEXT_GLOBAL_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CONTEXT_GLOBAL_MAX_MS", "22000"))
CONTEXT_BRIDGE_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CONTEXT_BRIDGE_MAX_MS", "12000"))
CONTEXT_HYBRID_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CONTEXT_HYBRID_MAX_MS", "22000"))
CURATOR_STEP_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CURATOR_STEP_MAX_MS", "9000"))
CURATOR_CYCLE_MAX_MS = float(os.environ.get("DOCGRAPH_STRESS_CURATOR_CYCLE_MAX_MS", "30000"))
CONTEXT_MAX_CHARS = int(os.environ.get("DOCGRAPH_STRESS_CONTEXT_MAX_CHARS", "180000"))
ACTIVE_CLAIMS_MAX = int(os.environ.get("DOCGRAPH_STRESS_CONTEXT_CLAIMS_MAX", "90"))
RELATED_EDGES_MAX = int(os.environ.get("DOCGRAPH_STRESS_CONTEXT_EDGES_MAX", "220"))
HYBRID_TO_GLOBAL_MAX_RATIO = float(os.environ.get("DOCGRAPH_STRESS_HYBRID_GLOBAL_RATIO_MAX", "80"))


def timed_call(fn):
    started = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return out, elapsed_ms


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_stress_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval:
  default_mode: local
  budgets:
    large:
      nodes: 40
      claims: 40
      edges: 90
      evidence: 8
""",
            encoding="utf-8",
        )

        evidence_lines = []
        for i in range(1, 401):
            evidence_lines.append(
                f"line {i}: soc boot uses clock/reset sequencing, simulation seed control, and protocol endpoints."
            )
        (repo / "soc_evidence.md").write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")

        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "soc_evidence.md")
        chunk_id = b.conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_id=? ORDER BY rowid LIMIT 1",
            (ingest["source_id"],),
        ).fetchone()["chunk_id"]

        seed_mutations: list[dict] = [
            {
                "op": "upsert_node",
                "node_id": "flow.soc_boot",
                "node_type": "flow",
                "canonical_name": "soc_boot",
                "summary": "SoC boot flow with clock/reset staging and protocol bring-up.",
                "visibility": "global",
                "audience_roles": ["firmware", "rtl", "test_debug", "build", "architecture"],
                "interface_tags": ["clock", "reset", "boot", "protocol", "simulation"],
            }
        ]

        for i in range(BLOCK_COUNT):
            block_node_id = f"block.ip_{i:03d}"
            alias = f"IP_{i:03d}"
            seed_mutations.extend(
                [
                    {
                        "op": "upsert_node",
                        "node_id": block_node_id,
                        "node_type": "block",
                        "canonical_name": f"ip_{i:03d}",
                        "summary": f"IP block {i:03d} used in simulation and palladium validation paths.",
                        "visibility": "shared",
                        "finder_role": "rtl",
                        "audience_roles": ["rtl", "firmware", "test_debug", "architecture"],
                        "interface_tags": ["protocol", "simulation", "palladium"],
                    },
                    {"op": "add_alias", "node_id": block_node_id, "alias": alias, "alias_kind": "name"},
                    {
                        "op": "add_edge",
                        "edge_id": f"edge.{block_node_id}.part_of.soc_boot",
                        "from_node_id": block_node_id,
                        "relation": "part_of",
                        "to_node_id": "flow.soc_boot",
                        "summary": "Block participates in SoC boot sequencing.",
                        "visibility": "shared",
                        "audience_roles": ["rtl", "firmware", "test_debug", "architecture"],
                        "interface_tags": ["boot", "protocol"],
                    },
                ]
            )

        for i in range(REGISTER_COUNT):
            block_idx = i % BLOCK_COUNT
            block_node_id = f"block.ip_{block_idx:03d}"
            reg_node_id = f"register.ip_{block_idx:03d}.reg_{i:03d}"
            seed_mutations.extend(
                [
                    {
                        "op": "upsert_node",
                        "node_id": reg_node_id,
                        "node_type": "register",
                        "canonical_name": f"ip_{block_idx:03d}_reg_{i:03d}",
                        "summary": "Register controls protocol timing and simulation seed behavior.",
                        "visibility": "shared",
                        "finder_role": "firmware",
                        "audience_roles": ["firmware", "rtl", "test_debug", "architecture"],
                        "interface_tags": ["register", "timing", "seed", "simulation"],
                    },
                    {
                        "op": "add_edge",
                        "edge_id": f"edge.{reg_node_id}.belongs.{block_node_id}",
                        "from_node_id": reg_node_id,
                        "relation": "belongs_to",
                        "to_node_id": block_node_id,
                        "summary": "Register belongs to the block.",
                        "visibility": "shared",
                        "audience_roles": ["firmware", "rtl", "test_debug", "architecture"],
                        "interface_tags": ["register", "protocol"],
                    },
                    {
                        "op": "upsert_claim",
                        "claim_id": f"claim.{reg_node_id}.timing",
                        "target_node_id": reg_node_id,
                        "claim_text": "Register participates in clock/reset and protocol timing behavior.",
                        "classification": "Fact",
                        "confidence": "medium",
                        "visibility": "shared",
                        "audience_roles": ["firmware", "rtl", "test_debug", "architecture"],
                        "interface_tags": ["clock", "reset", "timing", "protocol"],
                        "chunk_ids": [chunk_id],
                    },
                ]
            )

        prop_seed, propose_ms = timed_call(
            lambda: b.propose_update("stress seed graph", seed_mutations, created_by="stress_test")
        )
        assert propose_ms < CURATOR_STEP_MAX_MS, propose_ms
        _, commit_ms = timed_call(lambda: b.commit_update(prop_seed["proposal_id"]))
        assert commit_ms < CURATOR_STEP_MAX_MS, commit_ms

        search_times: list[float] = []
        for query in [
            "clock reset protocol",
            "simulation seed behavior",
            "palladium validation block",
            "soc boot register timing",
        ]:
            result, elapsed_ms = timed_call(lambda q=query: b.search(q, role="firmware", intent="debug", limit=25))
            search_times.append(elapsed_ms)
            assert result["results"], query
            assert elapsed_ms < SEARCH_MAX_MS, (query, elapsed_ms)

        ctx_local, local_ms = timed_call(
            lambda: b.context(anchors=["IP_010"], role="firmware", mode="local", budget="large")
        )
        ctx_global, global_ms = timed_call(
            lambda: b.context(
                anchors=["IP_010"],
                query="clock reset protocol simulation palladium",
                role="firmware",
                mode="global",
                budget="large",
            )
        )
        ctx_bridge, bridge_ms = timed_call(
            lambda: b.context(anchors=["IP_010", "IP_040"], role="firmware", mode="bridge", budget="large")
        )
        ctx_hybrid, hybrid_ms = timed_call(
            lambda: b.context(
                anchors=["IP_010", "IP_040"],
                query="clock reset protocol simulation palladium",
                role="test_debug",
                mode="hybrid",
                budget="large",
            )
        )

        context_times: dict[str, float] = {}
        for name, packet, elapsed_ms, limit_ms in [
            ("local", ctx_local, local_ms, CONTEXT_LOCAL_MAX_MS),
            ("global", ctx_global, global_ms, CONTEXT_GLOBAL_MAX_MS),
            ("bridge", ctx_bridge, bridge_ms, CONTEXT_BRIDGE_MAX_MS),
            ("hybrid", ctx_hybrid, hybrid_ms, CONTEXT_HYBRID_MAX_MS),
        ]:
            context_times[name] = elapsed_ms
            assert elapsed_ms < limit_ms, (name, elapsed_ms, limit_ms)
            assert len(packet["markdown"]) < CONTEXT_MAX_CHARS, (name, len(packet["markdown"]))
            assert len(packet["active_claims"]) <= ACTIVE_CLAIMS_MAX, (name, len(packet["active_claims"]))
            assert len(packet["related_edges"]) <= RELATED_EDGES_MAX, (name, len(packet["related_edges"]))

        hybrid_to_global_ratio = hybrid_ms / max(global_ms, 1.0)
        assert hybrid_to_global_ratio < HYBRID_TO_GLOBAL_MAX_RATIO, hybrid_to_global_ratio

        curator_cycle_times: list[float] = []
        curator_step_times: dict[str, list[float]] = {
            "proposal": [],
            "validate_pre": [],
            "commit": [],
            "render": [],
            "validate_post": [],
        }
        for i in range(3):
            claim_id = f"claim.curator_cycle_{i}"
            mutation = [
                {
                    "op": "upsert_claim",
                    "claim_id": claim_id,
                    "target_node_id": "flow.soc_boot",
                    "claim_text": f"Curator cycle {i} keeps boot sequencing evidence traceable.",
                    "classification": "Hypothesis",
                    "confidence": "low",
                    "chunk_ids": [chunk_id],
                }
            ]
            proposal, proposal_ms = timed_call(
                lambda m=mutation, idx=i: b.propose_update(f"curator cycle {idx}", m, created_by="stress_test")
            )
            _, validate1_ms = timed_call(b.validate)
            _, commit_cycle_ms = timed_call(lambda p=proposal: b.commit_update(p["proposal_id"]))
            _, render_ms = timed_call(lambda: b.render_docs("docs/rendered"))
            _, validate2_ms = timed_call(b.validate)
            cycle_total_ms = proposal_ms + validate1_ms + commit_cycle_ms + render_ms + validate2_ms
            curator_cycle_times.append(cycle_total_ms)
            for step_name, elapsed_ms in [
                ("proposal", proposal_ms),
                ("validate_pre", validate1_ms),
                ("commit", commit_cycle_ms),
                ("render", render_ms),
                ("validate_post", validate2_ms),
            ]:
                curator_step_times[step_name].append(elapsed_ms)
                assert elapsed_ms < CURATOR_STEP_MAX_MS, (i, step_name, elapsed_ms)
            assert cycle_total_ms < CURATOR_CYCLE_MAX_MS, (i, cycle_total_ms)

        assert b.validate()["ok"] is True
        assert b.conn.execute("SELECT COUNT(*) AS n FROM nodes WHERE status='active'").fetchone()["n"] >= 300
        assert b.conn.execute("SELECT COUNT(*) AS n FROM claims WHERE status='active'").fetchone()["n"] >= 220

        summary = {
            "search_ms": {
                "min": round(min(search_times), 2),
                "max": round(max(search_times), 2),
                "avg": round(statistics.mean(search_times), 2),
            },
            "context_ms": {k: round(v, 2) for k, v in context_times.items()},
            "hybrid_to_global_ratio": round(hybrid_to_global_ratio, 2),
            "curator_cycle_ms": {
                "min": round(min(curator_cycle_times), 2),
                "max": round(max(curator_cycle_times), 2),
                "avg": round(statistics.mean(curator_cycle_times), 2),
            },
            "curator_step_max_ms": {k: round(max(v), 2) for k, v in curator_step_times.items()},
            "context_size": {
                "local_chars": len(ctx_local["markdown"]),
                "global_chars": len(ctx_global["markdown"]),
                "bridge_chars": len(ctx_bridge["markdown"]),
                "hybrid_chars": len(ctx_hybrid["markdown"]),
            },
            "anchor_counts": {
                "local": len(ctx_local["selected_anchors"]),
                "global": len(ctx_global["selected_anchors"]),
                "bridge": len(ctx_bridge["selected_anchors"]),
                "hybrid": len(ctx_hybrid["selected_anchors"]),
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("STRESS_FETCH_TIMING_CONTEXT_CURATOR_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
