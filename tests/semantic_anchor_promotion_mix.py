#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import docgraph_mcp.backend as backend_mod  # noqa: E402
from docgraph_mcp import DocGraphBackend  # noqa: E402
from docgraph_mcp.models import ProviderState  # noqa: E402


class FakeSemanticProvider:
    model_name = "fake-semantic-promotion-v1"

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            if is_query:
                if any(token in lower for token in ("receiver", "sink", "underflow", "stride")):
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
                continue
            if "mipi_rx" in lower:
                vectors.append([1.0, 0.0])
            elif "display_dma" in lower or "dma_stride" in lower:
                vectors.append([0.8, 0.2])
            elif "audio" in lower:
                # Deliberately mimic a semantically similar but disconnected subsystem.
                vectors.append([0.98, 0.02])
            else:
                vectors.append([0.2, 0.8])
        return vectors


def precision_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    window = ids[:k]
    if not window:
        return 0.0
    return sum(1 for item in window if item in relevant) / float(len(window))


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_semantic_anchor_promotion_"))
    old_embed = backend_mod.make_embedding_provider
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval:
  budgets:
    small:
      nodes: 6
      claims: 6
      edges: 8
      evidence: 2
  modes:
    mix:
      semantic_anchor_promotion:
        enabled: false
        top_semantic_results: 12
        max_promoted_anchors: 2
        min_lexical_anchors: 1
        min_score: 0
        relative_delta: 40
        lexical_weight: 1.0
        semantic_weight: 1.2
        rrf_k: 60
        require_graph_coherence: true
        coherence_min_lexical_anchors: 2
        coherence_max_depth: 2
        coherence_max_depth_by_budget: {small: 2, medium: 2, large: 3}
retrieval_models:
  embeddings:
    enabled: true
    top_k: 12
    max_in_memory_items: 100
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )
        (repo / "evidence.md").write_text(
            "\n".join(
                [
                    "display dma sink underflow appears when stride is too tight.",
                    "audio coefficient controls equalizer saturation scenarios.",
                    "mipi lane ingest endpoint exists for capture front-end.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        backend_mod.make_embedding_provider = lambda config: (FakeSemanticProvider(), ProviderState(True, True))  # noqa: ARG005
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = b.ingest_source("doc_file", "evidence.md")
        chunk_id = b.conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_id=? ORDER BY rowid LIMIT 1",
            (ingest["source_id"],),
        ).fetchone()["chunk_id"]

        seed = b.propose_update(
            "seed semantic anchor promotion test graph",
            [
                {"op": "upsert_node", "node_id": "block.display_dma", "node_type": "block", "canonical_name": "display_dma", "summary": "Display DMA sink underflow path.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_node", "node_id": "register.dma_stride", "node_type": "register", "canonical_name": "dma_stride", "summary": "Stride register affects underflow margin.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_node", "node_id": "flow.capture_pipeline", "node_type": "flow", "canonical_name": "capture_pipeline", "summary": "Capture flow around DMA timing.", "visibility": "global", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_node", "node_id": "interface.capture_input", "node_type": "interface", "canonical_name": "capture_input", "summary": "Capture ingress interface.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_node", "node_id": "protocol_endpoint.mipi_rx", "node_type": "protocol_endpoint", "canonical_name": "mipi_rx", "summary": "Lane ingest endpoint at front-end.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_node", "node_id": "register.audio_sink_gain", "node_type": "register", "canonical_name": "audio_gain", "summary": "Audio equalizer coefficient.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "add_edge", "edge_id": "edge.dma_stride.configures.display_dma", "from_node_id": "register.dma_stride", "relation": "configures", "to_node_id": "block.display_dma", "summary": "stride configures DMA bursts.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "add_edge", "edge_id": "edge.display_dma.part.capture", "from_node_id": "block.display_dma", "relation": "part_of", "to_node_id": "flow.capture_pipeline", "summary": "DMA is part of capture flow.", "visibility": "global", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "add_edge", "edge_id": "edge.mipi.produces.capture_input", "from_node_id": "protocol_endpoint.mipi_rx", "relation": "produces", "to_node_id": "interface.capture_input", "summary": "Front-end endpoint produces capture ingress.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "add_edge", "edge_id": "edge.capture_input.produces.display_dma", "from_node_id": "interface.capture_input", "relation": "produces", "to_node_id": "block.display_dma", "summary": "Capture ingress supplies DMA.", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"]},
                {"op": "upsert_claim", "claim_id": "claim.dma_stride.underflow", "target_node_id": "register.dma_stride", "claim_text": "Tight stride can trigger sink underflow.", "classification": "Fact", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"], "chunk_ids": [chunk_id]},
                {"op": "upsert_claim", "claim_id": "claim.mipi.frontend", "target_node_id": "protocol_endpoint.mipi_rx", "claim_text": "MIPI RX is the capture front-end endpoint.", "classification": "Fact", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"], "chunk_ids": [chunk_id]},
                {"op": "upsert_claim", "claim_id": "claim.audio.sink", "target_node_id": "register.audio_sink_gain", "claim_text": "Audio coefficient impacts equalizer behavior.", "classification": "Fact", "visibility": "shared", "audience_roles": ["rtl", "firmware", "test_debug"], "chunk_ids": [chunk_id]},
            ],
            created_by="semantic_anchor_promotion_test",
        )
        b.commit_update(seed["proposal_id"])

        query = "receiver sink underflow stride issue"
        relevant = {"block.display_dma", "register.dma_stride", "flow.capture_pipeline", "interface.capture_input", "protocol_endpoint.mipi_rx"}

        baseline = b.context(query=query, role="rtl", mode="mix", budget="small")
        baseline_ids = [node["node_id"] for node in baseline["selected_anchors"]]
        baseline_p4 = precision_at_k(baseline_ids, relevant, 4)

        b.config["retrieval"]["modes"]["mix"]["semantic_anchor_promotion"]["enabled"] = True
        promoted = b.context(query=query, role="rtl", mode="mix", budget="small")
        promoted_ids = [node["node_id"] for node in promoted["selected_anchors"]]
        promoted_p4 = precision_at_k(promoted_ids, relevant, 4)

        assert "protocol_endpoint.mipi_rx" not in baseline_ids, baseline_ids
        assert "protocol_endpoint.mipi_rx" in promoted_ids, promoted_ids
        assert len(set(promoted_ids) & relevant) > len(set(baseline_ids) & relevant), (baseline_ids, promoted_ids)
        trace_row = b.conn.execute("SELECT trace_json FROM retrieval_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        trace = json.loads(trace_row["trace_json"])
        assert trace["semantic_promotion"]["promoted_anchors"], trace
        assert "protocol_endpoint.mipi_rx" in trace["semantic_promotion"]["promoted_anchors"], trace
        assert "register.audio_sink_gain" not in promoted_ids, promoted_ids
        coherence = trace["semantic_promotion"]["coherence_gate"]
        assert coherence["applied"] is True, coherence
        assert "register.audio_sink_gain" in coherence["rejected_semantic_only"], coherence
        accepted = {row["node_id"]: row for row in coherence["connections"]}
        assert accepted["protocol_endpoint.mipi_rx"]["distance"] <= 2, accepted

        b.config["retrieval"]["modes"]["mix"]["semantic_anchor_promotion"]["coherence_max_depth_by_budget"] = {
            "small": 1,
            "medium": 2,
            "large": 3,
        }
        shallow = b.context(query=query, role="rtl", mode="mix", budget="small")
        shallow_trace_row = b.conn.execute("SELECT trace_json FROM retrieval_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        shallow_trace = json.loads(shallow_trace_row["trace_json"])
        shallow_coherence = shallow_trace["semantic_promotion"]["coherence_gate"]
        assert shallow_coherence["max_depth"] == 1, shallow_coherence
        assert shallow_coherence["budget"] == "small", shallow_coherence

        broad = b.context(query=query, role="rtl", mode="mix", budget="large")
        broad_ids = [node["node_id"] for node in broad["selected_anchors"]]
        broad_trace_row = b.conn.execute("SELECT trace_json FROM retrieval_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        broad_trace = json.loads(broad_trace_row["trace_json"])
        broad_coherence = broad_trace["semantic_promotion"]["coherence_gate"]
        assert broad_coherence["max_depth"] == 3, broad_coherence
        assert broad_coherence["budget"] == "large", broad_coherence
        assert "protocol_endpoint.mipi_rx" in broad_ids, broad_ids

        b.config["retrieval"]["modes"]["mix"]["semantic_anchor_promotion"]["require_graph_coherence"] = False
        ungated = b.context(query=query, role="rtl", mode="mix", budget="small")
        ungated_ids = [node["node_id"] for node in ungated["selected_anchors"]]
        ungated_trace_row = b.conn.execute("SELECT trace_json FROM retrieval_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        ungated_trace = json.loads(ungated_trace_row["trace_json"])
        ungated_coherence = ungated_trace["semantic_promotion"]["coherence_gate"]
        assert ungated_coherence["enabled"] is False, ungated_coherence
        assert ungated_coherence["applied"] is False, ungated_coherence
        assert "register.audio_sink_gain" in ungated_ids, ungated_ids

        report = {
            "query": query,
            "baseline_anchor_ids": baseline_ids,
            "promoted_anchor_ids": promoted_ids,
            "ungated_anchor_ids": ungated_ids,
            "baseline_precision_at_4": round(baseline_p4, 3),
            "promoted_precision_at_4": round(promoted_p4, 3),
            "semantic_top": [f"{item.get('kind')}:{item.get('id')}" for item in promoted["semantic_candidates"]["results"][:8]],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        print("SEMANTIC_ANCHOR_PROMOTION_MIX_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
