#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import docgraph_mcp.backend as backend_mod  # noqa: E402
from docgraph_mcp import DocGraphBackend  # noqa: E402
from docgraph_mcp.models import ProviderState  # noqa: E402


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:  # noqa: ARG002
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            if "pixel" in lower or "gap" in lower:
                vectors.append([1.0, 0.0])
            elif "noise" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


class FakeRerankerProvider:
    def score(self, query: str, candidates: list[str]) -> list[float]:
        scores = []
        for cand in candidates:
            lower = cand.lower()
            score = 0.0
            if "pixel" in lower:
                score += 10.0
            if "gap" in lower:
                score += 5.0
            if "noise" in lower:
                score -= 5.0
            scores.append(score)
        return scores


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_mock_semantic_"))
    old_embed = backend_mod.make_embedding_provider
    old_rerank = backend_mod.make_reranker_provider
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval_models:
  embeddings:
    enabled: true
    top_k: 5
    max_in_memory_items: 50
  reranker:
    enabled: true
    top_k_input: 20
    top_k_output: 5
""",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        backend_mod.make_embedding_provider = lambda config: (FakeEmbeddingProvider(), ProviderState(True, True))
        backend_mod.make_reranker_provider = lambda config: (FakeRerankerProvider(), ProviderState(True, True))
        prop = b.propose_update(
            "mock semantic seed",
            [
                {"op": "upsert_node", "node_id": "register.pixel_gap", "node_type": "register", "canonical_name": "pixel_gap", "summary": "Pixel gap register semantic target.", "visibility": "global", "audience_roles": ["firmware", "architecture"]},
                {"op": "upsert_node", "node_id": "concept.noise", "node_type": "concept", "canonical_name": "noise_bucket", "summary": "Noise unrelated candidate.", "visibility": "global", "audience_roles": ["firmware", "architecture"]},
            ],
            created_by="mock_semantic_test",
        )
        b.commit_update(prop["proposal_id"])
        result = b.context(query="pixel gap", role="firmware", mode="mix", budget="small")
        sem = result["semantic_candidates"]
        assert sem["enabled"] is True and sem["available"] is True, sem
        assert sem["results"], sem
        assert sem["results"][0]["id"] == "register.pixel_gap", sem["results"]
        assert "reranker_score" in sem["results"][0], sem["results"][0]
        print("MOCK_SEMANTIC_RERANKER_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        backend_mod.make_reranker_provider = old_rerank
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
