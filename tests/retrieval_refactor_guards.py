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
    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            if "pixel" in lower or "gap" in lower:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.1, 0.9] if is_query else [0.0, 1.0])
        return vectors


class FakeRerankerProvider:
    def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        out: list[float] = []
        for candidate in candidates:
            lower = candidate.lower()
            score = 0.0
            if "pixel" in lower:
                score += 10.0
            if "gap" in lower:
                score += 5.0
            out.append(score)
        return out


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_refactor_guards_"))
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
  reranker:
    enabled: true
    top_k_input: 20
    top_k_output: 5
""",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)

        seed = b.propose_update(
            "seed retrieval candidates",
            [
                {
                    "op": "upsert_node",
                    "node_id": "register.pixel_gap",
                    "node_type": "register",
                    "canonical_name": "pixel_gap",
                    "summary": "Pixel gap semantic target.",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                }
            ],
            created_by="refactor_guard_test",
        )
        b.commit_update(seed["proposal_id"])

        calls = {"embed_factory": 0, "reranker_factory": 0}

        def embed_factory(config: dict) -> tuple[FakeEmbeddingProvider, ProviderState]:  # noqa: ARG001
            calls["embed_factory"] += 1
            return FakeEmbeddingProvider(), ProviderState(True, True)

        def rerank_factory(config: dict) -> tuple[FakeRerankerProvider, ProviderState]:  # noqa: ARG001
            calls["reranker_factory"] += 1
            return FakeRerankerProvider(), ProviderState(True, True)

        backend_mod.make_embedding_provider = embed_factory
        backend_mod.make_reranker_provider = rerank_factory

        semantic = b._semantic_context(query="pixel gap", role="firmware")
        assert semantic["enabled"] is True and semantic["available"] is True, semantic
        assert semantic["results"], semantic
        assert semantic["results"][0]["id"] == "register.pixel_gap", semantic["results"]
        assert "reranker_score" in semantic["results"][0], semantic["results"][0]
        assert calls["embed_factory"] >= 1, calls
        assert calls["reranker_factory"] >= 1, calls

        marker: dict[str, object] = {"called": False}
        original_context_markdown = b._retrieval_flow.context_markdown

        def fake_context_markdown(packet: dict[str, object]) -> str:
            marker["called"] = True
            marker["packet"] = packet
            return "MARKDOWN_SENTINEL\n"

        b._retrieval_flow.context_markdown = fake_context_markdown  # type: ignore[assignment]
        try:
            probe = {"probe": "value"}
            rendered = b._context_markdown(probe)
            assert rendered == "MARKDOWN_SENTINEL\n", rendered
            assert marker["called"] is True, marker
            assert marker["packet"] == probe, marker
        finally:
            b._retrieval_flow.context_markdown = original_context_markdown  # type: ignore[assignment]

        print("RETRIEVAL_REFACTOR_GUARDS_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        backend_mod.make_reranker_provider = old_rerank
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
