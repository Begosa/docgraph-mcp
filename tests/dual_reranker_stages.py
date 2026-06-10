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
from docgraph_mcp.models import (  # noqa: E402
    HttpChatRerankerProvider,
    ProviderState,
    parse_rerank_tool_call,
    select_first_available_model,
)


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


class FakeCrossEncoderReranker:
    def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        out: list[float] = []
        for candidate in candidates:
            lower = candidate.lower()
            score = 0.1
            if "pixel" in lower:
                score += 1.0
            if "gap" in lower:
                score += 0.5
            if "noise" in lower:
                score += 0.8
            out.append(score)
        return out


class FakeLlmReranker:
    def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        out: list[float] = []
        for candidate in candidates:
            lower = candidate.lower()
            out.append(0.95 if "pixel" in lower and "gap" in lower else 0.05)
        return out


class InvertedLlmReranker:
    def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        out: list[float] = []
        for candidate in candidates:
            lower = candidate.lower()
            out.append(0.1 if "pixel" in lower and "gap" in lower else 0.9)
        return out


class FailingLlmReranker:
    def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        raise RuntimeError("llm unavailable")


def seed_graph(b: DocGraphBackend) -> None:
    prop = b.propose_update(
        "dual reranker seed",
        [
            {
                "op": "upsert_node",
                "node_id": "register.pixel_gap",
                "node_type": "register",
                "canonical_name": "pixel_gap",
                "summary": "Pixel gap semantic target.",
                "visibility": "global",
                "audience_roles": ["firmware", "architecture"],
            },
            {
                "op": "upsert_node",
                "node_id": "concept.noise",
                "node_type": "concept",
                "canonical_name": "noise_bucket",
                "summary": "Noise unrelated candidate.",
                "visibility": "global",
                "audience_roles": ["firmware", "architecture"],
            },
        ],
        created_by="dual_reranker_test",
    )
    b.commit_update(prop["proposal_id"])


def write_config(repo: Path, extra_yaml: str) -> None:
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
  llm_reranker:
    enabled: true
    trigger: always
    top_k_input: 5
    top_k_output: 5
"""
        + extra_yaml,
        encoding="utf-8",
    )


def main() -> None:
    assert select_first_available_model(["b", "a", "c"], ["x", "a", "b"]) == "a"
    assert parse_rerank_tool_call(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_rerank_scores",
                        "arguments": '{"scores":[0.95,0.05]}',
                    }
                }
            ]
        },
        2,
    ) == [0.95, 0.05]

    provider = HttpChatRerankerProvider({"base_url": "https://example.com"})
    body = provider._chat_request_body(model="demo-model", prompt="task body", candidate_count=2)
    assert body["messages"][0] == {"role": "system", "content": ""}
    assert body["messages"][1]["role"] == "user"

    tmp = Path(tempfile.mkdtemp(prefix="docgraph_dual_reranker_"))
    old_embed = backend_mod.make_embedding_provider
    old_rerank = backend_mod.make_reranker_provider
    old_llm = backend_mod.make_llm_reranker_provider
    try:
        repo = tmp / "repo"
        repo.mkdir()
        write_config(repo, "")
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        seed_graph(b)

        llm_calls = {"count": 0}

        backend_mod.make_embedding_provider = lambda config: (FakeEmbeddingProvider(), ProviderState(True, True))  # noqa: ARG005
        backend_mod.make_reranker_provider = lambda config: (FakeCrossEncoderReranker(), ProviderState(True, True))  # noqa: ARG005

        def llm_factory(config: dict) -> tuple[FakeLlmReranker, ProviderState]:  # noqa: ARG001
            llm_calls["count"] += 1
            return FakeLlmReranker(), ProviderState(True, True)

        backend_mod.make_llm_reranker_provider = llm_factory

        semantic = b._semantic_context(query="pixel gap", role="firmware", lexical_anchor_count=1)
        assert semantic["enabled"] is True and semantic["available"] is True, semantic
        assert semantic["results"], semantic
        assert semantic["results"][0]["id"] == "register.pixel_gap", semantic["results"]
        assert "reranker_score" in semantic["results"][0], semantic["results"][0]
        assert "llm_reranker_score" in semantic["results"][0], semantic["results"][0]
        assert llm_calls["count"] >= 1, llm_calls
        assert semantic.get("rerank_trace", {}).get("llm", {}).get("applied") is True, semantic.get("rerank_trace")
        llm_scored = semantic.get("rerank_trace", {}).get("llm", {}).get("scored") or []
        assert llm_scored, llm_scored
        assert all("llm_reranker_score" in item and "final_score" in item for item in llm_scored), llm_scored

        write_config(
            repo,
            """
    trigger: always
    min_relevance_score: 0.3
""",
        )
        backend_mod.make_llm_reranker_provider = lambda config: (InvertedLlmReranker(), ProviderState(True, True))  # noqa: ARG005
        b_penalty = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        semantic_penalty = b_penalty._semantic_context(query="pixel gap", role="firmware", lexical_anchor_count=1)
        assert semantic_penalty["results"], semantic_penalty
        assert semantic_penalty["results"][0]["id"] == "concept.noise", semantic_penalty["results"]
        llm_penalty_trace = semantic_penalty.get("rerank_trace", {}).get("llm", {}).get("scored") or []
        assert any(item.get("below_min_relevance") is True for item in llm_penalty_trace), llm_penalty_trace

        write_config(
            repo,
            """
    trigger: rescue_only
""",
        )
        b2 = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        llm_calls["count"] = 0
        backend_mod.make_llm_reranker_provider = llm_factory
        semantic_skip = b2._semantic_context(query="pixel gap", role="firmware", lexical_anchor_count=5)
        assert semantic_skip.get("rerank_trace", {}).get("llm", {}).get("triggered") is False, semantic_skip.get("rerank_trace")
        assert llm_calls["count"] == 0, llm_calls

        backend_mod.make_llm_reranker_provider = lambda config: (FailingLlmReranker(), ProviderState(True, True))  # noqa: ARG005
        write_config(
            repo,
            """
    trigger: always
""",
        )
        b3 = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        semantic_fail = b3._semantic_context(query="pixel gap", role="firmware", lexical_anchor_count=1)
        assert semantic_fail["results"], semantic_fail
        assert semantic_fail.get("rerank_trace", {}).get("llm", {}).get("applied") is False, semantic_fail.get("rerank_trace")
        assert "reranker_score" in semantic_fail["results"][0], semantic_fail["results"][0]

        print("DUAL_RERANKER_STAGES_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        backend_mod.make_reranker_provider = old_rerank
        backend_mod.make_llm_reranker_provider = old_llm
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
