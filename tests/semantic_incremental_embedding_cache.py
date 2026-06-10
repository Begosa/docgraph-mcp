#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import docgraph_mcp.backend as backend_mod  # noqa: E402
from docgraph_mcp import DocGraphBackend  # noqa: E402
from docgraph_mcp.models import ProviderState  # noqa: E402


class FakeCachingEmbeddingProvider:
    model_name = "fake-cache-model-v1"

    def __init__(self) -> None:
        self.query_calls = 0
        self.candidate_calls = 0
        self.candidate_batch_sizes: list[int] = []

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if is_query:
            self.query_calls += 1
        else:
            self.candidate_calls += 1
            self.candidate_batch_sizes.append(len(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([float(digest[0]) / 255.0, float(digest[1]) / 255.0])
        return vectors


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_semantic_cache_"))
    old_embed = backend_mod.make_embedding_provider
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval_models:
  embeddings:
    enabled: true
    incremental_cache_enabled: true
    top_k: 5
    max_in_memory_items: 50
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )
        provider = FakeCachingEmbeddingProvider()
        backend_mod.make_embedding_provider = lambda config: (provider, ProviderState(True, True))  # noqa: ARG005
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)

        prop = b.propose_update(
            "seed cache candidates",
            [
                {
                    "op": "upsert_node",
                    "node_id": "register.pixel_gap",
                    "node_type": "register",
                    "canonical_name": "pixel_gap",
                    "summary": "Pixel gap timing register.",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                },
                {
                    "op": "upsert_node",
                    "node_id": "feature.line_timing",
                    "node_type": "feature",
                    "canonical_name": "line_timing",
                    "summary": "Line timing architecture feature.",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                },
            ],
            created_by="semantic_cache_test",
        )
        b.commit_update(prop["proposal_id"])

        first = b.context(query="pixel gap timing", role="firmware", mode="mix", budget="small")
        assert first["semantic_candidates"]["available"] is True, first["semantic_candidates"]
        assert provider.candidate_calls == 1, provider.candidate_calls
        assert provider.candidate_batch_sizes[-1] >= 2, provider.candidate_batch_sizes

        second = b.context(query="pixel gap timing", role="firmware", mode="mix", budget="small")
        assert second["semantic_candidates"]["available"] is True, second["semantic_candidates"]
        assert provider.candidate_calls == 1, provider.candidate_calls

        update = b.propose_update(
            "change one candidate text",
            [
                {
                    "op": "upsert_node",
                    "node_id": "feature.line_timing",
                    "node_type": "feature",
                    "canonical_name": "line_timing",
                    "summary": "Line timing feature updated with frame sync details.",
                    "visibility": "global",
                    "audience_roles": ["firmware", "architecture"],
                }
            ],
            created_by="semantic_cache_test",
        )
        b.commit_update(update["proposal_id"])

        third = b.context(query="pixel gap timing", role="firmware", mode="mix", budget="small")
        assert third["semantic_candidates"]["available"] is True, third["semantic_candidates"]
        assert provider.candidate_calls == 2, provider.candidate_calls
        assert provider.candidate_batch_sizes[-1] == 1, provider.candidate_batch_sizes

        row = b.conn.execute(
            "SELECT COUNT(*) AS n FROM embedding_items WHERE embedding_model=?",
            (provider.model_name,),
        ).fetchone()
        assert int(row["n"]) >= 2, row

        print("SEMANTIC_INCREMENTAL_EMBEDDING_CACHE_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
