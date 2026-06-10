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


class FakeProvider:
    pass


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_model_paths_"))
    old_embed = backend_mod.make_embedding_provider
    old_rerank = backend_mod.make_reranker_provider
    try:
        repo = tmp / "repo"
        (repo / "models" / "embedding-local").mkdir(parents=True)
        (repo / "models" / "reranker-local").mkdir(parents=True)
        repo.mkdir(exist_ok=True)
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval_models:
  embeddings:
    enabled: true
    provider: sentence_transformers
    model: models/embedding-local
    preload_on_boot: false
  reranker:
    enabled: true
    provider: sentence_transformers_cross_encoder
    model: models/reranker-local
    preload_on_boot: false
""",
            encoding="utf-8",
        )
        b = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        captured: dict[str, str] = {}

        def embed_factory(config: dict):
            captured["embedding_model"] = config["retrieval_models"]["embeddings"]["model"]
            return FakeProvider(), ProviderState(True, True)

        def rerank_factory(config: dict):
            captured["reranker_model"] = config["retrieval_models"]["reranker"]["model"]
            return FakeProvider(), ProviderState(True, True)

        backend_mod.make_embedding_provider = embed_factory
        backend_mod.make_reranker_provider = rerank_factory
        b._make_embedding_provider()
        b._make_reranker_provider()
        assert captured["embedding_model"] == str((repo / "models" / "embedding-local").resolve()), captured
        assert captured["reranker_model"] == str((repo / "models" / "reranker-local").resolve()), captured
        print("LOCAL_MODEL_PATH_RESOLUTION_OK")
    finally:
        backend_mod.make_embedding_provider = old_embed
        backend_mod.make_reranker_provider = old_rerank
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
