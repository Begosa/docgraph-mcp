#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docgraph_mcp import DocGraphBackend  # noqa: E402


def expect_value_error(fn: Callable[[], Any], needle: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), exc
    else:
        raise AssertionError(f"expected ValueError containing {needle!r}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_ingest_safeguards_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        (repo / "Firmware").mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
source_handling:
  max_file_ingest_bytes: 20
  max_inline_content_bytes: 10
  reject_inline_content_for_repo_files: true
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )

        small_source = repo / "Firmware" / "small.c"
        small_source.write_text("int v;\n", encoding="utf-8")

        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)

        expect_value_error(
            lambda: b.ingest_source("code_file", "Firmware/small.c", content="int v;\n"),
            "refusing inline content for repository-local source",
        )

        archive = b.ingest_source("historical_doc", "archive://old-docs/short.md", content="short")
        assert archive["status"] == "ingested", archive
        assert archive["content_bytes"] == 5, archive

        expect_value_error(
            lambda: b.ingest_source("historical_doc", "archive://old-docs/too-large.md", content="01234567890"),
            "max_inline_content_bytes",
        )

        big_source = repo / "Firmware" / "big.c"
        big_source.write_text("012345678901234567890", encoding="utf-8")
        expect_value_error(
            lambda: b.ingest_source("code_file", "Firmware/big.c"),
            "max_file_ingest_bytes",
        )

        stale_source = repo / "Firmware" / "stale.c"
        stale_source.write_text("int a;\n", encoding="utf-8")
        first = b.ingest_source("code_file", "Firmware/stale.c")
        assert first["status"] == "ingested", first
        stale_source.write_text("int b;\n", encoding="utf-8")

        scan = b.stale_scan(auto_ingest=True)
        assert len(scan["changed"]) == 1, scan
        assert len(scan["auto_ingested"]) == 1, scan
        assert scan["auto_ingested"][0]["status"] == "ingested", scan
        assert scan["auto_ingested"][0]["superseded_episode_id"], scan

        unchanged = b.ingest_source("code_file", "Firmware/stale.c")
        assert unchanged["status"] == "unchanged", unchanged
        assert unchanged["content_bytes"] == len("int b;\n".encode("utf-8")), unchanged

        print("SOURCE_INGEST_SAFEGUARDS_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
