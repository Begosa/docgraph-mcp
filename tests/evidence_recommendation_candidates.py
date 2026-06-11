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
    tmp = Path(tempfile.mkdtemp(prefix="docgraph_evidence_candidates_"))
    try:
        repo = tmp / "repo"
        repo.mkdir()
        src_dir = repo / "rtl"
        src_dir.mkdir()
        lines = []
        for i in range(1, 131):
            if i == 76:
                lines.append("// GFCLKMUX requires both CLK0 and CLK1 toggling when SEL changes")
            elif i == 77:
                lines.append("assign BUSY = select_clka ^ enable_clkb;")
            else:
                lines.append(f"wire filler_{i};")
        (src_dir / "mux.v").write_text("\n".join(lines) + "\n", encoding="utf-8")
        db = repo / "docs" / "docgraph.sqlite"
        b = DocGraphBackend(db, root=repo)

        no_hint = b.ingest_source("rtl_file", "rtl/mux.v")
        assert no_hint["recommended_evidence_candidates"] == [], no_hint
        assert no_hint["recommendation_warning"] is None, no_hint

        with_lines = b.ingest_source(
            "rtl_file",
            "rtl/mux.v",
            evidence_lines=[{"start": 76, "end": 77}],
            recommend_limit=2,
        )
        assert with_lines["status"] == "unchanged", with_lines
        assert with_lines["recommended_evidence_candidates"], with_lines
        top = with_lines["recommended_evidence_candidates"][0]
        assert top["locator"] == "lines 56-115", top
        assert top["match_type"].startswith("line_overlap"), top
        assert "Candidates are retrieval hints" in with_lines["recommendation_warning"], with_lines

        with_hint = b.ingest_source(
            "rtl_file",
            "rtl/mux.v",
            evidence_hint="GFCLKMUX BUSY SEL CLK0 CLK1 toggling",
            recommend_limit=1,
        )
        assert with_hint["recommended_evidence_candidates"], with_hint
        top_hint = with_hint["recommended_evidence_candidates"][0]
        assert top_hint["locator"] == "lines 56-115", top_hint
        assert top_hint["match_type"] in {"text_match", "line_overlap+text_match"}, top_hint
        assert {"gfclkmux", "busy", "sel"}.intersection(set(top_hint["matched_terms"])), top_hint

        print("EVIDENCE_RECOMMENDATION_CANDIDATES_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
