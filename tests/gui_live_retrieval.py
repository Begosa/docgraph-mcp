#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from docgraph_mcp import DocGraphBackend  # noqa: E402
import docgraph_gui_pyqt6 as gui  # noqa: E402


def main() -> None:
    gui.DEFAULT_DOCGRAPH_DB = Path("docs/docgraph.sqlite")
    with tempfile.TemporaryDirectory(prefix="docgraph_gui_live_") as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        (repo / "docgraph.config.yaml").write_text(
            """
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: false
""",
            encoding="utf-8",
        )
        (repo / "source.md").write_text("VSYNC_TIMEOUT controls capture frame timeout behavior.\n", encoding="utf-8")
        backend = DocGraphBackend(repo / "docs" / "docgraph.sqlite", root=repo)
        ingest = backend.ingest_source("doc_file", "source.md")
        chunk_id = backend.conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_id=?",
            (ingest["source_id"],),
        ).fetchone()["chunk_id"]
        proposal = backend.propose_update(
            "seed GUI live retrieval graph",
            [
                {"op": "upsert_node", "node_id": "register.vsync_timeout", "node_type": "register", "canonical_name": "VSYNC_TIMEOUT", "summary": "Capture frame timeout register."},
                {"op": "add_alias", "node_id": "register.vsync_timeout", "alias": "vsync timeout"},
                {"op": "upsert_claim", "claim_id": "claim.vsync_timeout.capture", "target_node_id": "register.vsync_timeout", "claim_text": "VSYNC_TIMEOUT controls capture frame timeout behavior.", "classification": "Fact", "chunk_ids": [chunk_id]},
            ],
            created_by="gui_live_retrieval_test",
        )
        backend.commit_update(proposal["proposal_id"])
        backend.close()

        args = gui.live_query_arguments(
            gui.detect_project_paths(repo),
            query="capture vsync timeout",
            anchors="VSYNC_TIMEOUT",
            role="firmware",
            intent="debug",
            mode="local",
            budget="small",
        )
        completed = subprocess.run([sys.executable, *args], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith(gui.LIVE_QUERY_RESULT_PREFIX))
        payload = json.loads(line[len(gui.LIVE_QUERY_RESULT_PREFIX):])
        assert payload["ok"] is True, payload
        assert "register.vsync_timeout" in payload["selected_anchor_ids"], payload
        assert payload["run_id"], payload

        ro = gui.open_readonly_connection(repo / "docs" / "docgraph.sqlite")
        details = gui.retrieval_run_details(ro, payload["run_id"])
        assert details["trace"]["final"]["anchor_ids"] == ["register.vsync_timeout"], details
        rendered = gui.format_retrieval_trace(details)
        assert "EXPLICIT ANCHOR RESOLUTION" in rendered
        assert "VSYNC_TIMEOUT -> register.vsync_timeout" in rendered
        ro.close()
    print("GUI_LIVE_RETRIEVAL_OK")


if __name__ == "__main__":
    main()
