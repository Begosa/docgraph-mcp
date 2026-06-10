#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import docgraph_gui_pyqt6 as gui  # noqa: E402


def main() -> None:
    gui.DEFAULT_DOCGRAPH_DB = Path("docs/docgraph.sqlite")
    with tempfile.TemporaryDirectory(prefix="docgraph_gui_test_") as td:
        root = Path(td) / "project with spaces"
        (root / "docs" / "logs").mkdir(parents=True)
        (root / "docgraph.config.yaml").write_text("retrieval_models: {}\n", encoding="utf-8")
        db = root / "docs" / "docgraph.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE nodes(node_id TEXT PRIMARY KEY, node_type TEXT, canonical_name TEXT, summary TEXT, visibility TEXT, finder_role TEXT, audience_roles_json TEXT, interface_tags_json TEXT, status TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE claims(claim_id TEXT PRIMARY KEY, target_node_id TEXT, target_edge_id TEXT, claim_text TEXT, classification TEXT, confidence TEXT, visibility TEXT, finder_role TEXT, audience_roles_json TEXT, interface_tags_json TEXT, status TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO nodes VALUES('node.a','concept','A','summary','shared','rtl','[\"firmware\"]','[\"config\"]','active','now')")
        conn.commit(); conn.close()

        paths = gui.detect_project_paths(root)
        assert paths.db_path == db.resolve()
        assert paths.log_path == (root / "docs" / "logs" / "docgraph-mcp.log").resolve()
        db_paths = gui.detect_project_paths_from_db(db)
        assert db_paths.root == root.resolve()
        assert db_paths.db_path == db.resolve()
        assert gui.supports_live_retrieval(db_paths) is True
        uri = gui.sqlite_readonly_uri(db)
        assert uri.startswith("file:") and "mode=ro" in uri and "project%20with%20spaces" in uri
        ro = gui.open_readonly_connection(db)
        counts = gui.overview_counts(ro)
        assert counts["nodes"] == 1
        rows = gui.search_nodes(ro, text="A")
        assert rows and rows[0]["node_id"] == "node.a"
        try:
            ro.execute("INSERT INTO nodes(node_id) VALUES('x')")
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError("read-only GUI connection allowed write")
        ro.close()

        standalone = Path(td) / "standalone.sqlite"
        sqlite3.connect(standalone).close()
        standalone_paths = gui.detect_project_paths_from_db(standalone)
        assert standalone_paths.root == standalone.parent.resolve()
        assert standalone_paths.db_path == standalone.resolve()
        assert standalone_paths.config_path == standalone.parent.resolve() / "docgraph.config.yaml"
        assert gui.supports_live_retrieval(standalone_paths) is False
    print("GUI_PYQT6_PATH_DETECTION_OK")


if __name__ == "__main__":
    main()
