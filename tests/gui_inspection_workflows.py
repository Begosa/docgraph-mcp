#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import docgraph_gui_pyqt6 as gui  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="docgraph_gui_inspection_") as td:
        db = Path(td) / "docgraph.sqlite"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sources(
              source_id TEXT PRIMARY KEY, source_type TEXT, uri TEXT, name TEXT, status TEXT
            );
            CREATE TABLE episodes(
              episode_id TEXT PRIMARY KEY, source_id TEXT, episode_type TEXT, status TEXT
            );
            CREATE TABLE chunks(
              chunk_id TEXT PRIMARY KEY, episode_id TEXT, source_id TEXT, locator TEXT, text TEXT, status TEXT
            );
            CREATE TABLE nodes(
              node_id TEXT PRIMARY KEY, node_type TEXT, canonical_name TEXT, summary TEXT,
              visibility TEXT, finder_role TEXT, audience_roles_json TEXT,
              interface_tags_json TEXT, status TEXT, updated_at TEXT
            );
            CREATE TABLE aliases(alias_id TEXT, node_id TEXT, alias TEXT, normalized_alias TEXT);
            CREATE TABLE edges(
              edge_id TEXT PRIMARY KEY, from_node_id TEXT, relation TEXT, to_node_id TEXT,
              summary TEXT, visibility TEXT, finder_role TEXT, audience_roles_json TEXT,
              interface_tags_json TEXT, status TEXT
            );
            CREATE TABLE claims(
              claim_id TEXT PRIMARY KEY, target_node_id TEXT, target_edge_id TEXT, claim_text TEXT,
              classification TEXT, confidence TEXT, visibility TEXT, finder_role TEXT,
              audience_roles_json TEXT, interface_tags_json TEXT, status TEXT, updated_at TEXT
            );
            CREATE TABLE claim_evidence(
              claim_id TEXT, chunk_id TEXT, evidence_role TEXT, strength TEXT, status TEXT
            );
            CREATE TABLE retrieval_runs(
              run_id TEXT PRIMARY KEY, query TEXT, anchors_json TEXT, mode TEXT, role TEXT,
              budget TEXT, result_summary_json TEXT, trace_json TEXT, created_at TEXT
            );
            """
        )
        conn.execute("INSERT INTO sources VALUES(?,?,?,?,?)", ("src.report", "agent_report", "agent://rtl/report", "RTL report", "active"))
        conn.execute("INSERT INTO episodes VALUES(?,?,?,?)", ("ep.report", "src.report", "agent_investigation", "active"))
        conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", ("chunk.report", "ep.report", "src.report", "paragraph 1", "Generated 248 headers around line 91.", "active"))
        conn.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?)", ("flow.build", "flow", "build_flow", "Build flow", "shared", "build", '["firmware"]', '["build"]', "active", "now"))
        conn.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?,?,?)", ("target.system", "build_target", "system_core", "System core", "shared", "build", '["firmware"]', '["build"]', "active", "now"))
        conn.execute("INSERT INTO edges VALUES(?,?,?,?,?,?,?,?,?,?)", ("edge.flow.target", "target.system", "part_of", "flow.build", "Target belongs to flow.", "shared", "build", '["firmware"]', '["build"]', "active"))
        conn.execute("INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", ("claim.transient", "target.system", None, "Build generated 248 headers at line 91.", "Fact", "medium", "local", "build", '["firmware"]', '["build"]', "active", "now"))
        conn.execute("INSERT INTO claim_evidence VALUES(?,?,?,?,?)", ("claim.transient", "chunk.report", "supports", "medium", "active"))
        trace = {
            "anchor_resolution": {
                "lexical_search": {
                    "anchor_candidates": [{"node_id": "target.system", "score": 70.0, "decision": "kept", "reasons": ["claim_fts_target"]}],
                    "anchor_filter": {"threshold": 45.0, "relative_delta": 25, "kept_count": 1, "filtered_count": 0},
                }
            },
            "semantic_promotion": {
                "enabled": True,
                "reason": "ok",
                "promoted_anchors": ["target.system"],
                "coherence_gate": {
                    "enabled": True,
                    "applied": True,
                    "reason": "lexical context established",
                    "max_depth": 2,
                    "connections": [{"node_id": "target.system", "lexical_anchor_id": "flow.build", "distance": 1, "edge_ids": ["edge.flow.target"]}],
                    "rejected_semantic_only": ["target.unrelated"],
                },
            },
            "final": {"anchor_ids": ["target.system"], "claim_ids": ["claim.transient"], "edge_ids": ["edge.flow.target"], "evidence_chunk_ids": ["chunk.report"]},
            "timings_ms": {"anchor_resolution": 1.0, "local": 0.5},
        }
        conn.execute(
            "INSERT INTO retrieval_runs VALUES(?,?,?,?,?,?,?,?,?)",
            ("run.1", "system core build headers", '["target.system"]', "local", "build", "small", '{"anchors": 1, "claims": 1}', json.dumps(trace), "now"),
        )
        conn.commit()
        conn.close()

        ro = gui.open_readonly_connection(db)
        runs = gui.search_retrieval_runs(ro, text="headers", mode="local", role="build")
        assert runs and runs[0]["run_id"] == "run.1", runs
        formatted_trace = gui.format_retrieval_trace(gui.retrieval_run_details(ro, "run.1"))
        assert "LEXICAL ANCHOR DECISIONS" in formatted_trace
        assert "target.system" in formatted_trace
        assert "claim.transient" in formatted_trace
        assert "rejected disconnected semantic-only anchors: target.unrelated" in formatted_trace

        graph = gui.neighborhood_graph(ro, "target.system", max_hops=1, max_nodes=5)
        assert {n["node_id"] for n in graph["nodes"]} == {"target.system", "flow.build"}, graph
        assert [e["edge_id"] for e in graph["edges"]] == ["edge.flow.target"], graph

        proof = gui.format_claim_proof(gui.claim_details(ro, "claim.transient"))
        assert "EVIDENCE CHAIN" in proof
        assert "agent://rtl/report" in proof
        assert "Generated 248 headers" in proof

        checks = gui.quality_checks(ro)
        assert checks["possible_transient_claim_materiality"], checks
        assert checks["active_claims_supported_only_by_agent_reports"], checks
        assert checks["local_claims_with_cross_role_tags"], checks
        ro.close()

        base_config = """
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: false
"""
        changed_config = """
retrieval_models:
  embeddings:
    enabled: false
  reranker:
    enabled: true
    model: models/Qwen--Qwen3-Reranker-0.6B
"""
        ok, message = gui.validate_config_text(base_config)
        assert ok, message
        assert gui.config_change_requires_relaunch(base_config, changed_config), changed_config
        rows = gui.config_structured_rows(changed_config)
        row_by_path = {row["path"]: row for row in rows}
        assert row_by_path["retrieval_models.reranker.enabled"]["type"] == "bool", row_by_path
        assert row_by_path["retrieval_models.reranker.enabled"]["source"] == "file", row_by_path
        assert row_by_path["retrieval_models.reranker.model"]["type"] == "text", row_by_path
        assert row_by_path["retrieval.default_mode"]["source"] == "default", row_by_path["retrieval.default_mode"]
        assert row_by_path["retrieval.modes.mix.semantic_anchor_promotion.coherence_max_depth_by_budget.small"]["type"] == "int", row_by_path
        assert "cross-encoder" in gui.config_path_description("retrieval_models.reranker.enabled"), gui.config_path_description("retrieval_models.reranker.enabled")
        assert gui.config_section_title("retrieval_models.reranker") == "Reranker"
        parsed_config = gui.parse_config_text(base_config)
        assert isinstance(parsed_config, dict), parsed_config
        gui.set_config_path_value(parsed_config, "retrieval_models.reranker.enabled", True)
        rendered_config = gui.dump_config_text(parsed_config)
        assert "enabled: true" in rendered_config or '"enabled": true' in rendered_config, rendered_config
        assert gui.parse_config_value("false", True) is False
        assert gui.parse_config_value("24", 12) == 24
        assert gui.parse_config_value('["firmware", "rtl"]', []) == ["firmware", "rtl"]
        if gui.yaml is not None:
            comment_only_config = base_config + "\n# GUI note only\n"
            assert not gui.config_change_requires_relaunch(base_config, comment_only_config), comment_only_config
            explicit_default_config = base_config + "\nretrieval:\n  default_mode: hybrid\n"
            assert not gui.config_change_requires_relaunch(base_config, explicit_default_config), explicit_default_config
    print("GUI_INSPECTION_WORKFLOWS_OK")


if __name__ == "__main__":
    main()
