#!/usr/bin/env python3
from __future__ import annotations

"""Windows-friendly PyQt6 inspector for DocGraph SQLite databases.

Browsing intentionally does not import the MCP backend and reads SQLite directly
in read-only mode. The explicit Live Retrieval action starts a separate backend
query process, which may write retrieval diagnostics/cache state but never
curates graph knowledge.
"""

import argparse
import copy
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

try:  # Python distributions can be built without stdlib sqlite3.
    import sqlite3  # type: ignore
except ImportError:  # pragma: no cover - environment specific
    import pysqlite3 as sqlite3  # type: ignore

try:  # Config editing is nicer with YAML validation, but browsing can work without it.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCGRAPH_DB = Path(os.environ.get("DOCGRAPH_DB", "docs/docgraph.sqlite")).expanduser()

_REPO_MCP_SRC = BUNDLE_ROOT / "src"
if _REPO_MCP_SRC.exists() and str(_REPO_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_MCP_SRC))

try:  # Keep the GUI's structured config view aligned with backend defaults.
    from docgraph_mcp.config import DEFAULT_CONFIG as DOCGRAPH_DEFAULT_CONFIG
except Exception:  # pragma: no cover - installed GUI may browse without package sources.
    DOCGRAPH_DEFAULT_CONFIG: dict[str, Any] = {}


DEFAULT_MAX_ROWS = 500
HIGH_LEVEL_TYPES = {"flow", "feature", "concept", "interface", "runbook"}
LIVE_QUERY_RESULT_PREFIX = "DOCGRAPH_QUERY_RESULT="
MISSING_CONFIG_VALUE = object()
CONFIG_FIELD_DESCRIPTIONS = {
    "logging": "Backend and MCP JSON-lines logging behavior.",
    "logging.enabled": "Enable structured backend/MCP logging.",
    "logging.level": "Minimum log level: trace, debug, info, warning, or error.",
    "logging.file": "Log file path relative to DOCGRAPH_ROOT unless absolute.",
    "logging.max_bytes": "Maximum log file size before rotation.",
    "logging.backup_count": "Number of rotated log files to keep.",
    "logging.stderr": "Also emit logs to stderr.",
    "logging.include_payloads": "Include full request/response payloads in logs; useful for debugging but can be noisy.",
    "logging.payload_preview_chars": "Maximum characters retained when payload logging is summarized.",
    "data_model": "Allowed claim/evidence lifecycle values.",
    "shared_knowledge": "Visibility, interface tag, and cross-role taxonomy controls.",
    "source_handling": "Source ingestion and chunking behavior.",
    "source_handling.lines_per_chunk": "Line count per chunk for code, RTL, build files, and logs.",
    "source_handling.line_chunk_overlap": "Number of overlapping lines between adjacent line chunks.",
    "source_handling.paragraph_max_chars": "Maximum paragraph chunk size for document-style sources.",
    "taxonomy": "Allowed node and relation taxonomy values for curated graph objects.",
    "retrieval": "Context retrieval defaults, modes, budgets, and ranking weights.",
    "retrieval.default_mode": "Default context mode when the caller does not specify one.",
    "retrieval.default_budget": "Default context budget when the caller does not specify one.",
    "retrieval.include_stale_warnings": "Include stale/superseded/conflict warnings in returned context.",
    "retrieval.max_extracted_terms": "Maximum terms extracted from natural-language queries.",
    "retrieval.max_anchor_expansion": "Maximum candidate anchors considered during query expansion.",
    "retrieval.neighbor_edges_per_anchor": "Limit for direct neighbor edges pulled per selected anchor.",
    "retrieval.anchor_filter": "Lexical anchor filter that drops candidates far below the best match.",
    "retrieval.anchor_filter.enabled": "Enable relative filtering of lexical anchor candidates.",
    "retrieval.anchor_filter.relative_delta": "Keep anchors whose score is within this delta of the best lexical score.",
    "retrieval.anchor_filter.min_score": "Absolute lexical score floor.",
    "retrieval.anchor_filter.min_anchors": "Minimum lexical anchors retained even if below the dynamic threshold.",
    "retrieval.modes": "Per-mode retrieval behavior.",
    "retrieval.modes.local.enabled": "Enable direct-neighborhood retrieval mode.",
    "retrieval.modes.global.enabled": "Enable high-level frame retrieval mode.",
    "retrieval.modes.global.max_depth": "Maximum upward graph distance for global frame discovery.",
    "retrieval.modes.bridge.enabled": "Enable bounded bridge-path retrieval mode.",
    "retrieval.modes.bridge.max_depth": "Maximum graph path length for bridge search.",
    "retrieval.modes.bridge.max_paths": "Maximum bridge paths returned.",
    "retrieval.modes.bridge.max_anchors": "Maximum anchors used for pairwise bridge search.",
    "retrieval.modes.hybrid.enabled": "Enable local + global + bridge retrieval mode.",
    "retrieval.modes.mix.enabled": "Enable hybrid retrieval plus optional semantic retrieval.",
    "retrieval.modes.mix.semantic_anchor_promotion": "Controls if semantic/vector hits may become anchors before expansion.",
    "retrieval.modes.mix.semantic_anchor_promotion.enabled": "Allow semantic candidates to become retrieval anchors in mix mode.",
    "retrieval.modes.mix.semantic_anchor_promotion.top_semantic_results": "Semantic result window considered for promotion.",
    "retrieval.modes.mix.semantic_anchor_promotion.require_graph_coherence": "Require semantic-only anchors to connect to lexical anchors before expansion.",
    "retrieval.modes.mix.semantic_anchor_promotion.max_promoted_anchors": "Maximum semantic-only anchors added to lexical anchors.",
    "retrieval.modes.mix.semantic_anchor_promotion.min_lexical_anchors": "Minimum lexical anchors kept before fusion can add semantic-only anchors.",
    "retrieval.modes.mix.semantic_anchor_promotion.min_score": "Absolute semantic score floor before promotion.",
    "retrieval.modes.mix.semantic_anchor_promotion.relative_delta": "Dynamic semantic score window from the top semantic result.",
    "retrieval.modes.mix.semantic_anchor_promotion.lexical_weight": "Reciprocal-rank-fusion weight for lexical anchors.",
    "retrieval.modes.mix.semantic_anchor_promotion.semantic_weight": "Reciprocal-rank-fusion weight for semantic candidates.",
    "retrieval.modes.mix.semantic_anchor_promotion.rrf_k": "RRF smoothing constant; higher values flatten rank differences.",
    "retrieval.modes.mix.semantic_anchor_promotion.coherence_min_lexical_anchors": "Minimum lexical anchor count before graph-coherence gating is applied.",
    "retrieval.modes.mix.semantic_anchor_promotion.coherence_max_depth": "Fallback max active graph distance from semantic-only node to lexical context.",
    "retrieval.modes.mix.semantic_anchor_promotion.coherence_max_depth_by_budget": "Per-budget graph-coherence depth override, e.g. small/medium=2, large=3.",
    "retrieval.global_scope": "High-level node/relation preferences for global frame expansion.",
    "retrieval.bridge_scope": "Relation preferences for bridge-path search.",
    "retrieval.budgets": "Node/claim/edge/evidence limits by small, medium, and large budgets.",
    "retrieval.ranking": "Ranking weights for lexical search, graph expansion, visibility, role fit, and semantic candidates.",
    "retrieval_models": "Optional embedding and reranker model configuration for mix mode.",
    "retrieval_models.embeddings": "Embedding/vector retrieval provider configuration.",
    "retrieval_models.embeddings.enabled": "Enable embedding search for mix mode.",
    "retrieval_models.embeddings.provider": "Embedding provider adapter: sentence_transformers or http.",
    "retrieval_models.embeddings.model": "Embedding model ID or local path.",
    "retrieval_models.embeddings.preload_on_boot": "Load embedding provider during backend startup when possible.",
    "retrieval_models.embeddings.incremental_cache_enabled": "Persist candidate vectors in embedding_items and re-embed only changed text.",
    "retrieval_models.embeddings.normalize": "Normalize vectors before cosine similarity scoring.",
    "retrieval_models.embeddings.query_instruction": "Optional query prefix used by some sentence-transformer retrieval models.",
    "retrieval_models.embeddings.max_in_memory_items": "Maximum active nodes/claims/edges materialized as semantic candidates.",
    "retrieval_models.embeddings.top_k": "Semantic candidates retained in the context packet.",
    "retrieval_models.reranker": "Cross-encoder reranker configuration for semantic candidate ordering.",
    "retrieval_models.reranker.enabled": "Enable cross-encoder reranking of semantic candidates.",
    "retrieval_models.reranker.provider": "Reranker provider adapter.",
    "retrieval_models.reranker.model": "Reranker model ID or local path.",
    "retrieval_models.reranker.preload_on_boot": "Load reranker provider during backend startup.",
    "retrieval_models.reranker.top_k_input": "Number of semantic candidates passed into the reranker.",
    "retrieval_models.reranker.top_k_output": "Number of reranked candidates retained.",
    "retrieval_models.reranker.min_relevance_score": "Candidates below this score are deprioritized instead of removed.",
    "retrieval_models.llm_reranker": "Optional second-stage LLM relevance reranker for rescue retrieval.",
    "retrieval_models.llm_reranker.enabled": "Enable LLM reranking after the cross-encoder stage when triggered.",
    "retrieval_models.llm_reranker.provider": "LLM reranker adapter. Currently http_chat only.",
    "retrieval_models.llm_reranker.base_url_env": "Environment variable holding the company agent base URL.",
    "retrieval_models.llm_reranker.api_key_env": "Environment variable holding the company agent bearer token.",
    "retrieval_models.llm_reranker.preferred_models": "Ordered model ids; the first id available from the models endpoint is used.",
    "retrieval_models.llm_reranker.model": "Optional fixed model override. Leave empty to use preferred_models discovery.",
    "retrieval_models.llm_reranker.trigger": "When to run the LLM stage: always, rescue_only, or disabled.",
    "retrieval_models.llm_reranker.top_k_input": "Semantic candidates passed into the LLM reranker.",
    "retrieval_models.llm_reranker.top_k_output": "Candidates retained after LLM reranking.",
    "retrieval_models.llm_reranker.min_relevance_score": "Candidates below this LLM score are deprioritized.",
    "retrieval_models.llm_reranker.tool_name": "Forced function/tool name used for structured rerank scores.",
    "retrieval_models.llm_reranker.rescue.max_lexical_anchors": "Run LLM reranking when lexical anchor count is at or below this value.",
    "roles": "Role-aware visibility, ranking affinity, and suggested verification checks.",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    db_path: Path
    config_path: Path
    log_path: Path
    rendered_dir: Path


def detect_project_paths(root: str | os.PathLike[str]) -> ProjectPaths:
    base = Path(root).expanduser().resolve()
    db_path = DEFAULT_DOCGRAPH_DB
    if not db_path.is_absolute():
        db_path = base / db_path
    config_path = base / "docgraph.config.yaml"
    bundled_config = base / ".opencode" / "docgraph" / "docgraph.config.yaml"
    if not config_path.exists() and bundled_config.exists():
        config_path = bundled_config
    return ProjectPaths(
        root=base,
        db_path=db_path.resolve(),
        config_path=config_path,
        log_path=base / "docs" / "logs" / "docgraph-mcp.log",
        rendered_dir=base / "docs" / "rendered",
    )


def detect_project_paths_from_db(db_path: str | os.PathLike[str]) -> ProjectPaths:
    db = Path(db_path).expanduser().resolve()
    configured = detect_project_paths(os.environ.get("DOCGRAPH_ROOT", BUNDLE_ROOT))
    if db == configured.db_path:
        return configured
    root = db.parent.parent if db.parent.name == "docs" else db.parent
    return ProjectPaths(
        root=root,
        db_path=db,
        config_path=root / "docgraph.config.yaml",
        log_path=root / "docs" / "logs" / "docgraph-mcp.log" if db.parent.name == "docs" else db.parent / "logs" / "docgraph-mcp.log",
        rendered_dir=root / "docs" / "rendered" if db.parent.name == "docs" else db.parent / "rendered",
    )


def supports_live_retrieval(paths: ProjectPaths) -> bool:
    return paths.config_path.exists()


def live_query_arguments(
    paths: ProjectPaths,
    *,
    query: str,
    anchors: str,
    role: str,
    intent: str,
    mode: str,
    budget: str,
) -> list[str]:
    args = [
        str(Path(__file__).resolve().with_name("docgraph_query.py")),
        "--root",
        str(paths.root),
        "--db",
        str(paths.db_path),
        "--config",
        str(paths.config_path),
        "--mode",
        mode,
        "--budget",
        budget,
        "--json-line",
    ]
    for flag, value in (("--query", query), ("--anchors", anchors), ("--role", role), ("--intent", intent)):
        if value.strip():
            args.extend([flag, value.strip()])
    return args


def sqlite_readonly_uri(db_path: str | os.PathLike[str]) -> str:
    path = Path(db_path).expanduser().resolve()
    # SQLite URI paths are more reliable cross-platform with forward slashes and
    # percent-encoding for spaces/special characters.
    encoded = quote(str(path).replace("\\", "/"), safe="/:_")
    return "file:" + encoded + "?mode=ro"


def open_readonly_connection(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    uri = sqlite_readonly_uri(db_path)
    conn = sqlite3.connect(uri, uri=True, timeout=3.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name=? AND type IN ('table','view')", (table,)).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(r["name"]) for r in conn.execute("PRAGMA table_info(%s)" % table).fetchall()}


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def safe_json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [str(value)]


def safe_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def preview(text: Any, limit: int = 220) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."


def overview_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        "sources",
        "episodes",
        "chunks",
        "nodes",
        "aliases",
        "edges",
        "claims",
        "claim_evidence",
        "proposals",
        "commits",
        "retrieval_runs",
    ]
    out: dict[str, int] = {}
    for table in tables:
        if table_exists(conn, table):
            out[table] = int(conn.execute("SELECT COUNT(*) AS n FROM %s" % table).fetchone()["n"])
    if table_exists(conn, "claims"):
        for status in ["active", "needs_review", "stale", "superseded", "contradicted", "retired"]:
            row = conn.execute("SELECT COUNT(*) AS n FROM claims WHERE status=?", (status,)).fetchone()
            out["claims." + status] = int(row["n"])
    return out


def search_nodes(conn: sqlite3.Connection, text: str = "", node_type: str = "", visibility: str = "", role: str = "", tag: str = "", limit: int = DEFAULT_MAX_ROWS) -> list[dict[str, Any]]:
    clauses = ["n.status IS NOT NULL"]
    args: list[Any] = []
    if text:
        like = "%" + text + "%"
        if table_exists(conn, "aliases"):
            clauses.append("(n.node_id LIKE ? OR n.canonical_name LIKE ? OR n.summary LIKE ? OR EXISTS (SELECT 1 FROM aliases a WHERE a.node_id=n.node_id AND a.alias LIKE ?))")
            args.extend([like, like, like, like])
        else:
            clauses.append("(n.node_id LIKE ? OR n.canonical_name LIKE ? OR n.summary LIKE ?)")
            args.extend([like, like, like])
    if node_type:
        clauses.append("n.node_type=?")
        args.append(node_type)
    if visibility:
        clauses.append("n.visibility=?")
        args.append(visibility)
    if role:
        like = "%\"" + role + "\"%"
        clauses.append("(n.finder_role=? OR n.audience_roles_json LIKE ? OR n.visibility IN ('shared','global','shared_candidate'))")
        args.extend([role, like])
    if tag:
        like = "%\"" + tag + "\"%"
        clauses.append("n.interface_tags_json LIKE ?")
        args.append(like)
    sql = """
        SELECT n.node_id, n.node_type, n.canonical_name, n.summary, n.visibility, n.finder_role,
               n.audience_roles_json, n.interface_tags_json, n.status, n.updated_at
        FROM nodes n
        WHERE %s
        ORDER BY n.canonical_name
        LIMIT ?
    """ % " AND ".join(clauses)
    args.append(limit)
    return rows_to_dicts(conn.execute(sql, args).fetchall())


def list_distinct(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    if not table_exists(conn, table) or column not in table_columns(conn, table):
        return []
    rows = conn.execute("SELECT DISTINCT %s AS v FROM %s WHERE %s IS NOT NULL AND %s <> '' ORDER BY %s" % (column, table, column, column, column)).fetchall()
    return [str(r["v"]) for r in rows]


def node_details(conn: sqlite3.Connection, node_id: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    node = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    data["node"] = dict(node) if node else None
    data["aliases"] = rows_to_dicts(conn.execute("SELECT * FROM aliases WHERE node_id=? ORDER BY alias", (node_id,)).fetchall()) if table_exists(conn, "aliases") else []
    data["claims"] = rows_to_dicts(conn.execute("SELECT * FROM claims WHERE target_node_id=? ORDER BY updated_at DESC", (node_id,)).fetchall()) if table_exists(conn, "claims") else []
    data["out_edges"] = rows_to_dicts(conn.execute("SELECT * FROM edges WHERE from_node_id=? ORDER BY relation, to_node_id", (node_id,)).fetchall()) if table_exists(conn, "edges") else []
    data["in_edges"] = rows_to_dicts(conn.execute("SELECT * FROM edges WHERE to_node_id=? ORDER BY relation, from_node_id", (node_id,)).fetchall()) if table_exists(conn, "edges") else []
    claim_ids = [c["claim_id"] for c in data["claims"]]
    edge_ids = [e["edge_id"] for e in data["out_edges"] + data["in_edges"]]
    if edge_ids and table_exists(conn, "claims"):
        q = ",".join("?" for _ in edge_ids)
        data["edge_claims"] = rows_to_dicts(conn.execute("SELECT * FROM claims WHERE target_edge_id IN (%s) ORDER BY updated_at DESC" % q, edge_ids).fetchall())
        claim_ids.extend([c["claim_id"] for c in data["edge_claims"]])
    else:
        data["edge_claims"] = []
    data["evidence"] = evidence_for_claims(conn, claim_ids)
    return data


def evidence_for_claims(conn: sqlite3.Connection, claim_ids: list[str]) -> list[dict[str, Any]]:
    if not claim_ids or not all(table_exists(conn, t) for t in ["claim_evidence", "chunks", "sources", "episodes"]):
        return []
    q = ",".join("?" for _ in claim_ids)
    sql = """
        SELECT ce.claim_id, ce.evidence_role, ce.strength, ce.status AS evidence_status,
               ch.chunk_id, ch.locator, ch.text, ch.status AS chunk_status,
               ep.episode_id, ep.episode_type, ep.status AS episode_status,
               s.source_id, s.uri, s.source_type, s.name, s.status AS source_status
        FROM claim_evidence ce
        JOIN chunks ch ON ch.chunk_id=ce.chunk_id
        JOIN episodes ep ON ep.episode_id=ch.episode_id
        JOIN sources s ON s.source_id=ch.source_id
        WHERE ce.claim_id IN (%s)
        ORDER BY ce.claim_id, s.uri, ch.locator
    """ % q
    return rows_to_dicts(conn.execute(sql, claim_ids).fetchall())


def claim_details(conn: sqlite3.Connection, claim_id: str) -> dict[str, Any]:
    """Return one claim with its target and full source-to-chunk evidence chain."""
    row = conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
    claim = dict(row) if row else None
    data: dict[str, Any] = {"claim": claim, "target_node": None, "target_edge": None, "evidence": []}
    if not claim:
        return data
    if claim.get("target_node_id") and table_exists(conn, "nodes"):
        target = conn.execute("SELECT * FROM nodes WHERE node_id=?", (claim["target_node_id"],)).fetchone()
        data["target_node"] = dict(target) if target else None
    if claim.get("target_edge_id") and table_exists(conn, "edges"):
        target = conn.execute("SELECT * FROM edges WHERE edge_id=?", (claim["target_edge_id"],)).fetchone()
        data["target_edge"] = dict(target) if target else None
    data["evidence"] = evidence_for_claims(conn, [claim_id])
    return data


def format_claim_proof(data: dict[str, Any]) -> str:
    claim = data.get("claim") or {}
    if not claim:
        return "No claim selected."
    lines = [
        "CLAIM",
        str(claim.get("claim_text") or ""),
        "",
        "Identity: %s" % claim.get("claim_id", ""),
        "Status: %s    Classification: %s    Confidence: %s" % (claim.get("status", ""), claim.get("classification", ""), claim.get("confidence", "")),
        "Visibility: %s    Finder: %s    Audience: %s" % (claim.get("visibility", ""), claim.get("finder_role", ""), ", ".join(safe_json_list(claim.get("audience_roles_json"))) or "-"),
        "Tags: %s" % (", ".join(safe_json_list(claim.get("interface_tags_json"))) or "-"),
        "",
        "TARGET",
    ]
    target_node = data.get("target_node")
    target_edge = data.get("target_edge")
    if target_node:
        lines.append("Node: %s (%s) - %s" % (target_node.get("node_id"), target_node.get("node_type"), target_node.get("canonical_name")))
    elif target_edge:
        lines.append("Edge: %s --%s--> %s" % (target_edge.get("from_node_id"), target_edge.get("relation"), target_edge.get("to_node_id")))
    else:
        lines.append("No resolved target.")
    lines += ["", "EVIDENCE CHAIN"]
    evidence = data.get("evidence") or []
    if not evidence:
        lines.append("No linked evidence.")
    for idx, ev in enumerate(evidence, 1):
        lines += [
            "%d. [%s/%s] %s" % (idx, ev.get("evidence_role", ""), ev.get("strength", ""), ev.get("uri", "")),
            "   source=%s (%s) episode=%s (%s) chunk=%s (%s)" % (
                ev.get("source_id", ""), ev.get("source_status", ""), ev.get("episode_id", ""), ev.get("episode_status", ""), ev.get("chunk_id", ""), ev.get("chunk_status", "")
            ),
            "   locator=%s" % (ev.get("locator") or "-"),
            "   excerpt=%s" % preview(ev.get("text"), 360),
        ]
    return "\n".join(lines)


def search_claims(conn: sqlite3.Connection, text: str = "", status: str = "", classification: str = "", visibility: str = "", role: str = "", limit: int = DEFAULT_MAX_ROWS) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    args: list[Any] = []
    if text:
        like = "%" + text + "%"
        clauses.append("(claim_id LIKE ? OR claim_text LIKE ?)")
        args.extend([like, like])
    if status:
        clauses.append("status=?")
        args.append(status)
    if classification:
        clauses.append("classification=?")
        args.append(classification)
    if visibility:
        clauses.append("visibility=?")
        args.append(visibility)
    if role:
        like = "%\"" + role + "\"%"
        clauses.append("(finder_role=? OR audience_roles_json LIKE ? OR visibility IN ('shared','global','shared_candidate'))")
        args.extend([role, like])
    sql = "SELECT * FROM claims WHERE %s ORDER BY updated_at DESC LIMIT ?" % " AND ".join(clauses)
    args.append(limit)
    return rows_to_dicts(conn.execute(sql, args).fetchall())


def search_edges(conn: sqlite3.Connection, text: str = "", relation: str = "", visibility: str = "", role: str = "", limit: int = DEFAULT_MAX_ROWS) -> list[dict[str, Any]]:
    clauses = ["1=1"]
    args: list[Any] = []
    if text:
        like = "%" + text + "%"
        clauses.append("(edge_id LIKE ? OR from_node_id LIKE ? OR to_node_id LIKE ? OR relation LIKE ? OR summary LIKE ?)")
        args.extend([like, like, like, like, like])
    if relation:
        clauses.append("relation=?")
        args.append(relation)
    if visibility:
        clauses.append("visibility=?")
        args.append(visibility)
    if role:
        like = "%\"" + role + "\"%"
        clauses.append("(finder_role=? OR audience_roles_json LIKE ? OR visibility IN ('shared','global','shared_candidate'))")
        args.extend([role, like])
    sql = "SELECT * FROM edges WHERE %s ORDER BY relation, from_node_id, to_node_id LIMIT ?" % " AND ".join(clauses)
    args.append(limit)
    return rows_to_dicts(conn.execute(sql, args).fetchall())


def all_neighbors(conn: sqlite3.Connection, node_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM edges WHERE status='active' AND (from_node_id=? OR to_node_id=?)", (node_id, node_id)).fetchall()
    return rows_to_dicts(rows)


def neighborhood_graph(conn: sqlite3.Connection, center_node_id: str, max_hops: int = 1, max_nodes: int = 25) -> dict[str, list[dict[str, Any]]]:
    """Return a bounded active graph neighborhood for visual inspection."""
    if not center_node_id or not table_exists(conn, "nodes") or not table_exists(conn, "edges"):
        return {"nodes": [], "edges": []}
    center = conn.execute("SELECT * FROM nodes WHERE node_id=?", (center_node_id,)).fetchone()
    if center is None:
        return {"nodes": [], "edges": []}
    node_rows: dict[str, dict[str, Any]] = {center_node_id: dict(center)}
    edge_rows: dict[str, dict[str, Any]] = {}
    queue: deque[tuple[str, int]] = deque([(center_node_id, 0)])
    expanded: set[str] = set()
    while queue:
        node_id, depth = queue.popleft()
        if node_id in expanded or depth >= max_hops:
            continue
        expanded.add(node_id)
        for edge in all_neighbors(conn, node_id):
            other = edge["to_node_id"] if edge["from_node_id"] == node_id else edge["from_node_id"]
            if other not in node_rows:
                if len(node_rows) >= max_nodes:
                    continue
                row = conn.execute("SELECT * FROM nodes WHERE node_id=?", (other,)).fetchone()
                if row is None:
                    continue
                node_rows[other] = dict(row)
                queue.append((other, depth + 1))
            if other in node_rows:
                edge_rows[edge["edge_id"]] = edge
    return {"nodes": list(node_rows.values()), "edges": list(edge_rows.values())}


def find_bridge_paths(conn: sqlite3.Connection, start_node_id: str, end_node_id: str, max_depth: int = 4, max_paths: int = 20) -> list[list[dict[str, Any]]]:
    if start_node_id == end_node_id:
        return []
    q: deque[tuple[str, list[dict[str, Any]], set[str]]] = deque([(start_node_id, [], {start_node_id})])
    paths: list[list[dict[str, Any]]] = []
    while q and len(paths) < max_paths:
        node_id, path, seen = q.popleft()
        if len(path) >= max_depth:
            continue
        for edge in all_neighbors(conn, node_id):
            other = edge["to_node_id"] if edge["from_node_id"] == node_id else edge["from_node_id"]
            if other in seen:
                continue
            next_path = path + [edge]
            if other == end_node_id:
                paths.append(next_path)
            else:
                q.append((other, next_path, seen | {other}))
    return paths


def format_bridge_path(path: list[dict[str, Any]]) -> str:
    if not path:
        return ""
    parts: list[str] = []
    current = path[0]["from_node_id"]
    for edge in path:
        if edge["from_node_id"] == current:
            parts.append("%s --%s--> %s" % (edge["from_node_id"], edge["relation"], edge["to_node_id"]))
            current = edge["to_node_id"]
        else:
            parts.append("%s <--%s-- %s" % (edge["to_node_id"], edge["relation"], edge["from_node_id"]))
            current = edge["from_node_id"]
    return "\n".join(parts)


def search_retrieval_runs(conn: sqlite3.Connection, text: str = "", mode: str = "", role: str = "", limit: int = DEFAULT_MAX_ROWS) -> list[dict[str, Any]]:
    if not table_exists(conn, "retrieval_runs"):
        return []
    columns = table_columns(conn, "retrieval_runs")
    trace_column = "trace_json" if "trace_json" in columns else "'{}' AS trace_json"
    clauses = ["1=1"]
    args: list[Any] = []
    if text:
        clauses.append("(run_id LIKE ? OR query LIKE ? OR anchors_json LIKE ?)")
        like = "%" + text + "%"
        args.extend([like, like, like])
    if mode:
        clauses.append("mode=?")
        args.append(mode)
    if role:
        clauses.append("role=?")
        args.append(role)
    sql = "SELECT run_id, query, anchors_json, mode, role, budget, result_summary_json, %s, created_at FROM retrieval_runs WHERE %s ORDER BY created_at DESC, rowid DESC LIMIT ?" % (trace_column, " AND ".join(clauses))
    args.append(limit)
    return rows_to_dicts(conn.execute(sql, args).fetchall())


def retrieval_run_details(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    rows = search_retrieval_runs(conn, text=run_id, limit=20)
    row = next((item for item in rows if item.get("run_id") == run_id), None)
    if not row:
        return {}
    return {
        "run": row,
        "anchors": safe_json_list(row.get("anchors_json")),
        "summary": safe_json_object(row.get("result_summary_json")),
        "trace": safe_json_object(row.get("trace_json")),
    }


def format_retrieval_trace(data: dict[str, Any]) -> str:
    run = data.get("run") or {}
    if not run:
        return "No retrieval run selected."
    trace = data.get("trace") or {}
    lines = [
        "RETRIEVAL RUN",
        "Query: %s" % (run.get("query") or "-"),
        "Run: %s    Mode: %s    Role: %s    Budget: %s" % (run.get("run_id", ""), run.get("mode", ""), run.get("role") or "-", run.get("budget", "")),
        "Created: %s" % run.get("created_at", ""),
        "",
        "FINAL SUMMARY",
        json.dumps(data.get("summary") or {}, sort_keys=True),
    ]
    resolution = trace.get("anchor_resolution") or {}
    lines += ["", "EXPLICIT ANCHOR RESOLUTION"]
    explicit = resolution.get("explicit") or []
    if not explicit:
        lines.append("No explicit anchor inputs.")
    for item in explicit:
        lines.append("- %s -> %s (%s)" % (item.get("input"), item.get("node_id"), item.get("match_type")))
    base_ids = resolution.get("resolved_base_anchor_ids") or []
    if base_ids:
        lines.append("Base anchors before semantic promotion: %s" % ", ".join(base_ids))
    lexical = resolution.get("lexical_search") or {}
    lines += ["", "LEXICAL ANCHOR DECISIONS"]
    decisions = lexical.get("anchor_candidates") or []
    if not decisions:
        lines.append("No persisted lexical decisions (older row or explicit anchors only).")
    for item in decisions:
        lines.append("- %-17s score=%-8s %-32s %s" % (
            item.get("decision", ""),
            item.get("score", ""),
            item.get("node_id", ""),
            ", ".join(item.get("reasons") or []),
        ))
    filt = lexical.get("anchor_filter") or {}
    if filt:
        lines.append("Filter: threshold=%s delta=%s kept=%s filtered=%s" % (
            filt.get("threshold"), filt.get("relative_delta"), filt.get("kept_count"), filt.get("filtered_count")
        ))
    promotion = trace.get("semantic_promotion") or {}
    lines += ["", "SEMANTIC PROMOTION"]
    lines.append("Enabled: %s    Reason: %s    Promoted: %s" % (
        promotion.get("enabled", False), promotion.get("reason", "-"), ", ".join(promotion.get("promoted_anchors") or []) or "-"
    ))
    coherence = promotion.get("coherence_gate") or {}
    if coherence:
        lines.append("Coherence gate: enabled=%s applied=%s reason=%s max_depth=%s" % (
            coherence.get("enabled"), coherence.get("applied"), coherence.get("reason"), coherence.get("max_depth")
        ))
        for connection in coherence.get("connections") or []:
            lines.append("- accepted %s via lexical anchor %s distance=%s edges=%s" % (
                connection.get("node_id"), connection.get("lexical_anchor_id"), connection.get("distance"),
                ", ".join(connection.get("edge_ids") or []) or "-"
            ))
        rejected = coherence.get("rejected_semantic_only") or []
        if rejected:
            lines.append("- rejected disconnected semantic-only anchors: %s" % ", ".join(rejected))
    for item in trace.get("semantic_candidates") or []:
        lines.append("- %s:%s score=%s reranker=%s llm=%s" % (
            item.get("kind"), item.get("id"), item.get("score"), item.get("reranker_score", "-"), item.get("llm_reranker_score", "-")
        ))
    rerank_trace = trace.get("rerank_trace") or {}
    if rerank_trace:
        lines += ["", "RERANK STAGES"]
        for stage_name, stage in rerank_trace.items():
            if isinstance(stage, dict):
                lines.append("- %s enabled=%s applied=%s reason=%s model=%s" % (
                    stage_name, stage.get("enabled"), stage.get("applied"), stage.get("reason"), stage.get("model") or "-"
                ))
                for item in stage.get("scored") or []:
                    lines.append("- %s:%s score=%s final=%s retained=%s" % (
                        item.get("kind"),
                        item.get("id"),
                        item.get("llm_reranker_score", item.get("reranker_score", "-")),
                        item.get("final_score", "-"),
                        item.get("retained", False),
                    ))
    lines += ["", "EXPANSION"]
    for frame in trace.get("global_frames") or []:
        lines.append("- global frame %s distance=%s via=%s score=%s" % (frame.get("node_id"), frame.get("distance"), frame.get("via_edge_id") or "-", frame.get("score")))
    for path in trace.get("bridge_paths") or []:
        lines.append("- bridge %s -> %s score=%s nodes=%s" % (path.get("from_node_id"), path.get("to_node_id"), path.get("score"), " -> ".join(path.get("node_ids") or [])))
    final = trace.get("final") or {}
    lines += [
        "",
        "RETURNED CONTEXT",
        "Anchors: %s" % (", ".join(final.get("anchor_ids") or data.get("anchors") or []) or "-"),
        "Claims: %s" % (", ".join(final.get("claim_ids") or []) or "-"),
        "Edges: %s" % (", ".join(final.get("edge_ids") or []) or "-"),
        "Evidence chunks: %s" % (", ".join(final.get("evidence_chunk_ids") or []) or "-"),
        "",
        "TIMINGS (ms)",
        json.dumps(trace.get("timings_ms") or {}, sort_keys=True),
    ]
    return "\n".join(lines)


def quality_checks(conn: sqlite3.Connection, limit: int = DEFAULT_MAX_ROWS) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if table_exists(conn, "claims"):
        out["claims_without_target"] = rows_to_dicts(conn.execute(
            "SELECT claim_id, classification, status, claim_text FROM claims WHERE target_node_id IS NULL AND target_edge_id IS NULL ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall())
        if table_exists(conn, "claim_evidence"):
            out["active_claims_without_active_support"] = rows_to_dicts(conn.execute("""
                SELECT c.claim_id, c.classification, c.status, c.claim_text
                FROM claims c
                WHERE c.status='active' AND c.classification <> 'OpenQuestion'
                  AND NOT EXISTS (
                    SELECT 1 FROM claim_evidence ce JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                    WHERE ce.claim_id=c.claim_id AND ce.status='active' AND ce.evidence_role='supports' AND ch.status='active'
                  )
                ORDER BY c.updated_at DESC LIMIT ?
            """, (limit,)).fetchall())
        out["shared_candidates"] = rows_to_dicts(conn.execute("SELECT claim_id, target_node_id, target_edge_id, claim_text, audience_roles_json FROM claims WHERE visibility='shared_candidate' ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall())
        out["possible_transient_claim_materiality"] = rows_to_dicts(conn.execute("""
            SELECT claim_id, classification, visibility, claim_text, status
            FROM claims
            WHERE status='active'
              AND (
                lower(claim_text) GLOB '*line [0-9]*'
                OR lower(claim_text) GLOB '*lines [0-9]*'
                OR lower(claim_text) GLOB '*[0-9]* generated*'
                OR lower(claim_text) GLOB '*generated *[0-9]*'
                OR lower(claim_text) GLOB '*[0-9]* ms*'
              )
            ORDER BY updated_at DESC LIMIT ?
        """, (limit,)).fetchall())
        if all(table_exists(conn, t) for t in ["claim_evidence", "chunks", "sources"]):
            out["active_claims_supported_only_by_agent_reports"] = rows_to_dicts(conn.execute("""
                SELECT c.claim_id, c.classification, c.visibility, c.claim_text
                FROM claims c
                JOIN claim_evidence ce ON ce.claim_id=c.claim_id AND ce.status='active' AND ce.evidence_role='supports'
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id AND ch.status='active'
                JOIN sources s ON s.source_id=ch.source_id AND s.status='active'
                WHERE c.status='active' AND c.classification <> 'OpenQuestion'
                GROUP BY c.claim_id
                HAVING SUM(CASE WHEN s.source_type <> 'agent_report' THEN 1 ELSE 0 END)=0
                ORDER BY c.updated_at DESC LIMIT ?
            """, (limit,)).fetchall())
        if "interface_tags_json" in table_columns(conn, "claims"):
            out["local_claims_with_cross_role_tags"] = rows_to_dicts(conn.execute("""
                SELECT claim_id, classification, visibility, interface_tags_json, claim_text
                FROM claims
                WHERE status='active' AND visibility='local'
                  AND (
                    interface_tags_json LIKE '%"register"%'
                    OR interface_tags_json LIKE '%"interrupt"%'
                    OR interface_tags_json LIKE '%"build"%'
                    OR interface_tags_json LIKE '%"simulation"%'
                    OR interface_tags_json LIKE '%"protocol"%'
                    OR interface_tags_json LIKE '%"timing"%'
                  )
                ORDER BY updated_at DESC LIMIT ?
            """, (limit,)).fetchall())
    if table_exists(conn, "aliases"):
        out["alias_collisions"] = rows_to_dicts(conn.execute("""
            SELECT normalized_alias, COUNT(DISTINCT node_id) AS node_count, GROUP_CONCAT(DISTINCT node_id) AS node_ids
            FROM aliases
            GROUP BY normalized_alias
            HAVING COUNT(DISTINCT node_id) > 1
            ORDER BY node_count DESC, normalized_alias
            LIMIT ?
        """, (limit,)).fetchall())
    if table_exists(conn, "nodes") and table_exists(conn, "edges") and table_exists(conn, "claims"):
        out["orphan_nodes"] = rows_to_dicts(conn.execute("""
            SELECT n.node_id, n.node_type, n.canonical_name, n.visibility, n.status
            FROM nodes n
            WHERE n.status='active'
              AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.from_node_id=n.node_id OR e.to_node_id=n.node_id)
              AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.target_node_id=n.node_id)
            ORDER BY n.node_type, n.canonical_name
            LIMIT ?
        """, (limit,)).fetchall())
    if table_exists(conn, "edges") and table_exists(conn, "claims"):
        out["edges_without_claims"] = rows_to_dicts(conn.execute("""
            SELECT e.edge_id, e.from_node_id, e.relation, e.to_node_id, e.visibility, e.summary
            FROM edges e
            WHERE e.status='active' AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.target_edge_id=e.edge_id)
            ORDER BY e.relation, e.from_node_id
            LIMIT ?
        """, (limit,)).fetchall())
    if table_exists(conn, "nodes") and table_exists(conn, "edges"):
        out["high_degree_nodes"] = rows_to_dicts(conn.execute("""
            SELECT n.node_id, n.node_type, n.canonical_name, COUNT(e.edge_id) AS active_edge_count
            FROM nodes n JOIN edges e
              ON e.status='active' AND (e.from_node_id=n.node_id OR e.to_node_id=n.node_id)
            WHERE n.status='active'
            GROUP BY n.node_id
            HAVING COUNT(e.edge_id) >= 8
            ORDER BY active_edge_count DESC, n.canonical_name
            LIMIT ?
        """, (limit,)).fetchall())
    return out


def tail_file(path: Path, max_lines: int = 500) -> str:
    if not path.exists():
        return "Log file not found: %s" % path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as exc:
        return "Failed to read %s: %s" % (path, exc)


def format_rows(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)


def parse_config_text(text: str) -> Any:
    """Parse config text for semantic comparison when PyYAML is available."""
    if yaml is None:
        return text
    try:
        return yaml.safe_load(text) if text.strip() else {}
    except Exception:
        return text


def merge_config_defaults(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return backend-style deep merge of defaults plus project overrides."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def effective_config_from_text(text: str) -> Any:
    """Return the config that the backend effectively sees after default merge."""
    parsed = parse_config_text(text)
    if not isinstance(parsed, dict):
        return parsed
    if not isinstance(DOCGRAPH_DEFAULT_CONFIG, dict) or not DOCGRAPH_DEFAULT_CONFIG:
        return parsed
    return merge_config_defaults(DOCGRAPH_DEFAULT_CONFIG, parsed)


def config_value_source(configured_value: Any) -> str:
    """Return whether a structured value came from YAML or backend defaults."""
    return "default" if configured_value is MISSING_CONFIG_VALUE else "file"


def validate_config_text(text: str) -> tuple[bool, str]:
    """Validate config text enough for a GUI save warning."""
    if yaml is None:
        return True, "PyYAML is not installed; saved without YAML syntax validation."
    try:
        parsed = yaml.safe_load(text) if text.strip() else {}
    except Exception as exc:
        return False, str(exc)
    if parsed is not None and not isinstance(parsed, dict):
        return False, "Top-level config must be a YAML mapping/object."
    return True, "YAML syntax OK."


def config_change_requires_relaunch(active_text: str | None, disk_text: str) -> bool:
    """Any semantic config change requires a long-running backend/MCP restart."""
    if active_text is None:
        return False
    return effective_config_from_text(active_text) != effective_config_from_text(disk_text)


def config_leaf_type(value: Any) -> str:
    """Return the UI edit kind for a config leaf value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "text"


def config_value_to_text(value: Any) -> str:
    """Render a config value for the structured table."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def config_path_description(path: str) -> str:
    """Return hover/help text for a config path."""
    if path in CONFIG_FIELD_DESCRIPTIONS:
        return CONFIG_FIELD_DESCRIPTIONS[path]
    parts = path.split(".")
    for idx in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:idx])
        if parent in CONFIG_FIELD_DESCRIPTIONS:
            return CONFIG_FIELD_DESCRIPTIONS[parent]
    if path.endswith(".enabled"):
        return "Boolean switch for this section."
    if path.endswith(".model"):
        return "Model ID or local filesystem path."
    if path.endswith(".provider"):
        return "Provider adapter name."
    if path.endswith(".audience_roles") or path.endswith(".audience_roles_json"):
        return "Roles that should see or use this knowledge."
    if ".budgets." in path:
        return "Context-size limit for the selected budget."
    if ".ranking." in path:
        return "Ranking weight; increasing it gives this signal more influence."
    return "Project configuration value. Hover parent sections for broader context."


def config_section_title(path: str) -> str:
    """Humanize a config section/key for display."""
    name = path.split(".")[-1] if path else "config"
    return name.replace("_", " ").title()


def parse_config_value(text: str, original_value: Any) -> Any:
    """Parse one structured-table value back to the original value family."""
    kind = config_leaf_type(original_value)
    stripped = text.strip()
    if kind == "bool":
        return stripped.lower() in {"1", "true", "yes", "on", "enabled", "checked"}
    if kind == "int":
        return int(stripped)
    if kind == "float":
        return float(stripped)
    if kind == "null":
        if stripped.lower() in {"", "null", "none", "~"}:
            return None
        return stripped
    if kind in {"list", "object"}:
        if yaml is not None:
            parsed = yaml.safe_load(stripped)
        else:
            parsed = json.loads(stripped)
        if kind == "list" and not isinstance(parsed, list):
            raise ValueError("Expected a list value.")
        if kind == "object" and not isinstance(parsed, dict):
            raise ValueError("Expected an object value.")
        return parsed
    return text


def flatten_config_values(value: Any, prefix: str = "", configured_value: Any = MISSING_CONFIG_VALUE) -> list[dict[str, Any]]:
    """Flatten a YAML config object into leaf rows for the structured editor."""
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            configured_child = configured_value.get(key, MISSING_CONFIG_VALUE) if isinstance(configured_value, dict) and key in configured_value else MISSING_CONFIG_VALUE
            rows.extend(flatten_config_values(child, child_prefix, configured_child))
        return rows
    kind = config_leaf_type(value)
    return [{
        "path": prefix,
        "type": kind,
        "value": value,
        "text": config_value_to_text(value),
        "source": config_value_source(configured_value),
        "requires_relaunch": "yes",
        "description": config_path_description(prefix),
    }]


def config_structured_rows(text: str) -> list[dict[str, Any]]:
    """Return structured-editor rows from config text."""
    parsed = parse_config_text(text)
    effective = effective_config_from_text(text)
    if not isinstance(parsed, dict) or not isinstance(effective, dict):
        return []
    return flatten_config_values(effective, configured_value=parsed)


def set_config_path_value(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a dotted config path in a nested mapping."""
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return
    current: dict[str, Any] = root
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def dump_config_text(config: dict[str, Any]) -> str:
    """Serialize config after structured edits."""
    if yaml is not None:
        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    return json.dumps(config, indent=2, ensure_ascii=False) + "\n"


def config_tree_leaf_rows(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Return all leaf rows under a config subtree."""
    return flatten_config_values(value, prefix)


def run_gui(root_arg: Optional[str] = None, db_arg: Optional[str] = None, max_rows: int = DEFAULT_MAX_ROWS) -> int:
    try:
        from PyQt6.QtCore import QProcess, Qt
        from PyQt6.QtGui import QAction, QBrush, QColor, QPen
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGraphicsScene,
            QGraphicsView,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QSplitter,
            QStyledItemDelegate,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextEdit,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional GUI dependency
        print("PyQt6 is required for the GUI. Install with: python -m pip install PyQt6", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    class ConfigValueDelegate(QStyledItemDelegate):
        """Restrict inline editing to value cells in the config tree."""

        def createEditor(self, parent: Any, option: Any, index: Any) -> Any:  # noqa: N802
            if index.column() != 2:
                return None
            if index.data(Qt.ItemDataRole.UserRole) is None:
                return None
            return super().createEditor(parent, option, index)

    class MainWindow(QMainWindow):
        def __init__(self, root_path: Optional[str], db_path: Optional[str]) -> None:
            super().__init__()
            self.setWindowTitle("DocGraph Inspector - live retrieval is explicit")
            self.max_rows = max_rows
            self.paths: Optional[ProjectPaths] = None
            self.conn: Optional[sqlite3.Connection] = None
            self.live_process: Optional[QProcess] = None
            self.config_disk_text = ""
            self.config_active_text: Optional[str] = None
            self.config_editor_loading = False
            self.config_table_loading = False
            self.config_table_applying = False
            self.config_dirty = False
            self.config_relaunch_required = False
            self.status = QLabel("No project loaded")
            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)
            self._build_actions()
            self._build_tabs()
            self.statusBar().addPermanentWidget(self.status)
            if db_path:
                self.load_db(Path(db_path))
            elif root_path:
                self.load_root(Path(root_path))
            else:
                self.show_empty_dashboard()

        def _build_actions(self) -> None:
            menu = self.menuBar().addMenu("Project")
            open_action = QAction("Open project root...", self)
            open_action.triggered.connect(self.choose_root)
            menu.addAction(open_action)
            load_db_action = QAction("Load DB...", self)
            load_db_action.triggered.connect(self.choose_db)
            menu.addAction(load_db_action)
            refresh_action = QAction("Refresh", self)
            refresh_action.triggered.connect(self.refresh_all)
            menu.addAction(refresh_action)

        def _build_tabs(self) -> None:
            self.overview_text = QTextEdit(readOnly=True)
            overview_tab = QWidget()
            overview_layout = QVBoxLayout(overview_tab)
            overview_buttons = QHBoxLayout()
            overview_root_btn = QPushButton("Open Project Root")
            overview_root_btn.clicked.connect(self.choose_root)
            overview_db_btn = QPushButton("Load DB")
            overview_db_btn.clicked.connect(self.choose_db)
            overview_refresh_btn = QPushButton("Refresh")
            overview_refresh_btn.clicked.connect(self.refresh_all)
            overview_buttons.addWidget(overview_root_btn)
            overview_buttons.addWidget(overview_db_btn)
            overview_buttons.addWidget(overview_refresh_btn)
            overview_buttons.addStretch(1)
            overview_layout.addLayout(overview_buttons)
            overview_layout.addWidget(self.overview_text)
            self.tabs.addTab(overview_tab, "Overview")

            self.node_search = QLineEdit()
            self.node_type = QComboBox()
            self.node_visibility = QComboBox()
            self.node_role = QLineEdit()
            self.node_tag = QLineEdit()
            self.node_table = QTableWidget()
            self.node_details_text = QTextEdit(readOnly=True)
            node_tab = QWidget()
            node_layout = QVBoxLayout(node_tab)
            node_filters = QHBoxLayout()
            node_filters.addWidget(QLabel("Search")); node_filters.addWidget(self.node_search)
            node_filters.addWidget(QLabel("Type")); node_filters.addWidget(self.node_type)
            node_filters.addWidget(QLabel("Visibility")); node_filters.addWidget(self.node_visibility)
            node_filters.addWidget(QLabel("Role")); node_filters.addWidget(self.node_role)
            node_filters.addWidget(QLabel("Tag")); node_filters.addWidget(self.node_tag)
            node_btn = QPushButton("Find")
            node_btn.clicked.connect(self.refresh_nodes)
            node_filters.addWidget(node_btn)
            node_layout.addLayout(node_filters)
            node_split = QSplitter(Qt.Orientation.Vertical)
            node_split.addWidget(self.node_table)
            node_split.addWidget(self.node_details_text)
            node_layout.addWidget(node_split)
            self.node_table.itemSelectionChanged.connect(self.show_selected_node)
            self.tabs.addTab(node_tab, "Nodes")

            self.edge_search = QLineEdit(); self.edge_relation = QComboBox(); self.edge_visibility = QComboBox(); self.edge_role = QLineEdit()
            self.edge_table = QTableWidget()
            edge_tab = QWidget(); edge_layout = QVBoxLayout(edge_tab); edge_filters = QHBoxLayout()
            edge_filters.addWidget(QLabel("Search")); edge_filters.addWidget(self.edge_search)
            edge_filters.addWidget(QLabel("Relation")); edge_filters.addWidget(self.edge_relation)
            edge_filters.addWidget(QLabel("Visibility")); edge_filters.addWidget(self.edge_visibility)
            edge_filters.addWidget(QLabel("Role")); edge_filters.addWidget(self.edge_role)
            edge_btn = QPushButton("Find"); edge_btn.clicked.connect(self.refresh_edges)
            edge_filters.addWidget(edge_btn); edge_layout.addLayout(edge_filters); edge_layout.addWidget(self.edge_table)
            self.tabs.addTab(edge_tab, "Edges")

            self.claim_search = QLineEdit(); self.claim_status = QComboBox(); self.claim_class = QComboBox(); self.claim_visibility = QComboBox(); self.claim_role = QLineEdit()
            self.claim_table = QTableWidget(); self.claim_details_text = QTextEdit(readOnly=True)
            claim_tab = QWidget(); claim_layout = QVBoxLayout(claim_tab); claim_filters = QHBoxLayout()
            claim_filters.addWidget(QLabel("Search")); claim_filters.addWidget(self.claim_search)
            claim_filters.addWidget(QLabel("Status")); claim_filters.addWidget(self.claim_status)
            claim_filters.addWidget(QLabel("Class")); claim_filters.addWidget(self.claim_class)
            claim_filters.addWidget(QLabel("Visibility")); claim_filters.addWidget(self.claim_visibility)
            claim_filters.addWidget(QLabel("Role")); claim_filters.addWidget(self.claim_role)
            claim_btn = QPushButton("Find"); claim_btn.clicked.connect(self.refresh_claims)
            claim_filters.addWidget(claim_btn); claim_layout.addLayout(claim_filters)
            claim_split = QSplitter(Qt.Orientation.Vertical); claim_split.addWidget(self.claim_table); claim_split.addWidget(self.claim_details_text)
            claim_layout.addWidget(claim_split)
            self.claim_table.itemSelectionChanged.connect(self.show_selected_claim)
            self.tabs.addTab(claim_tab, "Claims")

            self.live_query = QLineEdit(); self.live_anchors = QLineEdit(); self.live_role = QComboBox(); self.live_intent = QLineEdit()
            self.live_mode = QComboBox(); self.live_budget = QComboBox(); self.live_result_text = QTextEdit(readOnly=True)
            self.live_role.addItems(["", "firmware", "rtl", "test_debug", "build", "architecture"])
            self.live_mode.addItems(["local", "global", "bridge", "hybrid", "mix"]); self.live_mode.setCurrentText("local")
            self.live_budget.addItems(["small", "medium", "large"])
            live_tab = QWidget(); live_layout = QVBoxLayout(live_tab); live_form = QFormLayout()
            live_form.addRow("Query", self.live_query)
            live_form.addRow("Anchors (comma-separated)", self.live_anchors)
            live_form.addRow("Role", self.live_role); live_form.addRow("Intent", self.live_intent)
            live_form.addRow("Mode", self.live_mode); live_form.addRow("Budget", self.live_budget)
            live_layout.addLayout(live_form)
            live_warning = QLabel(
                "Explicit action: runs the real backend against the open database. "
                "It records retrieval telemetry, may migrate retrieval-support schema and update embedding cache, "
                "but does not create or edit nodes, edges, claims, or evidence."
            )
            live_warning.setWordWrap(True)
            live_warning.setStyleSheet("color: #92400e; padding: 8px; background: #fff7ed; border: 1px solid #fdba74;")
            live_layout.addWidget(live_warning)
            live_buttons = QHBoxLayout()
            self.live_run_btn = QPushButton("Run Live Retrieval")
            self.live_run_btn.clicked.connect(self.run_live_retrieval)
            live_to_trace_btn = QPushButton("Open Last Trace")
            live_to_trace_btn.clicked.connect(self.open_last_live_trace)
            live_buttons.addWidget(self.live_run_btn); live_buttons.addWidget(live_to_trace_btn)
            live_layout.addLayout(live_buttons); live_layout.addWidget(self.live_result_text)
            self.tabs.addTab(live_tab, "Live Retrieval")

            self.config_editor = QTextEdit()
            self.config_status = QLabel("No project loaded.")
            self.config_status.setWordWrap(True)
            self.config_save_btn = QPushButton("Save Config")
            self.config_reload_btn = QPushButton("Reload From Disk")
            self.config_relaunch_btn = QPushButton("Relaunch / Reload Backend")
            self.config_struct_refresh_btn = QPushButton("Refresh Structured View")
            self.config_tree = QTreeWidget()
            self.config_tree.setColumnCount(6)
            self.config_tree.setHeaderLabels(["Section / Setting", "Type", "Value", "Source", "Restart", "Help"])
            self.config_tree.setAlternatingRowColors(True)
            self.config_tree.setUniformRowHeights(False)
            self.config_tree.setItemDelegate(ConfigValueDelegate(self.config_tree))
            self.config_tree.setStyleSheet("""
                QTreeWidget {
                    background: #0f172a;
                    alternate-background-color: #111827;
                    color: #e5e7eb;
                    border: 1px solid #334155;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                }
                QTreeWidget::item {
                    color: #e5e7eb;
                    padding: 4px;
                }
                QTreeWidget::item:selected {
                    background: #2563eb;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: #020617;
                    color: #e5e7eb;
                    border: 1px solid #374151;
                    padding: 5px;
                    font-weight: 700;
                }
            """)
            self.config_editor.setStyleSheet("""
                QTextEdit {
                    background: #0f172a;
                    color: #e5e7eb;
                    border: 1px solid #334155;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                    font-family: Menlo, Consolas, monospace;
                    font-size: 12px;
                }
            """)
            self.config_save_btn.clicked.connect(self.save_config)
            self.config_reload_btn.clicked.connect(lambda: self.refresh_config(force=True))
            self.config_relaunch_btn.clicked.connect(self.relaunch_after_config_change)
            self.config_struct_refresh_btn.clicked.connect(self.refresh_config_table_from_editor)
            self.config_editor.textChanged.connect(self.config_text_changed)
            self.config_tree.itemChanged.connect(self.config_table_item_changed)
            config_tab = QWidget(); config_layout = QVBoxLayout(config_tab)
            config_info = QLabel(
                "Edit docgraph.config.yaml for the open project. The structured tree shows the effective backend config: file values plus backend defaults. "
                "The Source column shows whether a value is set in YAML or inherited from defaults. Editing a default value materializes that path into the YAML text. "
                "Editing structured values rewrites the YAML text, so use the raw editor for comments/advanced formatting. Saved changes affect the next GUI Live Retrieval "
                "because it starts a new backend subprocess. A persistent MCP server keeps its config in memory, so semantic config changes require relaunching that server."
            )
            config_info.setWordWrap(True)
            config_info.setStyleSheet("color: #1f2937; padding: 8px; background: #eff6ff; border: 1px solid #93c5fd;")
            config_buttons = QHBoxLayout()
            config_buttons.addWidget(self.config_save_btn)
            config_buttons.addWidget(self.config_reload_btn)
            config_buttons.addWidget(self.config_relaunch_btn)
            config_buttons.addWidget(self.config_struct_refresh_btn)
            config_split = QSplitter(Qt.Orientation.Vertical)
            config_split.addWidget(self.config_tree)
            config_split.addWidget(self.config_editor)
            config_layout.addWidget(config_info)
            config_layout.addLayout(config_buttons)
            config_layout.addWidget(self.config_status)
            config_layout.addWidget(config_split)
            self.tabs.insertTab(0, config_tab, "Config")
            self.tabs.setCurrentIndex(0)

            self.retrieval_search = QLineEdit(); self.retrieval_mode = QComboBox(); self.retrieval_role = QLineEdit()
            self.retrieval_table = QTableWidget(); self.retrieval_details_text = QTextEdit(readOnly=True)
            retrieval_tab = QWidget(); retrieval_layout = QVBoxLayout(retrieval_tab); retrieval_filters = QHBoxLayout()
            retrieval_filters.addWidget(QLabel("Query / ID")); retrieval_filters.addWidget(self.retrieval_search)
            retrieval_filters.addWidget(QLabel("Mode")); retrieval_filters.addWidget(self.retrieval_mode)
            retrieval_filters.addWidget(QLabel("Role")); retrieval_filters.addWidget(self.retrieval_role)
            retrieval_btn = QPushButton("Find"); retrieval_btn.clicked.connect(self.refresh_retrieval_runs)
            graph_run_btn = QPushButton("Graph first anchor"); graph_run_btn.clicked.connect(self.graph_selected_retrieval_anchor)
            retrieval_filters.addWidget(retrieval_btn); retrieval_filters.addWidget(graph_run_btn); retrieval_layout.addLayout(retrieval_filters)
            retrieval_split = QSplitter(Qt.Orientation.Vertical); retrieval_split.addWidget(self.retrieval_table); retrieval_split.addWidget(self.retrieval_details_text)
            retrieval_layout.addWidget(retrieval_split)
            self.retrieval_table.itemSelectionChanged.connect(self.show_selected_retrieval_run)
            self.retrieval_tab = retrieval_tab
            self.tabs.addTab(retrieval_tab, "Retrieval")

            self.graph_node_id = QLineEdit(); self.graph_hops = QSpinBox(); self.graph_hops.setRange(1, 3); self.graph_hops.setValue(1)
            self.graph_cap = QSpinBox(); self.graph_cap.setRange(5, 100); self.graph_cap.setValue(25)
            self.graph_scene = QGraphicsScene(); self.graph_view = QGraphicsView(self.graph_scene)
            self.graph_summary = QLabel("Enter a node_id or select a retrieval anchor.")
            graph_tab = QWidget(); graph_layout = QVBoxLayout(graph_tab); graph_controls = QHBoxLayout()
            graph_controls.addWidget(QLabel("Center node_id")); graph_controls.addWidget(self.graph_node_id)
            graph_controls.addWidget(QLabel("Hops")); graph_controls.addWidget(self.graph_hops)
            graph_controls.addWidget(QLabel("Node cap")); graph_controls.addWidget(self.graph_cap)
            graph_btn = QPushButton("Draw neighborhood"); graph_btn.clicked.connect(self.refresh_graph)
            graph_controls.addWidget(graph_btn)
            graph_layout.addLayout(graph_controls); graph_layout.addWidget(self.graph_summary); graph_layout.addWidget(self.graph_view)
            self.graph_tab = graph_tab
            self.tabs.addTab(graph_tab, "Focused Graph")

            self.bridge_a = QLineEdit(); self.bridge_b = QLineEdit(); self.bridge_depth = QSpinBox(); self.bridge_depth.setRange(1, 8); self.bridge_depth.setValue(4)
            self.bridge_text = QTextEdit(readOnly=True)
            bridge_tab = QWidget(); bridge_layout = QVBoxLayout(bridge_tab); bridge_form = QFormLayout()
            bridge_form.addRow("From node_id", self.bridge_a); bridge_form.addRow("To node_id", self.bridge_b); bridge_form.addRow("Max depth", self.bridge_depth)
            bridge_btn = QPushButton("Find bridge paths"); bridge_btn.clicked.connect(self.refresh_bridge)
            bridge_layout.addLayout(bridge_form); bridge_layout.addWidget(bridge_btn); bridge_layout.addWidget(self.bridge_text)
            self.tabs.addTab(bridge_tab, "Bridge")

            self.quality_combo = QComboBox(); self.quality_table = QTableWidget()
            quality_tab = QWidget(); quality_layout = QVBoxLayout(quality_tab); quality_top = QHBoxLayout()
            quality_top.addWidget(QLabel("Check")); quality_top.addWidget(self.quality_combo)
            quality_btn = QPushButton("Run"); quality_btn.clicked.connect(self.refresh_quality)
            quality_top.addWidget(quality_btn); quality_layout.addLayout(quality_top); quality_layout.addWidget(self.quality_table)
            self.tabs.addTab(quality_tab, "Quality")

            self.log_text = QTextEdit(readOnly=True)
            log_tab = QWidget(); log_layout = QVBoxLayout(log_tab); log_btn = QPushButton("Refresh log tail")
            log_btn.clicked.connect(self.refresh_log)
            log_layout.addWidget(log_btn); log_layout.addWidget(self.log_text)
            self.tabs.addTab(log_tab, "Logs")

        def choose_root(self) -> None:
            directory = QFileDialog.getExistingDirectory(self, "Select project root or DocGraph root")
            if directory:
                self.load_root(Path(directory))

        def choose_db(self) -> None:
            filename, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Select DocGraph SQLite database",
                "",
                "SQLite databases (*.sqlite *.sqlite3 *.db);;All files (*)",
            )
            if filename:
                self.load_db(Path(filename))

        def load_root(self, root: Path) -> None:
            paths = detect_project_paths(root)
            if not paths.db_path.exists():
                QMessageBox.critical(self, "DocGraph DB not found", "Expected DB at:\n%s" % paths.db_path)
                self.status.setText("DB not found: %s" % paths.db_path)
                return
            self.load_paths(paths)

        def load_db(self, db_path: Path) -> None:
            paths = detect_project_paths_from_db(db_path)
            if not paths.db_path.exists():
                QMessageBox.critical(self, "DocGraph DB not found", "Selected DB does not exist:\n%s" % paths.db_path)
                self.status.setText("DB not found: %s" % paths.db_path)
                return
            self.load_paths(paths)

        def load_paths(self, paths: ProjectPaths) -> None:
            project_changed = self.paths is None or self.paths.root != paths.root or self.paths.db_path != paths.db_path
            try:
                if self.conn is not None:
                    self.conn.close()
                self.conn = open_readonly_connection(paths.db_path)
                self.paths = paths
                if project_changed:
                    self.config_active_text = None
                    self.config_disk_text = ""
                    self.config_dirty = False
                    self.config_relaunch_required = False
                self.status.setText("Loaded %s" % paths.db_path)
                self.refresh_all()
            except Exception as exc:
                QMessageBox.critical(self, "Failed to open DB", str(exc))
                self.status.setText("Failed to open DB")

        def require_conn(self) -> sqlite3.Connection:
            if self.conn is None:
                raise RuntimeError("No DB loaded")
            return self.conn

        def show_empty_dashboard(self) -> None:
            self.status.setText("No DB loaded")
            self.overview_text.setPlainText(
                "No database loaded.\n\n"
                "Use the Load DB button to open any DocGraph SQLite file for read-only inspection.\n"
                "Use Open Project Root when you want Config, Logs, and Live Retrieval against the configured DocGraph DB."
            )
            self.update_config_status("No project loaded.")

        def refresh_all(self) -> None:
            if self.conn is None:
                self.show_empty_dashboard()
                return
            self.refresh_overview(); self.refresh_config(); self.refresh_filters(); self.refresh_nodes(); self.refresh_edges(); self.refresh_claims(); self.refresh_retrieval_runs(); self.refresh_quality_keys(); self.refresh_log()

        def refresh_overview(self) -> None:
            conn = self.require_conn()
            counts = overview_counts(conn)
            lines = ["Project root: %s" % (self.paths.root if self.paths else ""), "DB: %s" % (self.paths.db_path if self.paths else ""), "", "Counts:"]
            lines.extend("  %-24s %s" % (k, v) for k, v in sorted(counts.items()))
            if self.paths:
                live_status = "available" if supports_live_retrieval(self.paths) else "unavailable without docgraph.config.yaml"
                lines += [
                    "",
                    "Config: %s (%s)" % (self.paths.config_path, "exists" if self.paths.config_path.exists() else "missing"),
                    "Log: %s (%s)" % (self.paths.log_path, "exists" if self.paths.log_path.exists() else "missing"),
                    "Live Retrieval: %s" % live_status,
                ]
            self.overview_text.setPlainText("\n".join(lines))

        def refresh_config(self, force: bool = False) -> None:
            if not self.paths:
                self.config_editor.setPlainText("")
                self.config_tree.clear()
                self.config_status.setText("No project loaded.")
                return
            if self.config_dirty and not force:
                self.update_config_status("Unsaved editor changes; skipped disk reload.")
                return
            path = self.paths.config_path
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    self.update_config_status("Failed to read config: %s" % exc)
                    return
            else:
                text = ""
            self.config_disk_text = text
            if self.config_active_text is None:
                self.config_active_text = text
            self.config_editor_loading = True
            self.config_editor.setPlainText(text)
            self.config_editor_loading = False
            self.refresh_config_table_from_editor()
            self.config_dirty = False
            self.update_config_status()

        def config_text_changed(self) -> None:
            if self.config_editor_loading:
                return
            self.config_dirty = self.config_editor.toPlainText() != self.config_disk_text
            self.update_config_status()

        def refresh_config_table_from_editor(self, focus_path: str | None = None) -> None:
            self.config_table_loading = True
            try:
                had_tree = self.config_tree.topLevelItemCount() > 0
                expanded_paths = self.config_expanded_paths()
                focus_path = focus_path or self.current_config_tree_path()
                text = self.config_editor.toPlainText()
                parsed = parse_config_text(text)
                effective = effective_config_from_text(text)
                self.config_tree.clear()
                self.config_tree.setHeaderLabels(["Section / Setting", "Type", "Value", "Source", "Restart", "Help"])
                if not isinstance(parsed, dict) or not isinstance(effective, dict):
                    msg = "Structured view unavailable: install PyYAML or fix YAML syntax." if yaml is None else "Structured view unavailable for invalid/non-mapping YAML."
                    item = QTreeWidgetItem([msg, "", "", "", "", ""])
                    item.setToolTip(0, msg)
                    self.config_tree.addTopLevelItem(item)
                    return
                for key, value in effective.items():
                    configured_value = parsed.get(key, MISSING_CONFIG_VALUE)
                    self.add_config_tree_item(None, str(key), value, str(key), configured_value)
                self.restore_config_tree_state(expanded_paths, focus_path, default_expand=not had_tree)
                self.config_tree.resizeColumnToContents(0)
                self.config_tree.resizeColumnToContents(1)
                self.config_tree.resizeColumnToContents(2)
                self.config_tree.resizeColumnToContents(3)
                self.config_tree.resizeColumnToContents(4)
            finally:
                self.config_table_loading = False

        def add_config_tree_item(self, parent: QTreeWidgetItem | None, key: str, value: Any, path: str, configured_value: Any = MISSING_CONFIG_VALUE) -> QTreeWidgetItem:
            desc = config_path_description(path)
            source = config_value_source(configured_value)
            if isinstance(value, dict):
                item = QTreeWidgetItem([config_section_title(path), "section", "", source, "yes", desc])
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                item.setData(2, Qt.ItemDataRole.UserRole, None)
                item.setToolTip(0, "%s\n\n%s" % (path, desc))
                item.setToolTip(5, desc)
                for col in range(6):
                    item.setBackground(col, QBrush(QColor("#1e293b")))
                    item.setForeground(col, QBrush(QColor("#f8fafc")))
                    item.setToolTip(col, "%s\n\n%s" % (path, desc))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if parent is None:
                    self.config_tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                for child_key, child_value in value.items():
                    child_path = f"{path}.{child_key}"
                    configured_child = configured_value.get(child_key, MISSING_CONFIG_VALUE) if isinstance(configured_value, dict) and child_key in configured_value else MISSING_CONFIG_VALUE
                    self.add_config_tree_item(item, str(child_key), child_value, child_path, configured_child)
                return item

            kind = config_leaf_type(value)
            text_value = config_value_to_text(value)
            item = QTreeWidgetItem([str(key), kind, "enabled" if kind == "bool" and bool(value) else ("disabled" if kind == "bool" else text_value), source, "yes", desc])
            item.setData(0, Qt.ItemDataRole.UserRole, path)
            row = {
                "path": path,
                "type": kind,
                "value": value,
                "text": text_value,
                "source": source,
                "requires_relaunch": "yes",
                "description": desc,
            }
            item.setData(2, Qt.ItemDataRole.UserRole, row)
            for col in range(6):
                item.setBackground(col, QBrush(QColor("#0f172a")))
                item.setForeground(col, QBrush(QColor("#e5e7eb")))
                item.setToolTip(col, "%s\n\n%s\n\nSource: %s\nCurrent value: %s" % (path, desc, source, text_value))
            item.setForeground(1, QBrush(QColor("#93c5fd")))
            item.setForeground(3, QBrush(QColor("#86efac" if source == "file" else "#c4b5fd")))
            item.setForeground(4, QBrush(QColor("#fbbf24")))
            item.setForeground(5, QBrush(QColor("#cbd5e1")))
            if kind == "bool":
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(2, Qt.CheckState.Checked if bool(value) else Qt.CheckState.Unchecked)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            if parent is None:
                self.config_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            return item

        def iter_config_tree_items(self) -> list[QTreeWidgetItem]:
            items: list[QTreeWidgetItem] = []

            def visit(item: QTreeWidgetItem) -> None:
                items.append(item)
                for idx in range(item.childCount()):
                    visit(item.child(idx))

            for idx in range(self.config_tree.topLevelItemCount()):
                visit(self.config_tree.topLevelItem(idx))
            return items

        def config_expanded_paths(self) -> set[str]:
            paths: set[str] = set()
            for item in self.iter_config_tree_items():
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(path, str) and path and item.isExpanded():
                    paths.add(path)
            return paths

        def current_config_tree_path(self) -> str | None:
            item = self.config_tree.currentItem()
            if item is None:
                return None
            path = item.data(0, Qt.ItemDataRole.UserRole)
            return path if isinstance(path, str) and path else None

        def restore_config_tree_state(self, expanded_paths: set[str], focus_path: str | None, *, default_expand: bool) -> None:
            selected_item: QTreeWidgetItem | None = None
            for item in self.iter_config_tree_items():
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if not isinstance(path, str):
                    continue
                if item.childCount() > 0:
                    if default_expand:
                        item.setExpanded(item.parent() is None or item.parent().parent() is None)
                    else:
                        should_expand = path in expanded_paths or bool(focus_path and focus_path.startswith(path + "."))
                        item.setExpanded(should_expand)
                if focus_path and path == focus_path:
                    selected_item = item
            if selected_item is not None:
                self.config_tree.setCurrentItem(selected_item, 2)
                self.config_tree.scrollToItem(selected_item)

        def config_table_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
            if self.config_table_loading or self.config_table_applying or column != 2:
                return
            if item.data(2, Qt.ItemDataRole.UserRole) is None:
                return
            self.apply_config_item_to_editor(item)

        def iter_config_value_items(self) -> list[QTreeWidgetItem]:
            items: list[QTreeWidgetItem] = []

            def visit(item: QTreeWidgetItem) -> None:
                if isinstance(item.data(2, Qt.ItemDataRole.UserRole), dict):
                    items.append(item)
                for idx in range(item.childCount()):
                    visit(item.child(idx))

            for idx in range(self.config_tree.topLevelItemCount()):
                visit(self.config_tree.topLevelItem(idx))
            return items

        def apply_config_item_to_editor(self, item: QTreeWidgetItem) -> None:
            valid, validation_message = validate_config_text(self.config_editor.toPlainText())
            if not valid:
                self.update_config_status("Cannot apply structured config: %s" % validation_message)
                return
            parsed = parse_config_text(self.config_editor.toPlainText())
            if not isinstance(parsed, dict):
                self.update_config_status("Cannot apply structured config: top-level YAML is not a mapping.")
                return
            meta = item.data(2, Qt.ItemDataRole.UserRole)
            if not isinstance(meta, dict) or not meta.get("path"):
                return
            edited_path = str(meta["path"])
            self.config_table_applying = True
            try:
                if meta.get("type") == "bool":
                    value = item.checkState(2) == Qt.CheckState.Checked
                    item.setText(2, "enabled" if value else "disabled")
                else:
                    value = parse_config_value(item.text(2), meta.get("value"))
                set_config_path_value(parsed, edited_path, value)
            except Exception as exc:
                self.config_table_applying = False
                self.update_config_status("Structured config edit rejected: %s" % exc)
                return
            self.config_editor_loading = True
            try:
                self.config_editor.setPlainText(dump_config_text(parsed))
            finally:
                self.config_editor_loading = False
            self.config_dirty = self.config_editor.toPlainText() != self.config_disk_text
            try:
                self.update_config_item_after_edit(item, value)
            finally:
                self.config_table_applying = False
            self.update_config_status("Structured config value applied to YAML text.")

        def update_config_item_after_edit(self, item: QTreeWidgetItem, value: Any) -> None:
            """Update the edited row without rebuilding the tree inside itemChanged."""
            meta = item.data(2, Qt.ItemDataRole.UserRole)
            if not isinstance(meta, dict):
                return
            path = str(meta.get("path") or "")
            desc = str(meta.get("description") or config_path_description(path))
            kind = config_leaf_type(value)
            text_value = config_value_to_text(value)
            meta.update({
                "type": kind,
                "value": value,
                "text": text_value,
                "source": "file",
                "description": desc,
            })
            item.setData(2, Qt.ItemDataRole.UserRole, meta)
            item.setText(1, kind)
            item.setText(3, "file")
            item.setForeground(3, QBrush(QColor("#86efac")))
            if kind == "bool":
                item.setText(2, "enabled" if bool(value) else "disabled")
            else:
                item.setText(2, text_value)
            for col in range(6):
                item.setToolTip(col, "%s\n\n%s\n\nSource: file\nCurrent value: %s" % (path, desc, text_value))

        def apply_config_table_to_editor(self) -> None:
            """Materialize all structured values into YAML; kept for explicit/debug use."""
            valid, validation_message = validate_config_text(self.config_editor.toPlainText())
            if not valid:
                self.update_config_status("Cannot apply structured config: %s" % validation_message)
                return
            parsed = parse_config_text(self.config_editor.toPlainText())
            if not isinstance(parsed, dict):
                self.update_config_status("Cannot apply structured config: top-level YAML is not a mapping.")
                return
            self.config_table_applying = True
            try:
                for item in self.iter_config_value_items():
                    meta = item.data(2, Qt.ItemDataRole.UserRole)
                    if not isinstance(meta, dict) or not meta.get("path"):
                        continue
                    if meta.get("type") == "bool":
                        value = item.checkState(2) == Qt.CheckState.Checked
                        item.setText(2, "enabled" if value else "disabled")
                    else:
                        value = parse_config_value(item.text(2), meta.get("value"))
                    set_config_path_value(parsed, str(meta["path"]), value)
            except Exception as exc:
                self.config_table_applying = False
                self.update_config_status("Structured config edit rejected: %s" % exc)
                return
            self.config_editor_loading = True
            try:
                self.config_editor.setPlainText(dump_config_text(parsed))
            finally:
                self.config_editor_loading = False
                self.config_table_applying = False
            self.config_dirty = self.config_editor.toPlainText() != self.config_disk_text
            self.refresh_config_table_from_editor()
            self.update_config_status("Structured config values materialized into YAML text.")

        def update_config_status(self, prefix: str = "") -> None:
            if not self.paths:
                self.config_status.setText("No project loaded.")
                self.config_relaunch_btn.setEnabled(False)
                return
            editor_text = self.config_editor.toPlainText()
            valid, validation_message = validate_config_text(editor_text)
            disk_requires_relaunch = config_change_requires_relaunch(self.config_active_text, self.config_disk_text)
            editor_requires_relaunch = config_change_requires_relaunch(self.config_active_text, editor_text)
            self.config_relaunch_required = disk_requires_relaunch
            self.config_save_btn.setEnabled(self.config_dirty and valid)
            self.config_reload_btn.setEnabled(True)
            self.config_relaunch_btn.setEnabled(disk_requires_relaunch and not self.config_dirty)
            status_parts = [
                "Config: %s" % self.paths.config_path,
                "Unsaved changes: %s" % ("yes" if self.config_dirty else "no"),
                "Saved config requires backend relaunch: %s" % ("yes" if disk_requires_relaunch else "no"),
                "Editor would require relaunch after save: %s" % ("yes" if editor_requires_relaunch else "no"),
                "Validation: %s" % validation_message,
            ]
            if prefix:
                status_parts.insert(0, prefix)
            self.config_status.setText("\n".join(status_parts))

        def save_config(self) -> None:
            if not self.paths:
                self.update_config_status("Open a project first.")
                return
            text = self.config_editor.toPlainText()
            valid, validation_message = validate_config_text(text)
            if not valid:
                QMessageBox.critical(self, "Invalid config", validation_message)
                self.update_config_status("Save blocked.")
                return
            try:
                self.paths.config_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                QMessageBox.critical(self, "Failed to save config", str(exc))
                self.update_config_status("Save failed.")
                return
            self.config_disk_text = text
            self.config_dirty = False
            self.update_config_status("Config saved. Relaunch/reload is required for a persistent MCP backend.")

        def relaunch_after_config_change(self) -> None:
            if not self.paths:
                return
            if self.config_dirty:
                self.update_config_status("Save or discard config changes before relaunch/reload.")
                return
            if self.live_process is not None and self.live_process.state() != QProcess.ProcessState.NotRunning:
                self.live_process.terminate()
                if not self.live_process.waitForFinished(1500):
                    self.live_process.kill()
            self.config_active_text = self.config_disk_text
            self.config_relaunch_required = False
            root = self.paths.root
            self.load_root(root)
            self.update_config_status(
                "GUI reloaded the project config. If an external MCP server is running, restart that server separately."
            )

        def refresh_filters(self) -> None:
            conn = self.require_conn()
            self._set_combo(self.node_type, [""] + list_distinct(conn, "nodes", "node_type"))
            self._set_combo(self.node_visibility, ["", "local", "shared", "global", "shared_candidate"])
            self._set_combo(self.edge_relation, [""] + list_distinct(conn, "edges", "relation"))
            self._set_combo(self.edge_visibility, ["", "local", "shared", "global", "shared_candidate"])
            self._set_combo(self.claim_status, [""] + list_distinct(conn, "claims", "status"))
            self._set_combo(self.claim_class, [""] + list_distinct(conn, "claims", "classification"))
            self._set_combo(self.claim_visibility, ["", "local", "shared", "global", "shared_candidate"])
            self._set_combo(self.retrieval_mode, [""] + list_distinct(conn, "retrieval_runs", "mode"))

        def _set_combo(self, combo: QComboBox, values: list[str]) -> None:
            current = combo.currentText()
            combo.blockSignals(True); combo.clear(); combo.addItems(values); combo.blockSignals(False)
            if current in values:
                combo.setCurrentText(current)

        def fill_table(self, table: QTableWidget, rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> None:
            if columns is None:
                all_cols: list[str] = []
                for r in rows:
                    for k in r.keys():
                        if k not in all_cols:
                            all_cols.append(k)
                columns = all_cols
            table.setRowCount(len(rows)); table.setColumnCount(len(columns)); table.setHorizontalHeaderLabels(columns)
            for i, row in enumerate(rows):
                for j, col in enumerate(columns):
                    item = QTableWidgetItem(preview(row.get(col), 500))
                    item.setData(Qt.ItemDataRole.UserRole, row)
                    table.setItem(i, j, item)
            table.resizeColumnsToContents()

        def selected_row(self, table: QTableWidget) -> Optional[dict[str, Any]]:
            items = table.selectedItems()
            return items[0].data(Qt.ItemDataRole.UserRole) if items else None

        def refresh_nodes(self) -> None:
            rows = search_nodes(self.require_conn(), self.node_search.text(), self.node_type.currentText(), self.node_visibility.currentText(), self.node_role.text(), self.node_tag.text(), self.max_rows)
            self.fill_table(self.node_table, rows, ["node_id", "node_type", "canonical_name", "visibility", "finder_role", "audience_roles_json", "interface_tags_json", "summary", "status"])

        def show_selected_node(self) -> None:
            row = self.selected_row(self.node_table)
            if not row:
                return
            data = node_details(self.require_conn(), row["node_id"])
            self.node_details_text.setPlainText(format_rows([data]))
            self.graph_node_id.setText(row["node_id"])

        def refresh_edges(self) -> None:
            rows = search_edges(self.require_conn(), self.edge_search.text(), self.edge_relation.currentText(), self.edge_visibility.currentText(), self.edge_role.text(), self.max_rows)
            self.fill_table(self.edge_table, rows, ["edge_id", "from_node_id", "relation", "to_node_id", "visibility", "finder_role", "audience_roles_json", "interface_tags_json", "summary", "status"])

        def refresh_claims(self) -> None:
            rows = search_claims(self.require_conn(), self.claim_search.text(), self.claim_status.currentText(), self.claim_class.currentText(), self.claim_visibility.currentText(), self.claim_role.text(), self.max_rows)
            self.fill_table(self.claim_table, rows, ["claim_id", "classification", "confidence", "visibility", "target_node_id", "target_edge_id", "finder_role", "audience_roles_json", "claim_text", "status"])

        def show_selected_claim(self) -> None:
            row = self.selected_row(self.claim_table)
            if not row:
                return
            self.claim_details_text.setPlainText(format_claim_proof(claim_details(self.require_conn(), row["claim_id"])))

        def run_live_retrieval(self) -> None:
            if not self.paths:
                self.live_result_text.setPlainText("Open a project database first.")
                return
            if not supports_live_retrieval(self.paths):
                self.live_result_text.setPlainText(
                    "Live Retrieval requires a root with docgraph.config.yaml or .opencode/docgraph/docgraph.config.yaml. "
                    "This standalone DB can still be inspected read-only in the dashboard."
                )
                return
            if not self.live_query.text().strip() and not self.live_anchors.text().strip():
                self.live_result_text.setPlainText("Enter a query, one or more anchors, or both.")
                return
            if self.live_process is not None and self.live_process.state() != QProcess.ProcessState.NotRunning:
                self.live_result_text.setPlainText("A live retrieval request is already running.")
                return
            args = live_query_arguments(
                self.paths,
                query=self.live_query.text(),
                anchors=self.live_anchors.text(),
                role=self.live_role.currentText(),
                intent=self.live_intent.text(),
                mode=self.live_mode.currentText(),
                budget=self.live_budget.currentText(),
            )
            self.live_result_text.setPlainText(
                "Running real retrieval. Mix mode may load an embedding model and update embedding cache; wait for completion..."
            )
            self.live_run_btn.setEnabled(False)
            self.live_process = QProcess(self)
            self.live_process.setProgram(sys.executable)
            self.live_process.setArguments(args)
            self.live_process.finished.connect(self.live_retrieval_finished)
            self.live_process.errorOccurred.connect(self.live_retrieval_process_error)
            self.live_process.start()

        def live_retrieval_process_error(self, error: object) -> None:
            if error == QProcess.ProcessError.FailedToStart:
                self.live_run_btn.setEnabled(True)
                self.live_result_text.setPlainText(
                    "Live retrieval process could not start. Verify this Python can execute tools/docgraph_query.py."
                )

        def live_retrieval_finished(self, exit_code: int, _exit_status: object) -> None:
            if self.live_process is None:
                return
            stdout = bytes(self.live_process.readAllStandardOutput()).decode("utf-8", errors="replace")
            stderr = bytes(self.live_process.readAllStandardError()).decode("utf-8", errors="replace")
            self.live_run_btn.setEnabled(True)
            payload: dict[str, Any] | None = None
            for line in reversed(stdout.splitlines()):
                if line.startswith(LIVE_QUERY_RESULT_PREFIX):
                    try:
                        payload = json.loads(line[len(LIVE_QUERY_RESULT_PREFIX):])
                    except json.JSONDecodeError:
                        payload = None
                    break
            if not payload or not payload.get("ok"):
                error = (payload or {}).get("error") or stderr.strip() or stdout.strip() or ("Exit code %s" % exit_code)
                self.live_result_text.setPlainText("Live retrieval failed:\n%s" % error)
                return
            run_id = str(payload.get("run_id") or "")
            self.live_result_text.setPlainText(
                "Stored trace: %s\n\n%s" % (run_id or "<not recorded>", payload.get("markdown") or "")
            )
            if self.paths:
                self.load_paths(self.paths)
            if run_id:
                self.retrieval_search.setText(run_id)
                self.refresh_retrieval_runs()
                if self.retrieval_table.rowCount() > 0:
                    self.retrieval_table.selectRow(0)

        def open_last_live_trace(self) -> None:
            self.tabs.setCurrentWidget(self.retrieval_tab)

        def refresh_retrieval_runs(self) -> None:
            rows = search_retrieval_runs(self.require_conn(), self.retrieval_search.text(), self.retrieval_mode.currentText(), self.retrieval_role.text(), self.max_rows)
            self.fill_table(self.retrieval_table, rows, ["created_at", "run_id", "query", "mode", "role", "budget", "anchors_json", "result_summary_json"])

        def show_selected_retrieval_run(self) -> None:
            row = self.selected_row(self.retrieval_table)
            if not row:
                return
            self.retrieval_details_text.setPlainText(format_retrieval_trace(retrieval_run_details(self.require_conn(), row["run_id"])))

        def graph_selected_retrieval_anchor(self) -> None:
            row = self.selected_row(self.retrieval_table)
            if not row:
                self.retrieval_details_text.setPlainText("Select a retrieval run first.")
                return
            details = retrieval_run_details(self.require_conn(), row["run_id"])
            anchors = details.get("trace", {}).get("final", {}).get("anchor_ids") or details.get("anchors") or []
            if not anchors:
                self.retrieval_details_text.setPlainText("Selected retrieval run has no returned anchors.")
                return
            self.graph_node_id.setText(str(anchors[0]))
            self.refresh_graph()
            self.tabs.setCurrentWidget(self.graph_tab)

        def refresh_graph(self) -> None:
            node_id = self.graph_node_id.text().strip()
            graph = neighborhood_graph(self.require_conn(), node_id, self.graph_hops.value(), self.graph_cap.value())
            self.graph_scene.clear()
            nodes = graph["nodes"]
            edges = graph["edges"]
            if not nodes:
                self.graph_summary.setText("No active neighborhood found for node_id: %s" % (node_id or "<empty>"))
                return
            positions: dict[str, tuple[float, float]] = {node_id: (0.0, 0.0)}
            neighbors = [n for n in nodes if n.get("node_id") != node_id]
            radius = max(180.0, 45.0 * len(neighbors))
            for idx, node in enumerate(neighbors):
                angle = (2.0 * math.pi * idx / max(1, len(neighbors))) - (math.pi / 2.0)
                positions[str(node["node_id"])] = (radius * math.cos(angle), radius * math.sin(angle))
            for edge in edges:
                start = positions.get(str(edge.get("from_node_id")))
                end = positions.get(str(edge.get("to_node_id")))
                if not start or not end:
                    continue
                line_pen = QPen(QColor("#8795a1"), 2)
                if edge.get("status") != "active":
                    line_pen.setStyle(Qt.PenStyle.DashLine)
                self.graph_scene.addLine(start[0], start[1], end[0], end[1], line_pen)
                label = self.graph_scene.addText(str(edge.get("relation") or ""))
                label.setDefaultTextColor(QColor("#4b5563"))
                label.setPos((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            status_colors = {
                "active": "#1f9d76",
                "needs_review": "#d97706",
                "stale": "#d97706",
                "contradicted": "#c0392b",
                "superseded": "#697586",
                "retired": "#697586",
            }
            for node in nodes:
                nid = str(node.get("node_id"))
                x, y = positions[nid]
                fill = "#e8f5ee" if nid == node_id else "#f4f1ea"
                outline = status_colors.get(str(node.get("status")), "#697586")
                width = 4 if nid == node_id else 2
                self.graph_scene.addEllipse(x - 42, y - 28, 84, 56, QPen(QColor(outline), width), QBrush(QColor(fill)))
                text_item = self.graph_scene.addText("%s\n%s" % (node.get("canonical_name") or nid, node.get("node_type") or ""))
                text_item.setDefaultTextColor(QColor("#192a33"))
                text_item.setPos(x - 38, y - 21)
            self.graph_summary.setText("Center: %s    Nodes: %d    Edges: %d    View: active neighborhood only" % (node_id, len(nodes), len(edges)))
            self.graph_scene.setSceneRect(self.graph_scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))
            self.graph_view.fitInView(self.graph_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        def refresh_bridge(self) -> None:
            start = self.bridge_a.text().strip(); end = self.bridge_b.text().strip()
            if not start or not end:
                self.bridge_text.setPlainText("Enter both node IDs.")
                return
            try:
                paths = find_bridge_paths(self.require_conn(), start, end, self.bridge_depth.value(), self.max_rows)
                if not paths:
                    self.bridge_text.setPlainText("No active bridge path found within depth %s." % self.bridge_depth.value())
                else:
                    blocks = []
                    for idx, path in enumerate(paths, 1):
                        blocks.append("Path %d, length %d:\n%s" % (idx, len(path), format_bridge_path(path)))
                    self.bridge_text.setPlainText("\n\n".join(blocks))
            except Exception as exc:
                self.bridge_text.setPlainText("Bridge search failed: %s" % exc)

        def refresh_quality_keys(self) -> None:
            checks = sorted(quality_checks(self.require_conn(), self.max_rows).keys())
            self._set_combo(self.quality_combo, checks)
            self.refresh_quality()

        def refresh_quality(self) -> None:
            checks = quality_checks(self.require_conn(), self.max_rows)
            key = self.quality_combo.currentText()
            self.fill_table(self.quality_table, checks.get(key, []))

        def refresh_log(self) -> None:
            if not self.paths:
                return
            self.log_text.setPlainText(tail_file(self.paths.log_path, self.max_rows))

    app = QApplication(sys.argv)
    win = MainWindow(root_arg, db_arg)
    win.resize(1450, 900)
    win.show()
    return app.exec()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only PyQt6 GUI for DocGraph SQLite databases")
    parser.add_argument("--root", help="Project root or DocGraph root containing a config file")
    parser.add_argument("--db", help="SQLite DB file to inspect directly")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Maximum rows per table/query")
    args = parser.parse_args(argv)
    if args.root and args.db:
        parser.error("use --root or --db, not both")
    return run_gui(args.root, args.db, args.max_rows)


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())
