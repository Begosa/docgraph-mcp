from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from typing import Any, Callable, Optional, Tuple

try:
    import sqlite3
except ImportError:  # pragma: no cover - site-specific Python builds may omit stdlib sqlite3
    import pysqlite3 as sqlite3  # type: ignore

from .config import cfg_get, cfg_list
from .models import cosine_similarity
from .retrieval_types import ContextPacket, SemanticCandidates


class RetrievalFlow:
    """Role-aware retrieval orchestration extracted from ``DocGraphBackend``."""

    def __init__(
        self,
        backend: Any,
        *,
        fts_query: Callable[[str], str],
        extract_terms: Callable[[str], list[str]],
        id_factory: Callable[[str], str],
        ts_factory: Callable[[], str],
    ) -> None:
        """Bind backend access and injectable helpers used by retrieval paths."""
        self.backend = backend
        self._fts_query = fts_query
        self._extract_terms = extract_terms
        self._new_id = id_factory
        self._now_ts = ts_factory
        self.last_recorded_run_id: str | None = None

    def search(
        self,
        query: str,
        role: str | None = None,
        intent: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
        collect_trace: bool = False,
    ) -> dict[str, Any]:
        """Run layered lexical retrieval and return anchors plus ranked results.

        Results combine resolve hits, FTS/BM25 paths, and bounded one-hop
        expansion, then apply role visibility and ranking bonuses.
        """
        started = time.perf_counter()
        self.backend._log("debug", "search.start", query=query, role=role, intent=intent, limit=limit, include_stale=include_stale)
        fquery = self._fts_query(query)
        results: list[dict[str, Any]] = []
        anchor_scores: dict[str, float] = {}
        anchor_reasons: dict[str, list[str]] = {}

        def note_anchor(node_id: str, score: float, reason: str) -> None:
            previous = anchor_scores.get(node_id)
            if previous is None or score > previous:
                anchor_scores[node_id] = score
            reasons = anchor_reasons.setdefault(node_id, [])
            if reason not in reasons:
                reasons.append(reason)

        # exact/alias path
        for term in [query] + self._extract_terms(query)[: int(cfg_get(self.backend.config, "retrieval.max_extracted_terms", 12))]:
            res = self.backend.resolve(term, limit=5, include_stale=include_stale)["matches"]
            for m in res:
                nid = m["node_id"]
                node = self.backend._get_node(nid)
                if not node or not self.backend._row_visible_to_role(node, role):
                    continue
                score = m.get("score", 0) + self.backend._rank_weight("resolved_node_bonus", 20.0) + self.backend._role_node_bonus(nid, role)
                note_anchor(nid, score, f"resolve:{m.get('match_type')}")
                results.append({
                    "kind": "node",
                    "id": nid,
                    "score": score,
                    "reason": f"resolved from term '{term}' via {m.get('match_type')}",
                    "node": node,
                })

        # claims path
        if fquery:
            try:
                for r in self.backend.conn.execute(
                    """
                    SELECT c.*, bm25(claims_fts) AS rank
                    FROM claims_fts cf JOIN claims c ON c.claim_id=cf.claim_id
                    WHERE claims_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fquery, limit * 3),
                ).fetchall():
                    if not include_stale and r["status"] != "active":
                        continue
                    if not self.backend._claim_visible_to_role(r, role):
                        continue
                    score = self.backend._rank_weight("claim_fts", 70.0) - float(r["rank"]) + self.backend._role_claim_bonus(r, role)
                    target_nodes = {nid for nid in self.backend._claim_target_nodes(r) if self.backend._node_visible_to_role(nid, role)}
                    claim_target_anchor_bonus = self.backend._rank_weight("claim_target_anchor_bonus", -3.0)
                    for nid in target_nodes:
                        note_anchor(nid, score + claim_target_anchor_bonus, "claim_fts_target")
                    results.append({
                        "kind": "claim",
                        "id": r["claim_id"],
                        "score": score,
                        "reason": "claim text FTS/BM25 match",
                        "claim": dict(r),
                        "target_nodes": list(target_nodes),
                    })
            except sqlite3.OperationalError:
                pass

            # node summary/profile FTS path
            try:
                for r in self.backend.conn.execute(
                    """
                    SELECT n.*, bm25(nodes_fts) AS rank
                    FROM nodes_fts nf JOIN nodes n ON n.node_id=nf.node_id
                    WHERE nodes_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fquery, limit * 2),
                ).fetchall():
                    if not include_stale and r["status"] != "active":
                        continue
                    if not self.backend._row_visible_to_role(r, role):
                        continue
                    score = self.backend._rank_weight("node_fts", 55.0) - float(r["rank"]) + self.backend._role_node_bonus(r["node_id"], role) + self.backend._role_row_bonus(r, role)
                    note_anchor(r["node_id"], score, "node_fts")
                    results.append({
                        "kind": "node",
                        "id": r["node_id"],
                        "score": score,
                        "reason": "node name/summary FTS/BM25 match",
                        "node": dict(r),
                    })
            except sqlite3.OperationalError as exc:
                self.backend._log("debug", "search.node_fts.error", query=query, error_type=type(exc).__name__, error=str(exc))

            # edge relationship profile FTS path
            try:
                for r in self.backend.conn.execute(
                    """
                    SELECT e.*, bm25(edges_fts) AS rank
                    FROM edges_fts ef JOIN edges e ON e.edge_id=ef.edge_id
                    WHERE edges_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fquery, limit * 2),
                ).fetchall():
                    if not include_stale and r["status"] != "active":
                        continue
                    if not self.backend._row_visible_to_role(r, role):
                        continue
                    score = self.backend._rank_weight("edge_fts", 52.0) - float(r["rank"]) + self.backend._role_relation_bonus(r["relation"], role) + self.backend._role_row_bonus(r, role)
                    if self.backend._node_visible_to_role(r["from_node_id"], role):
                        note_anchor(r["from_node_id"], score + self.backend._rank_weight("edge_endpoint_anchor_bonus", -4.0), "edge_fts_endpoint")
                    if self.backend._node_visible_to_role(r["to_node_id"], role):
                        note_anchor(r["to_node_id"], score + self.backend._rank_weight("edge_endpoint_anchor_bonus", -4.0), "edge_fts_endpoint")
                    results.append({
                        "kind": "edge",
                        "id": r["edge_id"],
                        "score": score,
                        "reason": "edge relation/summary FTS/BM25 match",
                        "edge": self.backend._edge_with_names(dict(r)),
                    })
            except sqlite3.OperationalError as exc:
                self.backend._log("debug", "search.edge_fts.error", query=query, error_type=type(exc).__name__, error=str(exc))

            # chunks/evidence path
            try:
                for r in self.backend.conn.execute(
                    """
                    SELECT ch.*, bm25(chunks_fts) AS rank, s.uri, s.source_type
                    FROM chunks_fts f
                    JOIN chunks ch ON ch.chunk_id=f.chunk_id
                    JOIN sources s ON s.source_id=ch.source_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fquery, limit * 3),
                ).fetchall():
                    if not include_stale and r["status"] != "active":
                        continue
                    linked = self.backend._claims_for_chunk(r["chunk_id"], include_stale=include_stale, role=role)
                    target_nodes: set[str] = set()
                    for c in linked:
                        target_nodes.update(
                            nid for nid in self.backend._claim_target_nodes(c) if self.backend._node_visible_to_role(nid, role)
                        )
                    score = self.backend._rank_weight("chunk_fts", 45.0) - float(r["rank"]) + self.backend._role_nodes_bonus(target_nodes, role)
                    chunk_target_anchor_bonus = self.backend._rank_weight("chunk_target_anchor_bonus", -6.0)
                    for nid in target_nodes:
                        note_anchor(nid, score + chunk_target_anchor_bonus, "chunk_fts_linked_claim")
                    results.append({
                        "kind": "chunk",
                        "id": r["chunk_id"],
                        "score": score,
                        "reason": "raw evidence FTS/BM25 match" + (" linked to claim" if linked else " uncurated"),
                        "chunk": self.backend._short_chunk(dict(r)),
                        "linked_claims": [c["claim_id"] for c in linked],
                        "target_nodes": list(target_nodes),
                    })
            except sqlite3.OperationalError as exc:
                self.backend._log("debug", "search.chunk_fts.error", query=query, error_type=type(exc).__name__, error=str(exc))

        # one-hop graph expansion from anchors
        edge_hits = []
        expansion_seed_ids = [
            nid
            for nid, _score in sorted(
                anchor_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[: int(cfg_get(self.backend.config, "retrieval.max_anchor_expansion", 20))]
        ]
        for nid in expansion_seed_ids:
            for e in self.backend._edges_for_node(nid, include_stale=include_stale, limit=int(cfg_get(self.backend.config, "retrieval.neighbor_edges_per_anchor", 8))):
                edge_hits.append(e)
                other = e["to_node_id"] if e["from_node_id"] == nid else e["from_node_id"]
                if self.backend._node_visible_to_role(other, role):
                    note_anchor(
                        other,
                        anchor_scores.get(nid, 0.0)
                        + self.backend._rank_weight("expanded_anchor_step_bonus", -8.0)
                        + self.backend._role_relation_bonus(e.get("relation"), role),
                        f"one_hop:{nid}:{e.get('relation')}",
                    )
        for e in edge_hits[:limit * 2]:
            if not self.backend._row_visible_to_role(e, role):
                continue
            results.append({
                "kind": "edge",
                "id": e["edge_id"],
                "score": self.backend._rank_weight("edge_expansion", 35.0) + self.backend._role_relation_bonus(e.get("relation"), role) + self.backend._role_row_bonus(e, role),
                "reason": "one-hop graph expansion from matched anchor",
                "edge": e,
            })

        # sort and dedupe by kind/id
        out = []
        seen = set()
        for r in sorted(results, key=lambda x: -x.get("score", 0)):
            key = (r["kind"], r["id"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= limit:
                break
        ranked_anchor_rows: list[tuple[str, float, dict[str, Any]]] = []
        for nid, score in sorted(anchor_scores.items(), key=lambda item: (-item[1], item[0])):
            node = self.backend._get_node(nid)
            if not node or not self.backend._row_visible_to_role(node, role):
                continue
            ranked_anchor_rows.append((nid, float(score), node))

        anchor_filter_cfg = cfg_get(self.backend.config, "retrieval.anchor_filter", {})
        if not isinstance(anchor_filter_cfg, dict):
            anchor_filter_cfg = {}
        use_anchor_filter = bool(anchor_filter_cfg.get("enabled", True))
        relative_delta = float(anchor_filter_cfg.get("relative_delta", 25.0))
        min_score = float(anchor_filter_cfg.get("min_score", 0.0))
        min_anchors = max(1, int(anchor_filter_cfg.get("min_anchors", 4)))

        anchors: list[dict[str, Any]] = []
        anchor_decisions: list[dict[str, Any]] = []
        threshold = float("-inf")
        filtered_out = 0
        if ranked_anchor_rows:
            top_score = ranked_anchor_rows[0][1]
            threshold = max(min_score, top_score - relative_delta) if use_anchor_filter else float("-inf")
            for idx, (_nid, score, node) in enumerate(ranked_anchor_rows):
                keep = (idx < min_anchors) or (score >= threshold)
                selected = keep and len(anchors) < limit
                anchor_decisions.append({
                    "node_id": node["node_id"],
                    "score": round(score, 3),
                    "decision": "kept" if selected else ("dropped_limit" if keep else "dropped_threshold"),
                    "reasons": anchor_reasons.get(node["node_id"], []),
                })
                if selected:
                    anchors.append(node)
                elif not keep:
                    filtered_out += 1
            if use_anchor_filter:
                self.backend._log(
                    "debug",
                    "search.anchor_filter",
                    top_score=top_score,
                    threshold=threshold,
                    relative_delta=relative_delta,
                    min_score=min_score,
                    min_anchors=min_anchors,
                    kept_count=len(anchors),
                    filtered_count=filtered_out,
                )
        result = {
            "query": query,
            "role": role,
            "intent": intent,
            "anchors": anchors,
            "results": out,
        }
        if collect_trace:
            result["trace"] = {
                "fts_query": fquery,
                "anchor_filter": {
                    "enabled": use_anchor_filter,
                    "threshold": None if threshold == float("-inf") else round(threshold, 3),
                    "relative_delta": relative_delta,
                    "min_score": min_score,
                    "min_anchors": min_anchors,
                    "candidate_count": len(ranked_anchor_rows),
                    "kept_count": len(anchors),
                    "filtered_count": filtered_out,
                },
                "anchor_candidates": anchor_decisions[: max(limit * 3, 20)],
                "top_results": [
                    {
                        "kind": r.get("kind"),
                        "id": r.get("id"),
                        "score": round(float(r.get("score", 0.0)), 3),
                        "reason": r.get("reason"),
                    }
                    for r in out[:limit]
                ],
            }
        self.backend._log("info", "search.done", query=query, role=role, intent=intent, fts_query=fquery, result_count=len(out), anchor_count=len(anchor_scores), elapsed_ms=(time.perf_counter() - started) * 1000)
        self.backend._log(self.backend.trace_level, "search.results", query=query, results=out)
        return result

    def context(
        self,
        anchors: list[str] | None = None,
        query: str | None = None,
        role: str | None = None,
        intent: str | None = None,
        budget: str = "small",
        mode: str | None = None,
        include_stale_warnings: bool | None = None,
    ) -> ContextPacket:
        """Build a role-aware context packet.

        Retrieval modes are strategies, not stored DB objects:
        local  = direct node neighborhood.
        global = high-level flow/feature/concept/interface/runbook frame.
        bridge = bounded paths between anchors.
        hybrid = local + global + bridge.
        mix    = hybrid + optional semantic candidates + optional reranker.

        Returns a normalized packet with markdown and a persisted retrieval run.
        """
        started = time.perf_counter()
        limits = self.backend._budget_limits(budget)
        selected_mode = self.normalize_context_mode(mode)
        if include_stale_warnings is None:
            include_stale_warnings = bool(cfg_get(self.backend.config, "retrieval.include_stale_warnings", True))
        self.backend._log("info", "context.start", mode=selected_mode, anchors=anchors or [], query=query, role=role, intent=intent, budget=budget)

        resolution_trace: dict[str, Any] = {}
        stage_started = time.perf_counter()
        node_ids = self.resolve_context_nodes(anchors=anchors, query=query, role=role, intent=intent, limit=limits["nodes"], trace_sink=resolution_trace)
        timings_ms = {"anchor_resolution": round((time.perf_counter() - stage_started) * 1000, 3)}
        semantic: SemanticCandidates = {"enabled": False, "available": False, "reason": "mode does not request semantic retrieval", "results": []}
        promotion_stats: dict[str, Any] = {"enabled": False, "reason": "mode does not request semantic retrieval", "base_anchor_count": len(node_ids), "result_anchor_count": len(node_ids), "promoted_anchor_count": 0, "promoted_anchors": []}
        if selected_mode == "mix":
            stage_started = time.perf_counter()
            semantic = self.semantic_context(
                query=query or " ".join(anchors or []),
                role=role,
                lexical_anchor_count=len(node_ids),
            )
            node_ids, promotion_stats = self.maybe_promote_semantic_anchors(
                base_node_ids=node_ids,
                semantic=semantic,
                role=role,
                budget=budget,
                limit=limits["nodes"],
            )
            timings_ms["semantic"] = round((time.perf_counter() - stage_started) * 1000, 3)
            self.backend._log("debug", "context.semantic.done", available=semantic.get("available"), result_count=len(semantic.get("results", [])), reason=semantic.get("reason"))
            self.backend._log("debug", "context.semantic.promotion", **promotion_stats)

        stage_started = time.perf_counter()
        local = self.local_context_sections(node_ids, role=role, limits=limits, include_stale_warnings=include_stale_warnings)
        timings_ms["local"] = round((time.perf_counter() - stage_started) * 1000, 3)
        global_frames: list[dict[str, Any]] = []
        bridge_paths: list[dict[str, Any]] = []
        missing_links: list[dict[str, Any]] = []

        if selected_mode in {"global", "hybrid", "mix"}:
            stage_started = time.perf_counter()
            global_frames = self.global_context(node_ids, query=query, role=role, limits=limits)
            timings_ms["global"] = round((time.perf_counter() - stage_started) * 1000, 3)
            self.backend._log("debug", "context.global.done", frame_count=len(global_frames))
        if selected_mode in {"bridge", "hybrid", "mix"}:
            stage_started = time.perf_counter()
            bridge = self.bridge_context(node_ids, query=query, role=role, limits=limits)
            bridge_paths = bridge["paths"]
            missing_links.extend(bridge["missing_links"])
            timings_ms["bridge"] = round((time.perf_counter() - stage_started) * 1000, 3)
            self.backend._log("debug", "context.bridge.done", path_count=len(bridge_paths), missing_count=len(missing_links))
        stage_started = time.perf_counter()
        claims = list(local["claims"])
        edges = list(local["edges"])
        evidence = list(local["evidence"])
        for frame in global_frames:
            for c in frame.get("claims", []):
                claims.append(c)
                evidence.extend(self.backend._evidence_for_claim(c["claim_id"], limit=limits["evidence"]))
            if frame.get("via_edge"):
                edges.append(frame["via_edge"])
        for path in bridge_paths:
            for edge in path.get("edges", []):
                edges.append(edge)
                for c in self.backend._claims_for_edge(edge["edge_id"], include_stale=False, limit=limits["claims"], role=role):
                    claims.append(c)
                    evidence.extend(self.backend._evidence_for_claim(c["claim_id"], limit=limits["evidence"]))

        claims = self.backend._dedupe_by(claims, "claim_id")[: limits["claims"] * 2]
        edges = self.backend._sort_edges_for_role(self.backend._dedupe_by(edges, "edge_id"), role)[: limits["edges"] * 2]
        evidence = self.backend._dedupe_by(evidence, "chunk_id")[: limits["evidence"] * max(1, len(claims))]
        timings_ms["assembly"] = round((time.perf_counter() - stage_started) * 1000, 3)

        packet: ContextPacket = {
            "mode": selected_mode,
            "role": role,
            "intent": intent,
            "budget": budget,
            "query": query,
            "config_path": str(self.backend.config_path) if self.backend.config_path else None,
            "selected_anchors": local["nodes"],
            "global_frames": global_frames,
            "bridge_paths": bridge_paths,
            "active_claims": claims,
            "related_edges": edges,
            "evidence_refs": evidence,
            "semantic_candidates": semantic,
            "cross_role_notes": self.backend._cross_role_notes(claims, edges, role),
            "missing_links": missing_links,
            "stale_or_conflict_warnings": local["warnings"],
            "suggested_next_checks": self.backend._suggest_next_checks(local["nodes"], role, intent),
            "do_not_assume": [
                "DocGraph context is guidance, not proof.",
                "Verify against current code/RTL/logs when the task requires current truth.",
                "Bridge paths are retrieved graph paths, not new relation types.",
                "Semantic/vector candidates are discovery hints, not trusted graph facts.",
                "Stale warnings mean older evidence may no longer describe current implementation.",
            ],
        }
        retrieval_trace = {
            "requested_anchors": anchors or [],
            "anchor_resolution": resolution_trace,
            "semantic_promotion": promotion_stats,
            "semantic_candidates": [
                {
                    "kind": r.get("kind"),
                    "id": r.get("id"),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "reranker_score": r.get("reranker_score"),
                    "llm_reranker_score": r.get("llm_reranker_score"),
                }
                for r in semantic.get("results", [])[:12]
            ],
            "rerank_trace": semantic.get("rerank_trace", {}),
            "global_frames": [
                {
                    "node_id": f.get("node", {}).get("node_id"),
                    "distance": f.get("distance"),
                    "score": round(float(f.get("score", 0.0)), 3),
                    "via_edge_id": (f.get("via_edge") or {}).get("edge_id"),
                }
                for f in global_frames
            ],
            "bridge_paths": [
                {
                    "from_node_id": p.get("from_node_id"),
                    "to_node_id": p.get("to_node_id"),
                    "score": round(float(p.get("score", 0.0)), 3),
                    "node_ids": [n.get("node_id") for n in p.get("nodes", [])],
                    "edge_ids": [e.get("edge_id") for e in p.get("edges", [])],
                }
                for p in bridge_paths
            ],
            "final": {
                "anchor_ids": [n.get("node_id") for n in local["nodes"]],
                "claim_ids": [c.get("claim_id") for c in claims],
                "edge_ids": [e.get("edge_id") for e in edges],
                "evidence_chunk_ids": [e.get("chunk_id") for e in evidence],
            },
            "timings_ms": timings_ms,
        }
        retrieval_trace["timings_ms"]["packet_before_persist"] = round((time.perf_counter() - started) * 1000, 3)
        packet["markdown"] = self.context_markdown(packet)
        self.record_retrieval_run(packet, retrieval_trace)
        self.backend._log("info", "context.done", mode=selected_mode, anchor_count=len(packet["selected_anchors"]), global_frame_count=len(global_frames), bridge_path_count=len(bridge_paths), claim_count=len(claims), semantic_available=semantic.get("available"), elapsed_ms=(time.perf_counter() - started) * 1000)
        self.backend._log(self.backend.trace_level, "context.packet", packet=packet)
        return packet

    def normalize_context_mode(self, mode: str | None) -> str:
        """Validate mode against supported values and config-enabled switches."""
        selected = mode or str(cfg_get(self.backend.config, "retrieval.default_mode", "hybrid"))
        if selected not in {"local", "global", "bridge", "hybrid", "mix"}:
            raise ValueError(f"unsupported context mode: {selected}")
        if not bool(cfg_get(self.backend.config, f"retrieval.modes.{selected}.enabled", True)):
            raise ValueError(f"context mode disabled by config: {selected}")
        return selected

    def resolve_context_nodes(self, anchors: list[str] | None, query: str | None, role: str | None, intent: str | None, limit: int, trace_sink: dict[str, Any] | None = None) -> list[str]:
        """Resolve anchor strings and query hits into visible node ids."""
        node_ids: list[str] = []
        explicit: list[dict[str, Any]] = []
        if anchors:
            for a in anchors:
                direct = self.backend._get_node(a)
                if direct and self.backend._row_visible_to_role(direct, role):
                    node_ids.append(a)
                    explicit.append({"input": a, "node_id": a, "match_type": "direct_node_id"})
                else:
                    for m in self.backend.resolve(a, limit=3)["matches"]:
                        if self.backend._node_visible_to_role(m["node_id"], role):
                            node_ids.append(m["node_id"])
                            explicit.append({"input": a, "node_id": m["node_id"], "match_type": m.get("match_type")})
        if query and len(node_ids) < limit:
            sr = self.search(query, role=role, intent=intent, limit=limit, collect_trace=trace_sink is not None)
            for n in sr["anchors"]:
                if n:
                    node_ids.append(n["node_id"])
            if trace_sink is not None:
                trace_sink["lexical_search"] = sr.get("trace", {})
        result = list(dict.fromkeys(node_ids))[:limit]
        if trace_sink is not None:
            trace_sink["explicit"] = explicit
            trace_sink["resolved_base_anchor_ids"] = result
        return result

    def local_context_sections(self, node_ids: list[str], role: str | None, limits: dict[str, int], include_stale_warnings: bool) -> dict[str, list[dict[str, Any]]]:
        """Collect local nodes, claims, edges, evidence, and optional warnings."""
        nodes: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for nid in node_ids:
            n = self.backend._get_node(nid)
            if not n:
                continue
            if not self.backend._row_visible_to_role(n, role):
                continue
            n["aliases"] = self.backend._aliases_for_node(nid)
            nodes.append(n)
            for c in self.backend._claims_for_node(nid, include_stale=False, limit=limits["claims"], role=role):
                claims.append(c)
                evidence.extend(self.backend._evidence_for_claim(c["claim_id"], limit=limits["evidence"]))
            edges.extend([e for e in self.backend._edges_for_node(nid, include_stale=False, limit=limits["edges"]) if self.backend._row_visible_to_role(e, role)])
            if include_stale_warnings:
                for c in self.backend._claims_for_node(nid, include_stale=True, limit=10, role=role):
                    if c["status"] != "active":
                        warnings.append({"claim_id": c["claim_id"], "status": c["status"], "claim_text": c["claim_text"]})
        return {
            "nodes": self.backend._dedupe_by(nodes, "node_id"),
            "claims": self.backend._dedupe_by(claims, "claim_id"),
            "edges": self.backend._sort_edges_for_role(self.backend._dedupe_by(edges, "edge_id"), role)[: limits["edges"]],
            "evidence": self.backend._dedupe_by(evidence, "chunk_id"),
            "warnings": self.backend._dedupe_by(warnings, "claim_id"),
        }

    def global_context(self, node_ids: list[str], query: str | None, role: str | None, limits: dict[str, int]) -> list[dict[str, Any]]:
        """Find high-level frames around anchors using bounded upward traversal."""
        preferred_types = set(cfg_list(self.backend.config, "retrieval.global_scope.preferred_node_types")) or set(cfg_list(self.backend.config, "shared_knowledge.high_level_node_types"))
        upward_relations = set(cfg_list(self.backend.config, "retrieval.global_scope.upward_relations"))
        max_depth = int(cfg_get(self.backend.config, "retrieval.modes.global.max_depth", 3))
        frames: list[dict[str, Any]] = []
        seen_frames: set[str] = set()
        for start in node_ids:
            q: deque[tuple[str, int, dict[str, Any] | None]] = deque([(start, 0, None)])
            seen = {start}
            while q:
                nid, depth, via = q.popleft()
                if depth > 0:
                    node = self.backend._get_node(nid)
                    if node and node.get("node_type") in preferred_types and nid not in seen_frames and self.backend._row_visible_to_role(node, role):
                        seen_frames.add(nid)
                        claims = self.backend._claims_for_node(nid, include_stale=False, limit=max(1, limits["claims"] // 2), role=role)
                        frames.append({"node": node, "distance": depth, "via_edge": via, "score": self.backend._rank_weight("global_frame_bonus", 18.0) - depth + self.backend._role_node_bonus(nid, role), "claims": claims})
                if depth >= max_depth:
                    continue
                for edge in self.backend._edges_for_node(nid, include_stale=False, limit=limits["edges"] * 3):
                    if upward_relations and edge["relation"] not in upward_relations:
                        continue
                    if not self.backend._row_visible_to_role(edge, role):
                        continue
                    other = edge["to_node_id"] if edge["from_node_id"] == nid else edge["from_node_id"]
                    if other in seen:
                        continue
                    seen.add(other)
                    q.append((other, depth + 1, self.backend._edge_with_names(edge)))
        if query:
            for r in self.search(query, role=role, intent="architecture", limit=limits["nodes"] * 2)["results"]:
                if r.get("kind") != "node":
                    continue
                node = r.get("node") or {}
                nid = node.get("node_id")
                if nid and node.get("node_type") in preferred_types and nid not in seen_frames:
                    seen_frames.add(nid)
                    frames.append({"node": node, "distance": 0, "via_edge": None, "score": float(r.get("score", 0)) + self.backend._rank_weight("global_frame_bonus", 18.0), "claims": self.backend._claims_for_node(nid, include_stale=False, limit=max(1, limits["claims"] // 2), role=role)})
        return sorted(frames, key=lambda x: -x["score"])[: limits["nodes"]]

    def bridge_context(self, node_ids: list[str], query: str | None, role: str | None, limits: dict[str, int]) -> dict[str, Any]:
        """Build bounded bridge paths between resolved anchors."""
        max_depth = int(cfg_get(self.backend.config, "retrieval.modes.bridge.max_depth", 4))
        max_paths = int(cfg_get(self.backend.config, "retrieval.modes.bridge.max_paths", 6))
        max_anchors = int(cfg_get(self.backend.config, "retrieval.modes.bridge.max_anchors", 10))
        anchor_cap = max(2, min(limits["nodes"], max_anchors))
        anchors = list(dict.fromkeys(node_ids))[:anchor_cap]
        paths: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        if len(anchors) < 2:
            return {"paths": [], "missing_links": [{"reason": "bridge mode needs at least two resolved anchors", "anchors": anchors}]}
        for i, start in enumerate(anchors):
            for goal in anchors[i + 1 :]:
                pair_paths = self.find_bridge_paths(start, goal, role=role, max_depth=max_depth, max_paths=max_paths)
                if not pair_paths:
                    missing.append({"from_node_id": start, "to_node_id": goal, "reason": "no active path within configured bridge depth"})
                paths.extend(pair_paths)
        paths = sorted(paths, key=lambda x: -x["score"])[:max_paths]
        return {"paths": paths, "missing_links": missing[:max_paths]}

    def find_bridge_paths(self, start: str, goal: str, role: str | None, max_depth: int, max_paths: int) -> list[dict[str, Any]]:
        """Run bounded BFS to discover visible paths between two anchors."""
        preferred_relations = set(cfg_list(self.backend.config, "retrieval.bridge_scope.preferred_relations"))
        q: deque[tuple[str, list[dict[str, Any]], set[str]]] = deque([(start, [], {start})])
        paths: list[dict[str, Any]] = []
        while q and len(paths) < max_paths:
            nid, edge_path, seen = q.popleft()
            if len(edge_path) >= max_depth:
                continue
            candidate_edges = self.backend._edges_for_node(nid, include_stale=False, limit=64)
            candidate_edges = [e for e in candidate_edges if self.backend._row_visible_to_role(e, role)]
            candidate_edges = sorted(candidate_edges, key=lambda e: -((1 if e["relation"] in preferred_relations else 0) * 10 + self.backend._role_relation_bonus(e["relation"], role)))
            for edge in candidate_edges:
                other = edge["to_node_id"] if edge["from_node_id"] == nid else edge["from_node_id"]
                if other in seen:
                    continue
                next_path = edge_path + [self.backend._edge_with_names(edge)]
                if other == goal:
                    paths.append(self.format_bridge_path(start, goal, next_path, role=role))
                    continue
                q.append((other, next_path, seen | {other}))
        return paths

    def format_bridge_path(self, start: str, goal: str, edge_path: list[dict[str, Any]], role: str | None) -> dict[str, Any]:
        """Attach score, node sequence, and explanation text for one path."""
        ordered_nodes = [start]
        current = start
        relation_score = 0.0
        degree_penalty = 0.0
        for edge in edge_path:
            relation_score += self.backend._role_relation_bonus(edge.get("relation"), role)
            other = edge["to_node_id"] if edge["from_node_id"] == current else edge["from_node_id"]
            ordered_nodes.append(other)
            current = other
            degree_penalty += max(0, self.backend._node_degree(other) - 8) * self.backend._rank_weight("generic_node_degree_penalty", -2.0)
        nodes = [self.backend._get_node(nid) for nid in ordered_nodes]
        nodes = [n for n in nodes if n]
        score = self.backend._rank_weight("bridge_path_base", 80.0) + relation_score + degree_penalty + len(edge_path) * self.backend._rank_weight("bridge_path_step_penalty", -8.0)
        return {"from_node_id": start, "to_node_id": goal, "score": score, "nodes": nodes, "edges": edge_path, "explanation": " -> ".join(f"{e['from_node_id']} --{e['relation']}--> {e['to_node_id']}" for e in edge_path)}

    def maybe_promote_semantic_anchors(
        self,
        base_node_ids: list[str],
        semantic: SemanticCandidates,
        role: str | None,
        budget: str | None,
        limit: int,
    ) -> tuple[list[str], dict[str, Any]]:
        """Fuse lexical anchors with strong semantic hits when enabled."""
        # Preferred config path is mode-scoped; retain legacy fallback for compatibility.
        legacy_cfg = cfg_get(self.backend.config, "retrieval.semantic_anchor_promotion", {})
        if not isinstance(legacy_cfg, dict):
            legacy_cfg = {}
        mode_cfg = cfg_get(self.backend.config, "retrieval.modes.mix.semantic_anchor_promotion", {})
        if not isinstance(mode_cfg, dict):
            mode_cfg = {}
        cfg = dict(legacy_cfg)
        cfg.update(mode_cfg)
        if not bool(cfg.get("enabled", False)):
            return base_node_ids, {
                "enabled": False,
                "reason": "disabled",
                "base_anchor_count": len(base_node_ids),
                "result_anchor_count": len(base_node_ids),
                "promoted_anchor_count": 0,
            }
        if not semantic.get("available"):
            return base_node_ids, {
                "enabled": True,
                "reason": "semantic unavailable",
                "base_anchor_count": len(base_node_ids),
                "result_anchor_count": len(base_node_ids),
                "promoted_anchor_count": 0,
            }

        results = list(semantic.get("results", []))
        if not results:
            return base_node_ids, {
                "enabled": True,
                "reason": "no semantic results",
                "base_anchor_count": len(base_node_ids),
                "result_anchor_count": len(base_node_ids),
                "promoted_anchor_count": 0,
            }

        max_semantic_results = max(1, int(cfg.get("top_semantic_results", 16)))
        max_promoted = max(0, int(cfg.get("max_promoted_anchors", 4)))
        min_lexical = max(0, int(cfg.get("min_lexical_anchors", 3)))
        semantic_min_score = float(cfg.get("min_score", 0.0))
        semantic_relative_delta = float(cfg.get("relative_delta", 12.0))
        lexical_weight = float(cfg.get("lexical_weight", 1.0))
        semantic_weight = float(cfg.get("semantic_weight", 1.2))
        rrf_k = max(1.0, float(cfg.get("rrf_k", 60.0)))
        require_graph_coherence = bool(cfg.get("require_graph_coherence", True))
        coherence_min_lexical = max(1, int(cfg.get("coherence_min_lexical_anchors", 2)))
        coherence_max_depth = self.coherence_max_depth_for_budget(cfg, budget)

        subset = results[:max_semantic_results]
        top_score = float(subset[0].get("score", 0.0))
        threshold = max(semantic_min_score, top_score - semantic_relative_delta)

        semantic_rank: dict[str, int] = {}
        considered = 0
        accepted = 0
        for rank, cand in enumerate(subset, start=1):
            considered += 1
            score = float(cand.get("score", 0.0))
            if score < threshold:
                continue
            node_ids = self.semantic_candidate_node_ids(cand, role=role)
            if not node_ids:
                continue
            accepted += 1
            for nid in node_ids:
                prev = semantic_rank.get(nid)
                if prev is None or rank < prev:
                    semantic_rank[nid] = rank

        if not semantic_rank:
            return base_node_ids, {
                "enabled": True,
                "reason": "no semantic nodes passed filter",
                "base_anchor_count": len(base_node_ids),
                "result_anchor_count": len(base_node_ids),
                "promoted_anchor_count": 0,
                "considered_result_count": considered,
                "accepted_result_count": accepted,
                "threshold": threshold,
            }

        base_order = list(dict.fromkeys(base_node_ids))
        base_set = set(base_order)
        semantic_only = sorted((nid for nid in semantic_rank if nid not in base_set), key=lambda nid: (semantic_rank[nid], nid))
        coherence_applied = require_graph_coherence and len(base_order) >= coherence_min_lexical
        coherence_paths: dict[str, dict[str, Any]] = {}
        rejected_semantic_only: list[str] = []
        eligible_semantic_only = semantic_only
        if coherence_applied:
            coherence_paths = self.semantic_coherence_paths(
                base_order,
                semantic_only,
                role=role,
                max_depth=coherence_max_depth,
            )
            eligible_semantic_only = [nid for nid in semantic_only if nid in coherence_paths]
            rejected_semantic_only = [nid for nid in semantic_only if nid not in coherence_paths]
        promoted = eligible_semantic_only[:max_promoted] if max_promoted > 0 else []
        pool = list(dict.fromkeys(base_order + promoted))
        if not pool:
            return base_node_ids, {
                "enabled": True,
                "reason": "empty fusion pool",
                "base_anchor_count": len(base_node_ids),
                "result_anchor_count": len(base_node_ids),
                "promoted_anchor_count": 0,
            }

        lexical_rank = {nid: rank for rank, nid in enumerate(base_order, start=1)}

        def rrf(rank: int | None) -> float:
            if rank is None:
                return 0.0
            return 1.0 / (rrf_k + float(rank))

        scored = []
        for nid in pool:
            fused = lexical_weight * rrf(lexical_rank.get(nid)) + semantic_weight * rrf(semantic_rank.get(nid))
            scored.append((nid, fused))
        scored.sort(key=lambda item: (-item[1], lexical_rank.get(item[0], 10**9), item[0]))

        keep_lexical = base_order[: min(min_lexical, len(base_order), limit)]
        selected = list(keep_lexical)
        for nid, _score in scored:
            if nid in selected:
                continue
            selected.append(nid)
            if len(selected) >= limit:
                break

        result = selected[:limit]
        promoted_in_result = [nid for nid in result if nid not in base_set]
        return result, {
            "enabled": True,
            "reason": "ok",
            "base_anchor_count": len(base_order),
            "result_anchor_count": len(result),
            "promoted_anchor_count": len(promoted_in_result),
            "promoted_anchors": promoted_in_result,
            "considered_result_count": considered,
            "accepted_result_count": accepted,
            "semantic_node_count": len(semantic_rank),
            "threshold": threshold,
            "top_score": top_score,
            "coherence_gate": {
                "enabled": require_graph_coherence,
                "applied": coherence_applied,
                "reason": "lexical context established" if coherence_applied else (
                    "disabled" if not require_graph_coherence else "insufficient lexical anchors for coherence gating"
                ),
                "min_lexical_anchors": coherence_min_lexical,
                "max_depth": coherence_max_depth,
                "budget": budget or "small",
                "eligible_semantic_only": eligible_semantic_only,
                "rejected_semantic_only": rejected_semantic_only,
                "connections": [
                    {"node_id": nid, **coherence_paths[nid]}
                    for nid in eligible_semantic_only
                    if nid in coherence_paths
                ],
            },
        }

    def coherence_max_depth_for_budget(self, cfg: dict[str, Any], budget: str | None) -> int:
        """Resolve semantic-coherence depth, allowing budget-specific overrides."""
        default_depth = max(1, int(cfg.get("coherence_max_depth", 2)))
        by_budget = cfg.get("coherence_max_depth_by_budget", {})
        if isinstance(by_budget, dict):
            raw_depth = by_budget.get(str(budget or "small"))
            if raw_depth is not None:
                return max(1, int(raw_depth))
        return default_depth

    def semantic_coherence_paths(
        self,
        lexical_node_ids: list[str],
        semantic_node_ids: list[str],
        role: str | None,
        max_depth: int,
    ) -> dict[str, dict[str, Any]]:
        """Map semantic-only nodes reachable from lexical anchors in bounded active graph."""
        targets = set(semantic_node_ids) - set(lexical_node_ids)
        if not targets or not lexical_node_ids:
            return {}
        seeds = list(dict.fromkeys(lexical_node_ids))
        visited = set(seeds)
        queue: deque[tuple[str, str, list[str], list[str]]] = deque(
            (seed, seed, [seed], []) for seed in seeds
        )
        connections: dict[str, dict[str, Any]] = {}
        while queue:
            current, seed, node_path, edge_path = queue.popleft()
            if len(edge_path) >= max_depth:
                continue
            for edge in self.backend._edges_for_node(current, include_stale=False, limit=64):
                if not self.backend._row_visible_to_role(edge, role):
                    continue
                other = edge["to_node_id"] if edge["from_node_id"] == current else edge["from_node_id"]
                if other in visited or not self.backend._node_visible_to_role(other, role):
                    continue
                next_nodes = node_path + [other]
                next_edges = edge_path + [edge["edge_id"]]
                visited.add(other)
                if other in targets:
                    connections[other] = {
                        "lexical_anchor_id": seed,
                        "distance": len(next_edges),
                        "node_ids": next_nodes,
                        "edge_ids": next_edges,
                    }
                queue.append((other, seed, next_nodes, next_edges))
        return connections

    def semantic_candidate_node_ids(self, candidate: dict[str, Any], role: str | None) -> list[str]:
        """Project one semantic result into visible node ids."""
        kind = str(candidate.get("kind") or "")
        cid = str(candidate.get("id") or "")
        out: list[str] = []

        if kind == "node" and cid:
            if self.backend._node_visible_to_role(cid, role):
                out.append(cid)
            return out

        if kind == "claim" and cid:
            claim = candidate.get("claim")
            if not isinstance(claim, dict):
                row = self.backend.conn.execute("SELECT * FROM claims WHERE claim_id=?", (cid,)).fetchone()
                claim = dict(row) if row else None
            if isinstance(claim, dict) and self.backend._claim_visible_to_role(claim, role):
                for nid in self.backend._claim_target_nodes(claim):
                    if self.backend._node_visible_to_role(nid, role):
                        out.append(nid)
            return list(dict.fromkeys(out))

        if kind == "edge" and cid:
            edge = candidate.get("edge")
            if not isinstance(edge, dict):
                row = self.backend.conn.execute("SELECT * FROM edges WHERE edge_id=?", (cid,)).fetchone()
                edge = self.backend._edge_with_names(dict(row)) if row else None
            if isinstance(edge, dict) and self.backend._row_visible_to_role(edge, role):
                for nid in (edge.get("from_node_id"), edge.get("to_node_id")):
                    if isinstance(nid, str) and self.backend._node_visible_to_role(nid, role):
                        out.append(nid)
            return list(dict.fromkeys(out))

        return []

    def semantic_context(self, query: str | None, role: str | None, *, lexical_anchor_count: int = 0) -> SemanticCandidates:
        """Return optional embedding/reranker candidates for ``mix`` retrieval."""
        if not query or not query.strip():
            return {"enabled": False, "available": False, "reason": "no query supplied", "results": []}
        provider, state = self.backend._make_embedding_provider()
        if not state.enabled or not state.available or provider is None:
            self.backend._log("debug", "semantic.disabled_or_unavailable", enabled=state.enabled, available=state.available, reason=state.reason)
            return {"enabled": state.enabled, "available": state.available, "reason": state.reason, "results": []}
        candidates = self.semantic_candidate_texts(role=role)
        if not candidates:
            self.backend._log("debug", "semantic.no_candidates", query=query, role=role)
            return {"enabled": True, "available": True, "reason": "no active candidates to embed", "results": []}
        try:
            self.backend._log("debug", "semantic.embed.start", query=query, candidate_count=len(candidates), role=role)
            qvec = provider.embed_texts([query], is_query=True)[0]
            cvecs = self.candidate_embeddings(provider, candidates)
        except Exception as exc:  # pragma: no cover - optional model runtime
            self.backend._log("error", "semantic.embed.error", query=query, error_type=type(exc).__name__, error=str(exc))
            return {"enabled": True, "available": False, "reason": str(exc), "results": []}
        scored = []
        for cand, vec in zip(candidates, cvecs):
            score = self.backend._rank_weight("semantic_candidate", 55.0) * cosine_similarity(qvec, vec) + float(cand.get("role_bonus", 0.0))
            scored.append({**cand, "embedding_score": score, "score": score})
        scored = sorted(scored, key=lambda x: -x["score"])
        scored, rerank_trace = self.maybe_rerank(query, scored, lexical_anchor_count=lexical_anchor_count)
        top_k = int(cfg_get(self.backend.config, "retrieval_models.embeddings.top_k", 12))
        result: SemanticCandidates = {
            "enabled": True,
            "available": True,
            "reason": None,
            "results": scored[:top_k],
            "rerank_trace": rerank_trace,
        }
        self.backend._log("info", "semantic.done", query=query, candidate_count=len(candidates), result_count=len(result["results"]), role=role)
        return result

    def candidate_embeddings(self, provider: Any, candidates: list[dict[str, Any]]) -> list[list[float]]:
        """Return candidate embeddings, reusing incremental cache when enabled."""
        if not candidates:
            return []
        use_cache = bool(cfg_get(self.backend.config, "retrieval_models.embeddings.incremental_cache_enabled", True))
        if not use_cache:
            return provider.embed_texts([c["text"] for c in candidates], is_query=False)
        model_key = self.embedding_model_key(provider)
        vectors: list[list[float] | None] = [None] * len(candidates)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        cached_count = 0
        for idx, cand in enumerate(candidates):
            item_type = str(cand["kind"])
            target_id = str(cand["id"])
            text_hash = self.text_hash(str(cand["text"]))
            try:
                row = self.backend.conn.execute(
                    """
                    SELECT text_hash, index_ref
                    FROM embedding_items
                    WHERE item_type=? AND target_id=? AND embedding_model=?
                    """,
                    (item_type, target_id, model_key),
                ).fetchone()
            except sqlite3.Error as exc:
                self.backend._log("warning", "semantic.cache.read_error", error_type=type(exc).__name__, error=str(exc))
                return provider.embed_texts([c["text"] for c in candidates], is_query=False)
            vector = self.cached_vector_from_row(row, expected_hash=text_hash)
            if vector is None:
                missing_indexes.append(idx)
                missing_texts.append(str(cand["text"]))
            else:
                vectors[idx] = vector
                cached_count += 1
        if missing_indexes:
            new_vectors = provider.embed_texts(missing_texts, is_query=False)
            if len(new_vectors) != len(missing_indexes):
                raise RuntimeError("embedding provider returned an unexpected vector count")
            try:
                with self.backend.conn:
                    for idx, vec in zip(missing_indexes, new_vectors):
                        cand = candidates[idx]
                        vector = [float(v) for v in vec]
                        vectors[idx] = vector
                        self.backend.conn.execute(
                            """
                            INSERT INTO embedding_items(item_type, target_id, text_hash, embedding_model, index_ref, updated_at)
                            VALUES(?,?,?,?,?,?)
                            ON CONFLICT(item_type, target_id, embedding_model)
                            DO UPDATE SET text_hash=excluded.text_hash, index_ref=excluded.index_ref, updated_at=excluded.updated_at
                            """,
                            (
                                str(cand["kind"]),
                                str(cand["id"]),
                                self.text_hash(str(cand["text"])),
                                model_key,
                                json.dumps(vector, separators=(",", ":")),
                                self._now_ts(),
                            ),
                        )
            except sqlite3.Error as exc:
                self.backend._log("warning", "semantic.cache.write_error", error_type=type(exc).__name__, error=str(exc))
                return provider.embed_texts([c["text"] for c in candidates], is_query=False)
        out = []
        for vec in vectors:
            if vec is None:
                raise RuntimeError("embedding cache internal error: missing vector")
            out.append(vec)
        self.backend._log(
            "debug",
            "semantic.cache.stats",
            model=model_key,
            candidate_count=len(candidates),
            cached_count=cached_count,
            embedded_count=len(missing_indexes),
        )
        return out

    def embedding_model_key(self, provider: Any) -> str:
        """Return a stable cache key for the embedding provider/model."""
        for attr in ("model_name", "model"):
            value = getattr(provider, attr, None)
            if isinstance(value, str) and value:
                return value
        return type(provider).__name__

    def text_hash(self, text: str) -> str:
        """Hash text payload used for embedding cache invalidation."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def cached_vector_from_row(self, row: Any, expected_hash: str) -> list[float] | None:
        """Decode cached vector when hash matches; otherwise return None."""
        if row is None:
            return None
        if row["text_hash"] != expected_hash or not row["index_ref"]:
            return None
        try:
            payload = json.loads(row["index_ref"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, list) or not payload:
            return None
        try:
            return [float(v) for v in payload]
        except (TypeError, ValueError):
            return None

    def semantic_candidate_texts(self, role: str | None) -> list[dict[str, Any]]:
        """Materialize visible active node/claim/edge texts for embedding search."""
        max_items = int(cfg_get(self.backend.config, "retrieval_models.embeddings.max_in_memory_items", 200))
        candidates: list[dict[str, Any]] = []
        for r in self.backend.conn.execute("SELECT node_id, node_type, canonical_name, summary, visibility, finder_role, audience_roles_json, interface_tags_json FROM nodes WHERE status='active' LIMIT ?", (max_items,)):
            d = dict(r)
            if not self.backend._row_visible_to_role(d, role):
                continue
            text = f"{d['canonical_name']}\n{d['node_type']}\n{d.get('summary') or ''}".strip()
            candidates.append({"kind": "node", "id": d["node_id"], "text": text, "node": d, "role_bonus": self.backend._role_node_bonus(d["node_id"], role)})
        remaining = max(0, max_items - len(candidates))
        for r in self.backend.conn.execute("SELECT * FROM claims WHERE status='active' LIMIT ?", (remaining,)):
            d = dict(r)
            if self.backend._claim_visible_to_role(d, role):
                candidates.append({"kind": "claim", "id": d["claim_id"], "text": d["claim_text"], "claim": d, "role_bonus": self.backend._role_claim_bonus(d, role)})
        remaining = max(0, max_items - len(candidates))
        for r in self.backend.conn.execute("SELECT * FROM edges WHERE status='active' LIMIT ?", (remaining,)):
            d = self.backend._edge_with_names(dict(r))
            if self.backend._row_visible_to_role(d, role):
                text = f"{d['from_node_id']} {d['relation']} {d['to_node_id']}\n{d.get('summary') or ''}".strip()
                candidates.append({"kind": "edge", "id": d["edge_id"], "text": text, "edge": d, "role_bonus": self.backend._role_relation_bonus(d.get("relation"), role) + self.backend._role_row_bonus(d, role)})
        return candidates

    def maybe_rerank(self, query: str, candidates: list[dict[str, Any]], *, lexical_anchor_count: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply optional cross-encoder and LLM reranker stages without failing retrieval."""
        trace: dict[str, Any] = {}
        embedding_top_score = float(candidates[0].get("embedding_score", candidates[0].get("score", 0.0))) if candidates else 0.0
        ranked, cross_trace = self.apply_reranker_stage(
            query=query,
            candidates=candidates,
            cfg_group="reranker",
            provider_getter=self.backend._make_reranker_provider,
            score_field="reranker_score",
            stage_name="cross_encoder",
        )
        trace["cross_encoder"] = cross_trace

        llm_triggered, llm_reason = self.should_run_llm_reranker(
            candidates=ranked,
            lexical_anchor_count=lexical_anchor_count,
            embedding_top_score=embedding_top_score,
            cross_encoder_trace=cross_trace,
        )
        if llm_triggered:
            ranked, llm_trace = self.apply_reranker_stage(
                query=query,
                candidates=ranked,
                cfg_group="llm_reranker",
                provider_getter=self.backend._make_llm_reranker_provider,
                score_field="llm_reranker_score",
                stage_name="llm",
            )
            trace["llm"] = llm_trace
        else:
            llm_cfg = cfg_get(self.backend.config, "retrieval_models.llm_reranker", {})
            trace["llm"] = {
                "enabled": bool(isinstance(llm_cfg, dict) and llm_cfg.get("enabled", False)),
                "triggered": False,
                "reason": llm_reason,
            }
        return ranked, trace

    def should_run_llm_reranker(
        self,
        *,
        candidates: list[dict[str, Any]],
        lexical_anchor_count: int,
        embedding_top_score: float,
        cross_encoder_trace: dict[str, Any],
    ) -> tuple[bool, str]:
        """Decide whether the optional LLM reranker rescue stage should run."""
        cfg = cfg_get(self.backend.config, "retrieval_models.llm_reranker", {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return False, "llm reranker disabled"
        trigger = str(cfg.get("trigger", "rescue_only")).strip().lower()
        if trigger in {"disabled", "off", "false", "never"}:
            return False, "llm reranker trigger disabled"
        if trigger in {"always", "on", "true"}:
            return True, "trigger=always"

        rescue = cfg.get("rescue", {})
        if not isinstance(rescue, dict):
            rescue = {}
        max_lexical = max(0, int(rescue.get("max_lexical_anchors", 2)))
        if lexical_anchor_count <= max_lexical:
            return True, f"lexical_anchor_count={lexical_anchor_count}<={max_lexical}"

        min_top_score = float(rescue.get("min_top_score", 0.0))
        if min_top_score > 0.0 and candidates:
            top_score = float(candidates[0].get("reranker_score", candidates[0].get("score", 0.0)))
            if top_score < min_top_score:
                return True, f"top_score={top_score:.3f}<{min_top_score:.3f}"

        min_embedding = float(rescue.get("min_embedding_top_score", 0.0))
        if min_embedding > 0.0 and not bool(cross_encoder_trace.get("applied")) and embedding_top_score < min_embedding:
            return True, f"embedding_top_score={embedding_top_score:.3f}<{min_embedding:.3f}"

        return False, "rescue conditions not met"

    def apply_reranker_stage(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        cfg_group: str,
        provider_getter: Callable[[], Tuple[Optional[Any], Any]],
        score_field: str,
        stage_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run one reranker stage and return candidates plus compact stage telemetry."""
        top_input = int(cfg_get(self.backend.config, f"retrieval_models.{cfg_group}.top_k_input", 40))
        top_output = int(cfg_get(self.backend.config, f"retrieval_models.{cfg_group}.top_k_output", 12))
        min_relevance = float(cfg_get(self.backend.config, f"retrieval_models.{cfg_group}.min_relevance_score", 0.0))
        subset = candidates[:top_input]
        provider, state = provider_getter()
        if not state.enabled:
            self.backend._log("debug", f"{stage_name}.skipped", enabled=False, available=False, reason=state.reason, candidate_count=len(candidates))
            return candidates, {"enabled": False, "applied": False, "available": False, "reason": state.reason}
        if not state.available or provider is None:
            self.backend._log("debug", f"{stage_name}.skipped", enabled=True, available=False, reason=state.reason, candidate_count=len(candidates))
            return candidates, {"enabled": True, "applied": False, "available": False, "reason": state.reason}
        if not subset:
            return candidates, {"enabled": True, "applied": False, "available": True, "reason": "no candidates"}

        try:
            self.backend._log(
                "debug",
                f"{stage_name}.start",
                query=query,
                candidate_count=len(candidates),
                top_k_input=top_input,
                top_k_output=top_output,
            )
            scores = provider.score(query, [str(c.get("text") or "") for c in subset])
        except Exception as exc:  # pragma: no cover - optional model/runtime instability
            self.backend._log("error", f"{stage_name}.error", query=query, error_type=type(exc).__name__, error=str(exc))
            return candidates, {"enabled": True, "applied": False, "available": True, "reason": str(exc)}

        if len(scores) != len(subset):
            self.backend._log(
                "warning",
                f"{stage_name}.count_mismatch",
                expected=len(subset),
                actual=len(scores),
            )
            return candidates, {"enabled": True, "applied": False, "available": True, "reason": "score count mismatch"}

        reranked: list[dict[str, Any]] = []
        score_entries: list[dict[str, Any]] = []
        for cand, score in zip(subset, scores):
            relevance = float(score)
            item = {**cand, score_field: relevance}
            below_min_relevance = min_relevance > 0.0 and relevance < min_relevance
            if below_min_relevance:
                # Keep penalty on the reranker score scale to avoid embedding-scale rank inversions.
                final_score = relevance * 0.1
            else:
                final_score = relevance
            item["score"] = final_score
            reranked.append(item)
            score_entries.append(
                {
                    "kind": cand.get("kind"),
                    "id": cand.get("id"),
                    score_field: round(relevance, 4),
                    "final_score": round(final_score, 4),
                    "below_min_relevance": below_min_relevance,
                }
            )
        reranked = sorted(reranked, key=lambda x: -float(x.get("score", 0.0)))[:top_output]
        retained = {(item.get("kind"), item.get("id")) for item in reranked}
        for entry in score_entries:
            entry["retained"] = (entry.get("kind"), entry.get("id")) in retained
        tail = candidates[top_input:]
        done_fields: dict[str, Any] = {
            "input_count": len(subset),
            "output_count": len(reranked),
            "scored": score_entries,
        }
        if cfg_group == "llm_reranker" and hasattr(provider, "resolve_model"):
            try:
                done_fields["model"] = provider.resolve_model()
            except Exception:
                done_fields["model"] = None
        self.backend._log("info", f"{stage_name}.done", **done_fields)
        stage_trace = {
            "enabled": True,
            "applied": True,
            "available": True,
            "reason": None,
            "input_count": len(subset),
            "output_count": len(reranked),
            "scored": score_entries,
        }
        if cfg_group == "llm_reranker" and "model" in done_fields:
            stage_trace["model"] = done_fields.get("model")
        return reranked + tail, stage_trace

    def record_retrieval_run(self, packet: ContextPacket, retrieval_trace: dict[str, Any] | None = None) -> None:
        """Best-effort persistence of retrieval telemetry; never raises."""
        self.last_recorded_run_id = None
        try:
            summary = {"anchors": len(packet.get("selected_anchors", [])), "global_frames": len(packet.get("global_frames", [])), "bridge_paths": len(packet.get("bridge_paths", [])), "claims": len(packet.get("active_claims", [])), "missing_links": len(packet.get("missing_links", []))}
            with self.backend.conn:
                run_id = self._new_id("retrieval")
                self.backend.conn.execute(
                    "INSERT INTO retrieval_runs(run_id, query, anchors_json, mode, role, budget, result_summary_json, trace_json, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (run_id, packet.get("query"), json.dumps([n.get("node_id") for n in packet.get("selected_anchors", [])]), packet.get("mode") or "local", packet.get("role"), packet.get("budget") or "small", json.dumps(summary, sort_keys=True), json.dumps(retrieval_trace or {}, sort_keys=True), self._now_ts()),
                )
            self.last_recorded_run_id = run_id
            self.backend._log("debug", "retrieval_run.recorded", run_id=run_id, summary=summary)
        except sqlite3.Error as exc:
            self.backend._log("warning", "retrieval_run.record_failed", error_type=type(exc).__name__, error=str(exc))

    def context_markdown(self, packet: ContextPacket) -> str:
        """Render a stable markdown view of a context packet for operators."""
        lines = ["# DocGraph Context Packet", ""]
        lines.append(f"Mode: {packet.get('mode') or 'local'}")
        lines.append(f"Role: {packet.get('role') or 'unspecified'}")
        lines.append(f"Intent: {packet.get('intent') or 'unspecified'}")
        lines.append(f"Budget: {packet.get('budget')}")
        if packet.get("config_path"):
            lines.append(f"Config: {packet.get('config_path')}")
        lines.append("")
        lines.append("## Selected anchors")
        for n in packet["selected_anchors"]:
            lines.append(f"- {n['node_id']} ({n['node_type']}): {n.get('summary') or n['canonical_name']}")
            if n.get("aliases"):
                lines.append("  aliases: " + ", ".join(a["alias"] for a in n["aliases"][:8]))
        if packet.get("global_frames"):
            lines.append("")
            lines.append("## Global frames")
            for frame in packet["global_frames"]:
                n = frame["node"]
                lines.append(f"- {n['node_id']} ({n['node_type']}, distance={frame.get('distance')}): {n.get('summary') or n['canonical_name']}")
                if frame.get("via_edge"):
                    e = frame["via_edge"]
                    lines.append(f"  via: {e['from_node_id']} --{e['relation']}--> {e['to_node_id']}")
        if packet.get("bridge_paths"):
            lines.append("")
            lines.append("## Bridge paths")
            for path in packet["bridge_paths"]:
                lines.append(f"- score={path['score']:.1f}: {path['explanation']}")
        lines.append("")
        lines.append("## Active claims")
        for c in packet["active_claims"]:
            vis = c.get("visibility", "local")
            audiences = ",".join(self.backend._row_audience_roles(c)) or "-"
            tags = ",".join(self.backend._row_interface_tags(c)) or "-"
            lines.append(f"- [{c['confidence']} / {vis} / audience={audiences} / tags={tags}] {c['claim_text']} ({c['claim_id']})")
        lines.append("")
        lines.append("## Related edges")
        for e in packet["related_edges"]:
            vis = e.get("visibility", "local")
            summary = f" — {e['summary']}" if e.get("summary") else ""
            lines.append(f"- {e['from_node_id']} --{e['relation']}--> {e['to_node_id']} ({vis}){summary}")
        if packet.get("cross_role_notes"):
            lines.append("")
            lines.append("## Cross-role visibility notes")
            for n in packet["cross_role_notes"]:
                lines.append(f"- {n['kind']} {n['id']}: {n['visibility']} finder={n.get('finder_role')} audience={n.get('audience_roles')} tags={n.get('interface_tags')} — {n['note']}")
        lines.append("")
        lines.append("## Evidence refs")
        for ev in packet["evidence_refs"]:
            lines.append(f"- {ev['uri']} {ev.get('locator') or ''}: {ev['text_preview']}")
        sem = packet.get("semantic_candidates", {})
        if packet.get("mode") == "mix" or sem.get("enabled"):
            lines.append("")
            lines.append("## Semantic candidates")
            if not sem.get("available"):
                lines.append(f"- unavailable: {sem.get('reason')}")
            else:
                for r in sem.get("results", [])[:8]:
                    rerank_bits = []
                    if r.get("reranker_score") is not None:
                        rerank_bits.append(f"reranker={r.get('reranker_score'):.3f}")
                    if r.get("llm_reranker_score") is not None:
                        rerank_bits.append(f"llm={r.get('llm_reranker_score'):.3f}")
                    suffix = (" " + " ".join(rerank_bits)) if rerank_bits else ""
                    lines.append(f"- {r['kind']} {r['id']} score={r.get('score', 0):.3f}{suffix}: {self.backend._preview(r.get('text', ''))}")
        rerank_trace = packet.get("semantic_candidates", {}).get("rerank_trace")
        if isinstance(rerank_trace, dict) and rerank_trace:
            lines.append("")
            lines.append("## Rerank stages")
            for stage_name, stage in rerank_trace.items():
                if isinstance(stage, dict):
                    lines.append(
                        f"- {stage_name}: enabled={stage.get('enabled')} applied={stage.get('applied')} "
                        f"reason={stage.get('reason')} model={stage.get('model') or '-'}"
                    )
                    for item in stage.get("scored") or []:
                        lines.append(
                            f"  - {item.get('kind')}:{item.get('id')} "
                            f"score={item.get('llm_reranker_score', item.get('reranker_score', '-'))} "
                            f"final={item.get('final_score', '-')} retained={item.get('retained', False)}"
                        )
        if packet.get("missing_links"):
            lines.append("")
            lines.append("## Missing links")
            for m in packet["missing_links"]:
                if m.get("from_node_id"):
                    lines.append(f"- {m['from_node_id']} <-> {m['to_node_id']}: {m['reason']}")
                else:
                    lines.append(f"- {m.get('reason')}: {m.get('anchors')}")
        if packet["stale_or_conflict_warnings"]:
            lines.append("")
            lines.append("## Stale/conflict warnings")
            for w in packet["stale_or_conflict_warnings"]:
                lines.append(f"- {w['status']}: {w['claim_text']} ({w['claim_id']})")
        lines.append("")
        lines.append("## Suggested next checks")
        for c in packet["suggested_next_checks"]:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("## Do not assume")
        for c in packet["do_not_assume"]:
            lines.append(f"- {c}")
        return "\n".join(lines) + "\n"
