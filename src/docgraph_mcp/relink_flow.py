from __future__ import annotations

import difflib
import re
import time
from collections.abc import Callable
from typing import Any

try:
    import sqlite3
except ImportError:  # pragma: no cover - site-specific Python builds may omit stdlib sqlite3
    import pysqlite3 as sqlite3  # type: ignore

from .logging_utils import TRACE_LEVEL


class RelinkFlow:
    """Read-only stale evidence relink suggestion flow."""

    def __init__(
        self,
        backend: Any,
        *,
        fts_query: Callable[[str], str],
        text_hash: Callable[[str], str],
    ) -> None:
        self.backend = backend
        self._fts_query = fts_query
        self._sha256_text = text_hash

    def suggest_evidence_relinks(self, claim_id: str, limit: int = 10) -> dict[str, Any]:
        """Suggest active replacement chunks for stale claim evidence.

        This is intentionally read-only. It helps a curator repair evidence links
        after a source was re-ingested, but it does not attach evidence or mark a
        claim active. Exact/near code-equivalence is treated as stronger than
        semantic similarity because evidence relinking changes proof, not search.
        """
        started = time.perf_counter()
        limit = max(1, min(int(limit), 50))
        claim = self.backend.conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        if not claim:
            raise ValueError(f"unknown claim_id: {claim_id}")
        claim_d = dict(claim)
        stale_supports = self._stale_support_chunks_for_claim(claim_id)
        active_supports = self._active_support_chunks_for_claim(claim_id)
        candidates_by_chunk: dict[str, dict[str, Any]] = {}

        for stale in stale_supports:
            for candidate in self._active_chunks_for_source(stale["source_id"]):
                if candidate["chunk_id"] == stale["chunk_id"]:
                    continue
                scored = self._score_relink_candidate(claim_d, stale, candidate, same_source=True)
                self._merge_relink_candidate(candidates_by_chunk, scored, stale)

        if len(candidates_by_chunk) < limit:
            for stale in stale_supports:
                query = self._fts_query(f"{claim_d.get('claim_text', '')} {stale.get('text', '')}")
                if not query:
                    continue
                try:
                    rows = self.backend.conn.execute(
                        """
                        SELECT ch.*, bm25(chunks_fts) AS rank, s.uri, s.source_type
                        FROM chunks_fts f
                        JOIN chunks ch ON ch.chunk_id=f.chunk_id
                        JOIN sources s ON s.source_id=ch.source_id
                        WHERE chunks_fts MATCH ?
                          AND ch.status='active'
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, limit * 4),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    self.backend._log("debug", "suggest_evidence_relinks.fts_error", claim_id=claim_id, error_type=type(exc).__name__, error=str(exc))
                    rows = []
                for row in rows:
                    candidate = dict(row)
                    if candidate["chunk_id"] == stale["chunk_id"]:
                        continue
                    scored = self._score_relink_candidate(
                        claim_d,
                        stale,
                        candidate,
                        same_source=candidate.get("source_id") == stale.get("source_id"),
                        fts_rank=float(candidate.get("rank") or 0.0),
                    )
                    self._merge_relink_candidate(candidates_by_chunk, scored, stale)

        candidates = sorted(candidates_by_chunk.values(), key=lambda c: (-float(c["score"]), c["chunk_id"]))[:limit]
        equivalent = [c for c in candidates if c["support_level"] == "equivalent"]
        recommendation = "no_candidate_found"
        if equivalent:
            recommendation = "safe_equivalent_relink_candidate"
        elif candidates:
            recommendation = "review_candidates"

        draft_mutations = [
            {
                "op": "attach_evidence",
                "claim_id": claim_id,
                "chunk_id": c["chunk_id"],
                "evidence_role": "supports",
                "strength": "high" if c["support_level"] == "equivalent" else "medium",
            }
            for c in equivalent[:1]
        ]

        result = {
            "claim": {
                "claim_id": claim_d["claim_id"],
                "status": claim_d["status"],
                "classification": claim_d["classification"],
                "confidence": claim_d["confidence"],
                "claim_text": claim_d["claim_text"],
            },
            "active_support_count": len(active_supports),
            "stale_support_count": len(stale_supports),
            "stale_supports": [self._chunk_relink_summary(c) for c in stale_supports],
            "candidates": candidates,
            "recommendation": recommendation,
            "draft_mutations": draft_mutations,
            "rules": [
                "Read-only suggestion only; no database mutation was performed.",
                "Equivalent candidates still require curator review, proposal validation, commit, and render.",
                "If current source changed behavior, create contradiction/supersession instead of relinking.",
            ],
        }
        self.backend._log(
            "info",
            "suggest_evidence_relinks.done",
            claim_id=claim_id,
            stale_supports=len(stale_supports),
            active_supports=len(active_supports),
            candidates=len(candidates),
            recommendation=recommendation,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self.backend._log(TRACE_LEVEL, "suggest_evidence_relinks.result", result=result)
        return result

    def suggest_source_relinks(
        self,
        source_id: str | None = None,
        uri: str | None = None,
        limit_per_claim: int = 5,
        max_claims: int = 500,
    ) -> dict[str, Any]:
        """Batch stale-evidence relink suggestions for one source.

        This is the source-level wrapper around ``suggest_evidence_relinks``.
        It keeps the repetitive per-claim loop inside the MCP backend so the
        agent/curator receives one grouped, read-only result.
        """
        started = time.perf_counter()
        limit_per_claim = max(1, min(int(limit_per_claim), 25))
        max_claims = max(1, min(int(max_claims), 2000))
        source = self._source_for_relink(source_id=source_id, uri=uri)
        source_id = source["source_id"]
        affected_claim_ids = self._affected_claim_ids_for_source(source_id, max_claims=max_claims)

        safe_relinks: list[dict[str, Any]] = []
        review_candidates: list[dict[str, Any]] = []
        unresolved_claims: list[dict[str, Any]] = []
        already_supported: list[dict[str, Any]] = []
        claim_results: list[dict[str, Any]] = []
        draft_mutations: list[dict[str, Any]] = []

        for claim_id in affected_claim_ids:
            stale_supports_for_source = [c for c in self._stale_support_chunks_for_claim(claim_id) if c["source_id"] == source_id]
            stale_chunk_ids_for_source = {c["chunk_id"] for c in stale_supports_for_source}
            suggestion = self.suggest_evidence_relinks(claim_id=claim_id, limit=limit_per_claim)
            source_candidates = [
                c
                for c in suggestion["candidates"]
                if c.get("source_id") == source_id and stale_chunk_ids_for_source.intersection(c.get("matched_stale_chunk_ids", []))
            ]
            safe_candidates = [c for c in source_candidates if c.get("support_level") == "equivalent"]
            review_only_candidates = [c for c in source_candidates if c.get("support_level") != "equivalent"]
            claim_summary = {
                "claim": suggestion["claim"],
                "active_support_count": suggestion["active_support_count"],
                "stale_support_count_for_source": len(stale_supports_for_source),
                "stale_supports_for_source": [self._chunk_relink_summary(c) for c in stale_supports_for_source],
                "candidate_count_for_source": len(source_candidates),
                "recommendation": "no_candidate_found",
            }

            if safe_candidates:
                best = safe_candidates[0]
                mutation = {
                    "op": "attach_evidence",
                    "claim_id": claim_id,
                    "chunk_id": best["chunk_id"],
                    "evidence_role": "supports",
                    "strength": "high",
                }
                item = claim_summary | {
                    "recommendation": "safe_equivalent_relink_candidate",
                    "candidate": best,
                    "draft_mutation": mutation,
                }
                safe_relinks.append(item)
                draft_mutations.append(mutation)
                claim_results.append(item)
            elif review_only_candidates:
                item = claim_summary | {
                    "recommendation": "review_candidates",
                    "candidates": review_only_candidates[:limit_per_claim],
                }
                review_candidates.append(item)
                claim_results.append(item)
            elif suggestion["active_support_count"] > 0:
                item = claim_summary | {
                    "recommendation": "already_has_active_support",
                }
                already_supported.append(item)
                claim_results.append(item)
            else:
                item = claim_summary | {
                    "recommendation": "no_candidate_found",
                }
                unresolved_claims.append(item)
                claim_results.append(item)

        result = {
            "source": {
                "source_id": source["source_id"],
                "uri": source["uri"],
                "source_type": source["source_type"],
                "name": source["name"],
                "status": source["status"],
                "current_hash": source["current_hash"],
            },
            "summary": {
                "affected_claim_count": len(affected_claim_ids),
                "safe_relink_claim_count": len(safe_relinks),
                "review_candidate_claim_count": len(review_candidates),
                "unresolved_claim_count": len(unresolved_claims),
                "already_supported_claim_count": len(already_supported),
                "draft_mutation_count": len(draft_mutations),
                "truncated": len(affected_claim_ids) >= max_claims,
            },
            "safe_relinks": safe_relinks,
            "review_candidates": review_candidates,
            "unresolved_claims": unresolved_claims,
            "already_supported": already_supported,
            "claim_results": claim_results,
            "draft_mutations": draft_mutations,
            "rules": [
                "Read-only batch suggestion only; no database mutation was performed.",
                "Safe relinks require exact chunk content-hash equality or same-source normalized-token hash equality.",
                "Review candidates from difflib/token/locator/FTS/LLM signals must be inspected before proposing evidence changes.",
                "Draft mutations attach replacement evidence only; curator must decide whether to mark claims active after validation.",
            ],
        }
        self.backend._log(
            "info",
            "suggest_source_relinks.done",
            source_id=source_id,
            uri=source["uri"],
            affected_claims=len(affected_claim_ids),
            safe_relinks=len(safe_relinks),
            review_candidates=len(review_candidates),
            unresolved_claims=len(unresolved_claims),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self.backend._log(TRACE_LEVEL, "suggest_source_relinks.result", result=result)
        return result

    def _source_for_relink(self, source_id: str | None = None, uri: str | None = None) -> dict[str, Any]:
        if bool(source_id) == bool(uri):
            raise ValueError("provide exactly one of source_id or uri")
        if source_id:
            row = self.backend.conn.execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
        else:
            row = self.backend.conn.execute("SELECT * FROM sources WHERE uri=?", (uri,)).fetchone()
        if not row:
            ref = source_id if source_id else uri
            raise ValueError(f"unknown source: {ref}")
        return dict(row)

    def _affected_claim_ids_for_source(self, source_id: str, max_claims: int) -> list[str]:
        return [
            str(r["claim_id"])
            for r in self.backend.conn.execute(
                """
                SELECT DISTINCT ce.claim_id
                FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ch.source_id=?
                  AND ce.evidence_role='supports'
                  AND (ce.status <> 'active' OR ch.status <> 'active' OR ep.status <> 'active' OR s.status <> 'active')
                ORDER BY ce.claim_id
                LIMIT ?
                """,
                (source_id, max_claims),
            )
        ]

    def _active_support_chunks_for_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return self._support_chunks_for_claim(claim_id, active=True)

    def _stale_support_chunks_for_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return self._support_chunks_for_claim(claim_id, active=False)

    def _support_chunks_for_claim(self, claim_id: str, active: bool) -> list[dict[str, Any]]:
        if active:
            status_filter = "ce.status='active' AND ch.status='active' AND ep.status='active' AND s.status='active'"
        else:
            status_filter = "(ce.status <> 'active' OR ch.status <> 'active' OR ep.status <> 'active' OR s.status <> 'active')"
        return [
            dict(r)
            for r in self.backend.conn.execute(
                f"""
                SELECT ce.evidence_role, ce.strength, ce.status AS evidence_status,
                       ch.chunk_id, ch.episode_id, ch.source_id, ch.locator, ch.text,
                       ch.content_hash, ch.status AS chunk_status,
                       ep.status AS episode_status,
                       s.uri, s.source_type, s.status AS source_status
                FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ce.claim_id=?
                  AND ce.evidence_role='supports'
                  AND {status_filter}
                ORDER BY ce.created_at DESC
                """,
                (claim_id,),
            )
        ]

    def _active_chunks_for_source(self, source_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.backend.conn.execute(
                """
                SELECT ch.*, s.uri, s.source_type
                FROM chunks ch
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ch.source_id=?
                  AND ch.status='active'
                  AND ep.status='active'
                  AND s.status='active'
                ORDER BY ch.rowid
                """,
                (source_id,),
            )
        ]

    def _score_relink_candidate(
        self,
        claim: dict[str, Any],
        stale: dict[str, Any],
        candidate: dict[str, Any],
        *,
        same_source: bool,
        fts_rank: float | None = None,
    ) -> dict[str, Any]:
        stale_norm = self._evidence_normalized_text(stale.get("text", ""))
        candidate_norm = self._evidence_normalized_text(candidate.get("text", ""))
        stale_norm_hash = self._sha256_text(stale_norm) if stale_norm else ""
        candidate_norm_hash = self._sha256_text(candidate_norm) if candidate_norm else ""
        stale_tokens = self._evidence_token_set(stale.get("text", ""))
        candidate_tokens = self._evidence_token_set(candidate.get("text", ""))
        claim_tokens = self._evidence_token_set(claim.get("claim_text", ""))

        reasons: list[str] = []
        score = 0.0
        support_level = "weak"
        if same_source:
            score += 25.0
            reasons.append("same source identity")

        if stale.get("content_hash") and stale.get("content_hash") == candidate.get("content_hash"):
            score += 180.0
            if same_source:
                support_level = "equivalent"
                reasons.append("exact chunk content hash matches in the same source")
            else:
                support_level = "strong_candidate"
                reasons.append("exact chunk content hash matches in a different source")
        elif stale_norm_hash and stale_norm_hash == candidate_norm_hash:
            score += 120.0
            if same_source:
                support_level = "equivalent"
                reasons.append("normalized token hash matches in the same source")
            else:
                support_level = "strong_candidate"
                reasons.append("normalized token hash matches in a different source")
        elif stale_norm and candidate_norm and (stale_norm in candidate_norm or candidate_norm in stale_norm):
            score += 85.0
            support_level = "strong_candidate"
            reasons.append("normalized text/code contains the stale evidence")

        sequence_ratio = 0.0
        if stale_norm and candidate_norm:
            sequence_ratio = difflib.SequenceMatcher(None, stale_norm, candidate_norm, autojunk=False).ratio()
            if sequence_ratio >= 0.90:
                score += sequence_ratio * 35.0
                reasons.append(f"normalized sequence similarity {sequence_ratio:.2f}")
                if support_level == "weak":
                    support_level = "review_candidate"

        token_overlap = self._jaccard(stale_tokens, candidate_tokens)
        if token_overlap:
            score += token_overlap * 60.0
            reasons.append(f"stale/candidate token overlap {token_overlap:.2f}")
        if support_level == "weak" and token_overlap >= 0.82:
            support_level = "strong_candidate"
        elif support_level == "weak" and token_overlap >= 0.55:
            support_level = "review_candidate"

        claim_overlap = self._jaccard(claim_tokens, candidate_tokens)
        if claim_overlap:
            score += claim_overlap * 25.0
            reasons.append(f"claim/candidate token overlap {claim_overlap:.2f}")

        line_distance = self._locator_line_distance(stale.get("locator"), candidate.get("locator"))
        if line_distance is not None:
            bonus = max(0.0, 20.0 - min(float(line_distance), 200.0) / 10.0)
            if bonus:
                score += bonus
                reasons.append(f"near previous locator ({line_distance} lines)")

        if fts_rank is not None:
            score += max(0.0, 8.0 - min(abs(float(fts_rank)), 8.0))
            reasons.append("FTS candidate")

        return {
            "chunk_id": candidate["chunk_id"],
            "source_id": candidate["source_id"],
            "uri": candidate.get("uri"),
            "source_type": candidate.get("source_type"),
            "locator": candidate.get("locator"),
            "score": round(score, 3),
            "support_level": support_level,
            "reasons": reasons,
            "text_preview": self.backend._preview(candidate.get("text", "")),
            "matched_stale_chunk_ids": [stale["chunk_id"]],
        }

    def _merge_relink_candidate(self, candidates_by_chunk: dict[str, dict[str, Any]], candidate: dict[str, Any], stale: dict[str, Any]) -> None:
        existing = candidates_by_chunk.get(candidate["chunk_id"])
        if existing is None:
            candidates_by_chunk[candidate["chunk_id"]] = candidate
            return
        existing["matched_stale_chunk_ids"] = sorted(set(existing["matched_stale_chunk_ids"]) | {stale["chunk_id"]})
        existing["reasons"] = sorted(set(existing["reasons"]) | set(candidate["reasons"]))
        if float(candidate["score"]) > float(existing["score"]):
            existing["score"] = candidate["score"]
            existing["support_level"] = candidate["support_level"]
            existing["locator"] = candidate["locator"]
            existing["text_preview"] = candidate["text_preview"]

    def _chunk_relink_summary(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "uri": chunk.get("uri"),
            "source_type": chunk.get("source_type"),
            "locator": chunk.get("locator"),
            "evidence_status": chunk.get("evidence_status"),
            "chunk_status": chunk.get("chunk_status"),
            "episode_status": chunk.get("episode_status"),
            "source_status": chunk.get("source_status"),
            "text_preview": self.backend._preview(chunk.get("text", "")),
        }

    def _evidence_normalized_text(self, text: str) -> str:
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = re.sub(r"//.*?$", " ", text, flags=re.M)
        tokens = re.findall(
            r"0x[0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||<<|>>|->|::|[{}()\[\];,.:?~!%^&*+=|/<>-]",
            text,
        )
        return " ".join(tokens)

    def _evidence_token_set(self, text: str) -> set[str]:
        norm = self._evidence_normalized_text(text).lower()
        return {t for t in norm.split() if len(t) > 1}

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _locator_line_distance(self, left: str | None, right: str | None) -> int | None:
        left_line = self._locator_start_line(left)
        right_line = self._locator_start_line(right)
        if left_line is None or right_line is None:
            return None
        return abs(left_line - right_line)

    def _locator_start_line(self, locator: str | None) -> int | None:
        if not locator:
            return None
        match = re.search(r"(?:line|lines)\s+(\d+)", locator, flags=re.I)
        return int(match.group(1)) if match else None


