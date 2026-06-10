from __future__ import annotations

from typing import Any, TypedDict


class SemanticCandidateResult(TypedDict, total=False):
    kind: str
    id: str
    score: float
    text: str
    embedding_score: float
    reranker_score: float
    llm_reranker_score: float


class _SemanticCandidatesRequired(TypedDict):
    enabled: bool
    available: bool
    reason: str | None
    results: list[SemanticCandidateResult]


class SemanticCandidates(_SemanticCandidatesRequired, total=False):
    rerank_trace: dict[str, Any]


class ContextPacket(TypedDict):
    mode: str
    role: str | None
    intent: str | None
    budget: str
    query: str | None
    config_path: str | None
    selected_anchors: list[dict[str, Any]]
    global_frames: list[dict[str, Any]]
    bridge_paths: list[dict[str, Any]]
    active_claims: list[dict[str, Any]]
    related_edges: list[dict[str, Any]]
    evidence_refs: list[dict[str, Any]]
    semantic_candidates: SemanticCandidates
    cross_role_notes: list[dict[str, Any]]
    missing_links: list[dict[str, Any]]
    stale_or_conflict_warnings: list[dict[str, Any]]
    suggested_next_checks: list[str]
    do_not_assume: list[str]
    markdown: str
