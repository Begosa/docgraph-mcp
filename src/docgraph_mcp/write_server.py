from __future__ import annotations

from typing import Any

from .server_runtime import build_server_runtime

runtime = build_server_runtime(server_name="docgraph_write", description="DocGraph write/curation MCP server")
mcp = runtime.mcp
get_backend = runtime.get_backend
tool = runtime.tool


@tool
def dg_ingest_source(
    source_type: str,
    uri: str,
    content: str | None = None,
    episode_type: str = "snapshot",
    name: str | None = None,
    evidence_hint: str | None = None,
    claim_text: str | None = None,
    evidence_lines: list[dict[str, int]] | None = None,
    recommend_limit: int = 5,
) -> dict[str, Any]:
    """Ingest a source and optionally recommend supporting chunks from that source only."""
    return get_backend().ingest_source(
        source_type=source_type,
        uri=uri,
        content=content,
        episode_type=episode_type,
        name=name,
        evidence_hint=evidence_hint,
        claim_text=claim_text,
        evidence_lines=evidence_lines,
        recommend_limit=recommend_limit,
    )


@tool
def dg_ingest_investigation_report(
    title: str,
    report: str,
    created_by: str = "specialist",
    evidence_hint: str | None = None,
    claim_text: str | None = None,
    evidence_lines: list[dict[str, int]] | None = None,
    recommend_limit: int = 5,
) -> dict[str, Any]:
    """Ingest an agent/specialist investigation report and optionally recommend supporting chunks."""
    from .backend import sha256_text

    digest = sha256_text(f"{created_by}:{title}:{report}")[:16]
    uri = f"agent-report:{created_by}:{digest}"
    return get_backend().ingest_source(
        source_type="agent_report",
        uri=uri,
        content=report,
        episode_type="agent_investigation",
        name=title,
        evidence_hint=evidence_hint,
        claim_text=claim_text,
        evidence_lines=evidence_lines,
        recommend_limit=recommend_limit,
    )


@tool
def dg_mutation_schema(detail: str = "compact") -> dict[str, Any]:
    """Return the proposal mutation schema. detail: compact/full."""
    return get_backend().mutation_schema(detail=detail)


@tool
def dg_propose_update(reason: str, mutations: list[dict[str, Any]], created_by: str = "curator") -> dict[str, Any]:
    """Create a pending curated graph update proposal. Does not mutate graph truth."""
    return get_backend().propose_update(reason=reason, mutations=mutations, created_by=created_by)


@tool
def dg_commit_update(proposal_id: str, error_limit: int = 20) -> dict[str, Any]:
    """Validate and atomically commit a pending proposal."""
    return get_backend().commit_update(proposal_id=proposal_id, error_limit=error_limit)


@tool
def dg_validate(limit: int = 20, detail: str = "compact") -> dict[str, Any]:
    """Validate graph integrity and claim evidence rules."""
    return get_backend().validate(limit=limit, detail=detail)


@tool
def dg_render_docs(output_dir: str | None = None) -> dict[str, Any]:
    """Render Markdown docs from SQLite DocGraph state."""
    return get_backend().render_docs(output_dir=output_dir)


@tool
def dg_stale_scan(
    auto_ingest: bool = False,
    source_id: str | None = None,
    uri: str | None = None,
    max_sources: int = 20,
    result_limit: int = 50,
    detail: str = "compact",
) -> dict[str, Any]:
    """Detect changed/missing file-like sources and optionally re-ingest a bounded filtered set."""
    return get_backend().stale_scan(
        auto_ingest=auto_ingest,
        source_id=source_id,
        uri=uri,
        max_sources=max_sources,
        result_limit=result_limit,
        detail=detail,
    )


def main() -> None:
    runtime.run()


if __name__ == "__main__":
    main()
