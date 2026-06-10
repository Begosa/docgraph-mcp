from __future__ import annotations

from typing import Any

from .server_runtime import build_server_runtime

runtime = build_server_runtime(server_name="docgraph_read", description="DocGraph read-only MCP server")
mcp = runtime.mcp
get_backend = runtime.get_backend
tool = runtime.tool


@tool
def dg_resolve(text: str, limit: int = 10, include_stale: bool = False) -> dict[str, Any]:
    """Resolve a free-text symbol/name/alias to canonical DocGraph nodes."""
    return get_backend().resolve(text=text, limit=limit, include_stale=include_stale)


@tool
def dg_search(query: str, role: str | None = None, intent: str | None = None, limit: int = 10, include_stale: bool = False) -> dict[str, Any]:
    """Search aliases/nodes, curated claims, raw chunks, and shallow graph neighbors."""
    return get_backend().search(query=query, role=role, intent=intent, limit=limit, include_stale=include_stale)


@tool
def dg_context(
    anchors: list[str] | None = None,
    query: str | None = None,
    role: str | None = None,
    intent: str | None = None,
    budget: str = "small",
    mode: str | None = None,
) -> dict[str, Any]:
    """Build a compact context packet. mode: local/global/bridge/hybrid/mix."""
    return get_backend().context(anchors=anchors, query=query, role=role, intent=intent, budget=budget, mode=mode)


@tool
def dg_related_context_check(finding: str, finder_role: str | None = None, interface_tags: list[str] | None = None, limit: int = 10) -> dict[str, Any]:
    """Bounded check for existing high-level/shared context before marking a finding local."""
    return get_backend().related_context_check(finding=finding, finder_role=finder_role, interface_tags=interface_tags, limit=limit)


@tool
def dg_suggest_evidence_relinks(claim_id: str, limit: int = 10) -> dict[str, Any]:
    """Suggest active replacement chunks for stale claim evidence without mutating the graph."""
    return get_backend().suggest_evidence_relinks(claim_id=claim_id, limit=limit)


@tool
def dg_suggest_source_relinks(source_id: str | None = None, uri: str | None = None, limit_per_claim: int = 5, max_claims: int = 500) -> dict[str, Any]:
    """Batch stale-evidence relink suggestions for one changed source without mutating the graph."""
    return get_backend().suggest_source_relinks(source_id=source_id, uri=uri, limit_per_claim=limit_per_claim, max_claims=max_claims)


@tool
def dg_mutation_schema() -> dict[str, Any]:
    """Return the canonical proposal mutation schema, aliases, examples, and visibility contract."""
    return get_backend().mutation_schema()


@tool
def dg_validate() -> dict[str, Any]:
    """Validate graph integrity and claim evidence rules."""
    return get_backend().validate()


def main() -> None:
    runtime.run()


if __name__ == "__main__":
    main()
