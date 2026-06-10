# DocGraph Tool Contract

Agents must use MCP tools instead of reading SQLite directly.

## Read/search tools

Exposed by the `docgraph_read` MCP server as OpenCode tools prefixed with `docgraph_read_`:

```text
docgraph_read_dg_resolve
docgraph_read_dg_search
docgraph_read_dg_context
docgraph_read_dg_related_context_check
docgraph_read_dg_suggest_evidence_relinks
docgraph_read_dg_suggest_source_relinks
docgraph_read_dg_mutation_schema
docgraph_read_dg_validate
```

`docgraph_read_dg_related_context_check` is a bounded helper for curator/retriever use. It helps decide whether a finding is local/shared/global/shared_candidate by searching related high-level context. It does not mutate graph state.

`docgraph_read_dg_suggest_evidence_relinks(claim_id, limit=10)` is a read-only stale-review helper. It ranks active chunks that may replace stale supporting chunks for a claim. Safe draft relinks require exact chunk content-hash equality or same-source normalized-token hash equality. Fuzzy sequence similarity, token overlap, FTS/BM25, and LLM semantic judgment are review-only signals and must not produce automatic relinks. The tool may return draft `attach_evidence` mutations for curator review, but it does not attach evidence, mark the claim active, or decide whether changed behavior should supersede/contradict the old claim.

`docgraph_read_dg_suggest_source_relinks(source_id=... | uri=..., limit_per_claim=5, max_claims=500)` is the source-level batch helper. It finds all claims with stale supporting chunks from the changed source, runs the same safe comparison inside the backend, groups safe relinks/review candidates/unresolved claims, and returns one batch `draft_mutations` list. It is read-only and does not commit, attach evidence, or reactivate claims.

## Existing Atlassian read tools

This repository does not declare an Atlassian MCP server. It relies on the user's existing local `atlassian-sirc` MCP setup.

Because the concrete `atlassian-sirc` tool names are environment-owned, `opencode.jsonc` allows the `atlassian-sirc_*` prefix only for `context-retriever`. Narrow this wildcard to concrete read-only tool names when the exact tool list is known.

```text
atlassian-sirc_*  # context-retriever only
```

The retriever may use only tools whose name/description clearly means:

```text
search
fetch
get
read
```

Confluence/design-doc results are design intent or architecture intent. They are useful for high-level perspective, feature expectations, project/block layout, and intended behavior, but they are not proof of current implementation. Specialists must validate against current source, RTL, logs, simulations, or emulation evidence.

All other Atlassian tools are denied by default. In particular, the retriever may not create/edit/update/delete/comment/transition Jira issues, modify Confluence, or call worklog/write/mutation tools. A Jira result is a historical report or workaround candidate that a specialist must validate against current evidence.

## Mutation tools

Only the curator should use these through the `docgraph_write` MCP server:

```text
docgraph_write_dg_ingest_source
docgraph_write_dg_ingest_investigation_report
docgraph_write_dg_mutation_schema
docgraph_write_dg_propose_update
docgraph_write_dg_commit_update
docgraph_write_dg_validate
docgraph_write_dg_render_docs
docgraph_write_dg_stale_scan(auto_ingest=true)
```

## Correct mutation order

```text
ingest evidence if needed
capture returned chunk_refs/chunk_ids
resolve aliases
search duplicates/conflicts
run related-context check if cross-role impact is possible
propose update with visibility/finder/audience/tags
validate
commit
render docs
validate
```

## Mutation compatibility notes

`docgraph_write_dg_ingest_source` returns `chunk_ids` and `chunk_refs`; curators must use those exact IDs for evidence and must not construct chunk IDs from source IDs, episode IDs, file paths, or counts.

When re-ingesting a changed source, `docgraph_write_dg_ingest_source` also returns `stale_chunk_ids`, `affected_claim_ids`, and `claims_marked_needs_review`. Use these fields to start Flow 6 stale review directly instead of guessing which claims were invalidated.

Before calling `docgraph_write_dg_ingest_source`, decide path-vs-inline ingestion. Repository-local stable files should use path ingestion: pass a repo-relative `uri` and omit `content`. Do not paste full repository file contents into the tool call; the backend reads and snapshots the file under `DOCGRAPH_ROOT`. External, temporary, or not-to-be-kept documents must use inline `content` with a stable non-file URI.

Do not ingest a source merely because it was inspected during navigation. Ingest only when its chunks are needed as evidence for a durable claim, contradiction, hypothesis, inference, or explicit open-question context.

The backend enforces ingestion guardrails from `source_handling`: oversized file-backed or inline snapshots are rejected, and inline `content` for an existing repository-local file is rejected by default. Raise limits only when the curator deliberately needs that evidence.

When `content` is omitted, `docgraph_write_dg_ingest_source` reads `uri` as a file under `DOCGRAPH_ROOT`; paths outside the project root are rejected. For external/archive documents that should not stay in the repository, pass inline `content`, use a stable non-file URI such as `archive://old-docs/name.md`, and choose a non-file-backed `source_type` such as `historical_doc` so stale scan does not later report the deleted file as missing.

`docgraph_write_dg_propose_update` accepts canonical mutation ops and common aliases. Curators should prefer canonical ops and call `docgraph_write_dg_mutation_schema` before proposing updates so taxonomy/visibility constraints are explicit. `upsert_claim` may include `chunk_ids`, which the backend expands into `attach_evidence`. Proposal references are preflighted before commit, so missing node, edge, claim, and chunk references fail with typed errors instead of raw SQLite foreign-key errors. Generated rendered docs such as `docs/rendered/*` and files marked `Generated by render_docs` must not be ingested as source evidence.


## Context modes

`docgraph_read_dg_context` accepts `mode`: `local`, `global`, `bridge`, `hybrid`, or `mix`. `mix` uses optional embeddings/reranker if enabled; when disabled, it returns graph context and reports semantic retrieval unavailable.


## Search coverage

`docgraph_read_dg_search` searches alias FTS, node summary FTS, claim FTS, chunk FTS, edge summary FTS, and one-hop graph neighbors. Node and edge summaries are searchable relationship/profile text, not trusted evidence by themselves.
