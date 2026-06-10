# DocGraph Configuration

`docgraph.config.yaml` is the project policy layer. The MCP backend must remain generic.

## What belongs in config

```text
roles
role-specific retrieval preferences
context budgets
ranking weights
taxonomy node types
taxonomy relation types
claim/evidence enum values
suggested next checks
source/chunk handling defaults
```

## What belongs in code

```text
SQLite schema
transaction boundaries
FTS/BM25 implementation over aliases, nodes, claims, chunks, and edges
proposal/commit mechanics
stale scan execution
validation execution
safe render path enforcement
structured logging
```

## Retrieval impact

`dg_search` applies config in three places:

```text
1. allowed term expansion limits
2. ranking weights
3. role-preferred node/relation boosts
```

`dg_context` applies config in three places:

```text
1. budget sizes: small/medium/large
2. role/intent suggested next checks
3. mode behavior, including bounded semantic anchor promotion/coherence checks in mix
```

## Adding a role

Add a role under `roles`:

```yaml
roles:
  gdb_debug:
    preferred_node_types: [function, file, test, runbook]
    preferred_relations: [debugged_by, calls, depends_on]
    suggested_checks:
      - Verify breakpoints against current binary/source mapping.
```

No MCP code change should be required.

## Source/chunk policy

`source_handling` controls which source types use line-based chunking and which source types are checked by `dg_stale_scan`. This keeps environment-specific file/source categories out of MCP code.

It also controls ingestion safety limits:

```yaml
source_handling:
  max_file_ingest_bytes: 1048576
  max_inline_content_bytes: 262144
  reject_inline_content_for_repo_files: true
```

Set a byte limit to `0` only when a project deliberately accepts unbounded evidence snapshots. Repository-local files should normally be ingested by repo-relative `uri` with `content` omitted; inline `content` is for external/archive/temporary material that cannot be read as a stable repo file.


## Shared visibility / cross-role knowledge

DocGraph is one shared system graph with role-aware retrieval. Nodes, edges, and claims can include `visibility`, `finder_role`, `audience_roles`, and `interface_tags`. This prevents important facts discovered by one role from staying trapped in that role.

Use `dg_related_context_check` before classifying a finding as local when it touches configuration, registers, timing/vsync, channels, data path, interrupts/status, build/generated files, tests, debug, or runbooks.

Visibility values:

```text
local
shared
global
shared_candidate
```

`shared_candidate` means likely cross-role impact but not fully proven. Do not create edges from lexical similarity alone.


## Retrieval model configuration

`retrieval_models.embeddings` and `retrieval_models.reranker` are optional. Default model IDs are `BAAI/bge-large-en-v1.5` and `Qwen/Qwen3-Reranker-0.6B`, both disabled by default. Project config may point at local paths such as `models/Qwen--Qwen3-Reranker-0.6B`; relative local model paths are resolved from `DOCGRAPH_ROOT`. The MCP core uses generic provider adapters, not model-specific retrieval logic.

`retrieval.modes.mix.semantic_anchor_promotion` controls whether semantic results may become anchors. The coherence controls are:

```yaml
require_graph_coherence: true
coherence_min_lexical_anchors: 2
coherence_max_depth: 2
coherence_max_depth_by_budget:
  small: 2
  medium: 2
  large: 3
```

Once enough lexical anchors define a context, graph coherence requires each semantic-only promoted anchor to be reachable through a bounded active visible path from that lexical context. `coherence_max_depth` is the fallback; `coherence_max_depth_by_budget` can widen only larger budgets, for example `large: 3`, without making small/debug retrieval noisy. This prevents disconnected subsystems with similar wording from expanding into context. When semantic promotion is enabled and `require_graph_coherence` is omitted, the backend defaults it to `true` as the safer behavior.

For reranker-only experiments, set:

```yaml
retrieval:
  modes:
    mix:
      semantic_anchor_promotion:
        require_graph_coherence: false
retrieval_models:
  reranker:
    enabled: true
```

This allows semantically similar but graph-disconnected candidates to compete by reranker score. It is useful for measurement, but less safe than graph coherence when multiple subsystems share terms like timeout, underflow, sink, or stride.

## Logging configuration

`logging.level`, `logging.file`, `logging.include_payloads`, and `DOCGRAPH_LOG_LEVEL` control structured JSON-lines MCP/backend logs.


## SQLite/runtime hardening

The backend enables foreign keys, busy timeout, and WAL/synchronous=NORMAL when the filesystem allows it. Schema version and migration metadata are stored in SQLite. Render output is restricted to `DOCGRAPH_ROOT`.
