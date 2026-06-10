# Retrieval Flow

## Retrieval effort policy

Retrieval should be cheapest-sufficient, not maximal.

```text
none          = simple process/design question or direct action
resolve       = identity mapping only
local/small   = exact entity
bridge/small  = relation between two anchors
global/medium = broad lifecycle/system frame
hybrid/medium = real investigation/debug/development ambiguity
mix           = fuzzy discovery only after exact/FTS/local failed
```

Default config should be cheap (`local`/`small`). The context-retriever escalates only when the previous effort is insufficient.

## Source Boundary

The context-retriever is a constrained background-knowledge role:

```text
allowed = DocGraph read tools + approved read/search Jira tools
allowed = approved read-only Confluence/design-doc tools when design intent is needed
denied  = workspace files, current source/log inspection, build execution, Atlassian mutation
```

This repository does not declare an Atlassian MCP server; it relies on the user's existing local `atlassian-sirc` MCP setup. The requesting specialist supplies current-evidence anchors after inspecting code/logs/tests or executing a build. Confluence/design-doc results are intended behavior/spec context and Jira results are historical reports or workaround candidates. Both must be validated against current evidence before being treated as current system behavior.

`dg_search` uses multiple entry points:

```text
query
  ├─ exact alias/node resolution
  ├─ alias FTS/BM25
  ├─ node canonical_name/summary FTS/BM25
  ├─ claim FTS/BM25
  ├─ chunk FTS/BM25
  ├─ edge relation/summary FTS/BM25
  └─ one-hop graph expansion from discovered anchors
        ↓
rank by exact match, BM25, role preferences, visibility/audience, relation fit, graph proximity
```

## Role-aware visibility

For a role request, retrieval includes:

```text
role-local facts
shared facts relevant to the role
global facts
shared_candidate facts with warnings
```

It excludes another role's `local` details unless the requesting role is listed in `audience_roles`.

## Context modes

```text
local  = direct node neighborhood.
global = high-level flow/feature/concept/interface/runbook context.
bridge = bounded graph paths between anchors.
hybrid = local + global + bridge.
mix    = hybrid + optional embedding candidates + optional reranker.
```

`bridge` is a retrieval path made from existing edges. It is not a relation type.

## Context packet

`dg_context` returns:

```text
selected anchors
global frames
bridge paths
active claims
related edges
evidence refs
semantic candidates, if mix mode
cross-role visibility notes
missing links
stale/conflict warnings
suggested next checks
```

## Related-context check

`dg_related_context_check` performs a bounded check before the curator marks a finding local.

It searches the finding and interface tags, prefers high-level nodes, and returns candidate flows/features/concepts/interfaces/runbooks. It does not create edges or claims.

Important rule:

```text
Do not create edges from lexical similarity alone.
If impact is likely but relation is unproven, use shared_candidate or OpenQuestion.
```

## Optional models

`mix` mode calls the embedding provider and then the optional reranker if configured.

```text
embeddings = discovery candidates
reranker   = candidate ordering
DocGraph   = trusted memory only after curator commits evidence-backed claims
```

Embeddings/reranker are disabled by default and are behind generic adapters.
Local model paths configured under `retrieval_models.*.model` are resolved from `DOCGRAPH_ROOT`, so `models/Qwen--Qwen3-Reranker-0.6B` works even when the MCP server is launched from another directory.

## Semantic Promotion Coherence

When `mix` has already found enough lexical anchors to define a context, semantic-only anchors are not expanded merely because their text is similar. With `require_graph_coherence` enabled, a new semantic anchor must have a bounded active visible-edge path to the lexical anchors before it enters local/global/bridge expansion.

```text
lexical MIPI/DMA anchors + connected VSYNC semantic hit -> promote
lexical MIPI/DMA anchors + disconnected audio-underflow semantic hit -> keep as hint only
few/no lexical anchors -> allow semantic discovery to recover context
```

This prevents repeated subsystem vocabulary such as `sink`, `timeout`, `underflow`, and `stride` from pulling disconnected claims and evidence into an otherwise coherent context packet.

Coherence depth can be budget-aware. The default keeps `small` and `medium` at depth `2`, while `large` may use depth `3` for broader architecture investigations:

```yaml
coherence_max_depth_by_budget: {small: 2, medium: 2, large: 3}
```

Set `require_graph_coherence: false` to deliberately test whether the reranker can reject disconnected semantic matches on score alone. That experiment measures reranker quality, but it removes the graph safety guard.
