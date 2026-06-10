# SQLite Schema Summary

Core tables:

```text
metadata(key PK)
schema_migrations(version PK)
sources(source_id PK)
episodes(episode_id PK, source_id FK)
chunks(chunk_id PK, episode_id FK, source_id FK)
nodes(node_id PK, visibility, finder_role, audience_roles_json, interface_tags_json)
aliases(alias_id PK, node_id FK)
edges(edge_id PK, from_node_id FK, to_node_id FK, summary, visibility, finder_role, audience_roles_json, interface_tags_json)
claims(claim_id PK, target_node_id FK nullable, target_edge_id FK nullable, visibility, finder_role, audience_roles_json, interface_tags_json)
claim_evidence(claim_id FK, chunk_id FK)
proposals(proposal_id PK)
commits(commit_id PK, proposal_id FK)
```

FTS/BM25 tables:

```text
aliases_fts
nodes_fts
claims_fts
chunks_fts
edges_fts
```

Retrieval/debug support tables:

```text
node_terms
edge_terms
embedding_items
retrieval_runs
```

`retrieval_runs.trace_json` stores compact context-retrieval diagnostics for the read-only viewer: candidate/anchor decisions, semantic promotion summaries, expanded result IDs, final returned IDs, and stage timings. It does not store embedding vectors or evidence text copies.

Important indexes exist for common foreign keys, visibility/status filtering, aliases, edges, claims, and evidence lookup.

Proof path:

```text
claim -> claim_evidence -> chunk -> episode -> source
```

Active non-open-question claims require an active supporting evidence chunk whose episode and source are also active.

Retrieval path:

```text
query -> alias/node/claim/chunk/edge FTS -> anchors -> visibility filter -> local/global/bridge/hybrid/mix context
```
