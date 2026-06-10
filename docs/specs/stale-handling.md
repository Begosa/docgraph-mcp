# Stale Handling

Evidence is not deleted by default.

When a source changes:

```text
same source_id
new episode_id
new chunks
old episode marked superseded
old chunks marked stale/removed
claim_evidence linked to old chunks marked stale
claims depending only on stale evidence marked needs_review
dg_suggest_evidence_relinks can suggest active replacement chunks
dg_suggest_source_relinks can batch suggestions for all affected claims from one source
```

Claim statuses:

```text
active
needs_review
stale
superseded
contradicted
retired
```

A claim should become stale only when its usable evidence is gone or contradicted.

## Relink review

When a file changes but the relevant code/doc text is unchanged or equivalently moved, use `docgraph_read_dg_suggest_source_relinks(source_id=... | uri=...)` first. It batches all claims affected by that source and returns grouped safe/review/unresolved results plus one draft mutation list. Use `docgraph_read_dg_suggest_evidence_relinks(claim_id)` for a focused per-claim follow-up. Both tools are read-only and return draft `attach_evidence` mutations only for exact chunk content-hash matches or same-source normalized-token hash matches.

Relink confidence requirements:

```text
safe equivalent      = same source plus exact chunk content_hash match, or same source plus normalized-token hash match
strong candidate     = exact/normalized hash match in a different source, or normalized containment
review candidate     = difflib sequence similarity, token overlap, locator proximity, FTS/BM25, or LLM semantic judgment
manual only          = changed behavior, conflicting evidence, weak/ambiguous candidate, or no replacement
```

Curator decision:

```text
equivalent current chunk supports same claim -> propose attach_evidence to new chunk, then mark claim active if support is sufficient
current chunk changed behavior              -> create contradiction/supersession or leave needs_review
only weak semantic/lexical candidate         -> inspect current evidence manually before proposing anything
no candidate                                -> leave needs_review/open question and report missing evidence
```
