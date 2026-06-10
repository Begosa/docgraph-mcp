# MCP and Backend Logging

DocGraph writes structured JSON-lines logs so MCP interactions, retrieval, and curation can be debugged.

## Configuration

```yaml
logging:
  enabled: true
  level: info        # off | error | warning | info | debug | trace
  file: docs/logs/docgraph-mcp.log
  max_bytes: 5000000
  backup_count: 3
  stderr: false
  include_payloads: false
  payload_preview_chars: 500
```

Temporary overrides:

```bash
DOCGRAPH_LOG_LEVEL=debug
DOCGRAPH_LOG_FILE=/tmp/docgraph-mcp.log
```

## Main events

MCP boundary:

```text
mcp.tool.start
mcp.tool.done
mcp.tool.error
```

Backend lifecycle:

```text
backend.init.start
backend.init.done
schema.init.done
schema.migrate.start
schema.migrate.done
schema.column_added
```

Retrieval:

```text
resolve.start / resolve.done
search.start / search.done
context.start / context.done
context.global.done
context.bridge.done
context.semantic.done
related_context_check.start / related_context_check.done
retrieval_run.recorded
```

Models:

```text
semantic.disabled_or_unavailable
semantic.embed.start
semantic.done
semantic.embed.error
reranker.start
reranker.done
reranker.skipped
reranker.error
```

Curation/write:

```text
ingest.start / ingest.unchanged / ingest.done / ingest.rejected
proposal.start / proposal.done
commit.start / mutation.apply / commit.done / commit.error
validate.start / validate.done
render_docs.start / render_docs.done
stale_scan.start / stale_scan.done
claim.needs_review.no_active_support
```

## Payload policy

By default `include_payloads: false` logs compact previews and shapes, not full source/report text. Use `trace` and `include_payloads: true` only for short local debugging sessions.

Ingestion events include compact size metadata such as `inline_content_bytes`, `content_bytes`, and `chunks`. Rejected ingestion logs include the reason and configured byte limit without storing the rejected payload.
