# DocGraph MCP

Evidence-backed memory for engineering agents.

DocGraph is a SQLite/FTS-backed Model Context Protocol (MCP) server for curated project knowledge. It separates raw evidence from durable claims, links claims to supporting chunks, tracks stale evidence, and supports role-aware retrieval for coding and engineering agents.

## Why

Most agent memory systems make it easy to store conclusions. DocGraph is intentionally stricter:

```text
source/report -> episode -> chunks -> claim_evidence -> claim -> node/edge context
```

A durable claim should point back to evidence. If the evidence changes, the claim can be marked for review instead of silently remaining trusted.

## MCP Servers

Recommended split:

```text
docgraph_read   -> read-only retrieval/context tools
docgraph_write  -> curator-only ingestion/proposal/commit/render tools
```

Compatibility server:

```text
docgraph_mcp.server -> all tools for local/manual debugging
```

## Tools

Read tools:

```text
dg_resolve
dg_search
dg_context
dg_related_context_check
dg_suggest_evidence_relinks
dg_suggest_source_relinks
dg_mutation_schema
dg_validate
```

Write tools:

```text
dg_ingest_source
dg_ingest_investigation_report
dg_mutation_schema
dg_propose_update
dg_commit_update
dg_validate
dg_render_docs
dg_stale_scan
```

## Install

```bash
python3 -m pip install -e .
```

Optional local model support:

```bash
python3 -m pip install -e '.[models]'
```

Optional SQLite fallback for Python builds without stdlib `sqlite3`:

```bash
python3 -m pip install -e '.[sqlite]'
```

## Run

```bash
DOCGRAPH_ROOT=/path/to/project \
DOCGRAPH_CONFIG=/path/to/project/docgraph.config.yaml \
python3 -m docgraph_mcp.read_server

DOCGRAPH_ROOT=/path/to/project \
DOCGRAPH_CONFIG=/path/to/project/docgraph.config.yaml \
python3 -m docgraph_mcp.write_server
```

Self-test:

```bash
DOCGRAPH_ROOT=/tmp/docgraph-demo python3 -m docgraph_mcp.read_server --self-test
DOCGRAPH_ROOT=/tmp/docgraph-demo python3 -m docgraph_mcp.write_server --self-test
```

## Configuration

Start from:

```text
examples/docgraph.config.yaml
```

Important environment variables:

```text
DOCGRAPH_ROOT       project/source root; defaults to current directory
DOCGRAPH_DB         SQLite DB path; defaults to $DOCGRAPH_ROOT/docs/docgraph.sqlite
DOCGRAPH_CONFIG     YAML config path; optional
DOCGRAPH_LOG_LEVEL  temporary log level override
DOCGRAPH_LOG_FILE   temporary log destination override
```

## Retrieval Modes

```text
local   direct neighborhood of selected anchors
global  high-level framing nodes
bridge  bounded graph paths between anchors
hybrid  local + global + bridge
mix     hybrid + optional semantic candidates + optional rerankers
```

Semantic retrieval and rerankers are disabled by default. Configure them under `retrieval_models` in `examples/docgraph.config.yaml`.

## Safety Properties

- Active `Fact`, `Inference`, `Hypothesis`, and `Contradiction` claims require active supporting evidence chunks.
- `OpenQuestion` may exist without proof.
- Generated rendered DocGraph output is rejected as source evidence.
- `render_docs()` refuses output paths outside `DOCGRAPH_ROOT`.
- Stale source re-ingest marks old chunks/evidence stale and can mark dependent claims `needs_review`.
- Relink suggestion tools are read-only and only produce auto-safe drafts for exact/equivalent chunk matches.

## Tests

```bash
for t in tests/*.py; do python3 "$t"; done
```

## Repository Layout

```text
src/docgraph_mcp/       MCP server and backend package
docs/specs/             data model, retrieval, stale handling, tool contracts
tools/                  optional CLI/GUI helpers
examples/               starter configuration
tests/                  standalone backend/retrieval tests
```

## Status

This repository is an extracted public core. Agent prompts, project-specific runbooks, private graph databases, and organization-specific configuration should live in a separate private repository that consumes this package or mounts it as a submodule.
