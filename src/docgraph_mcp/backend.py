from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
try:
    import sqlite3
except ImportError:  # pragma: no cover - site-specific Python builds may omit stdlib sqlite3
    import pysqlite3 as sqlite3  # type: ignore
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from .config import cfg_get, cfg_list, cfg_number, load_config
from .logging_utils import TRACE_LEVEL, configure_logging, log_event, sanitize_for_log
from .models import ProviderState, cosine_similarity, make_embedding_provider, make_llm_reranker_provider, make_reranker_provider
from .mutation_flow import CANONICAL_MUTATION_OPS, MUTATION_OP_ALIASES, MutationFlow
from .retrieval_flow import RetrievalFlow
from .retrieval_types import ContextPacket, SemanticCandidates
from .visibility_policy import VisibilityPolicy


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_name(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def extract_terms(text: str) -> list[str]:
    terms: set[str] = set()
    # code-like identifiers, paths, quoted terms, acronyms, and short natural phrases
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
        s = m.group(0)
        if len(s) >= 2:
            terms.add(s)
            terms.add(normalize_name(s))
    for m in re.finditer(r"[A-Za-z0-9_./-]+\.(?:c|h|cpp|hpp|cc|sv|v|vh|tcl|mk|cmake)", text):
        terms.add(m.group(0))
    for m in re.finditer(r"['\"]([^'\"]{2,80})['\"]", text):
        terms.add(m.group(1))
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 2]
    for i in range(len(words) - 1):
        terms.add(words[i] + " " + words[i + 1])
    return sorted(terms, key=lambda x: (-len(x), x))[:40]


def fts_query(text: str) -> str:
    toks = []
    for t in re.findall(r"[A-Za-z0-9_]+", text):
        if len(t) >= 2:
            toks.append(t)
            n = normalize_name(t)
            if n != t.lower():
                toks.append(n)
    toks = list(dict.fromkeys(toks))[:12]
    if not toks:
        return ""
    # Quote every token to avoid FTS parser surprises.
    return " OR ".join(f'"{t}"' for t in toks)




def json_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value if v not in (None, "")]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v not in (None, "")]
        except Exception:
            pass
        return [x.strip() for x in re.split(r"[,;]", text) if x.strip()]
    return [str(value)]


def json_list_text(value: Any) -> str:
    return json.dumps(json_list(value), sort_keys=True)


class DocGraphBackend:
    """SQLite/FTS backend for DocGraph.

    Agents should not use SQL directly. The backend exposes MCP-safe operations:
    resolve, search, context, ingest, propose, commit, validate, render, stale_scan.
    """

    def __init__(self, db_path: str | Path, root: str | Path | None = None, config_path: str | Path | None = None):
        self.db_path = Path(db_path)
        self.root = Path(root or os.environ.get("DOCGRAPH_ROOT", ".")).resolve()
        if config_path is None:
            env_config = os.environ.get("DOCGRAPH_CONFIG")
            config_path = Path(env_config) if env_config else self.root / "docgraph.config.yaml"
        else:
            config_path = Path(config_path)
        if config_path is not None and not Path(config_path).is_absolute():
            config_path = self.root / Path(config_path)
        self.config_path = Path(config_path) if config_path is not None else None
        self.config = load_config(self.config_path)
        self.logger, self.logging_settings = configure_logging(self.config, self.root, component="backend")
        self.trace_level = TRACE_LEVEL
        self._embedding_provider_cache: tuple[Any | None, ProviderState] | None = None
        self._reranker_provider_cache: tuple[Any | None, ProviderState] | None = None
        self._llm_reranker_provider_cache: tuple[Any | None, ProviderState] | None = None
        self._mutation_flow = MutationFlow(
            self,
            id_factory=new_id,
            ts_factory=now_ts,
            normalize_name=normalize_name,
            list_parser=json_list,
            list_to_text=json_list_text,
        )
        self._visibility_policy = VisibilityPolicy(
            self,
            list_parser=json_list,
        )
        self._retrieval_flow = RetrievalFlow(
            self,
            fts_query=fts_query,
            extract_terms=extract_terms,
            id_factory=new_id,
            ts_factory=now_ts,
        )
        self._log("info", "backend.init.start", db_path=str(self.db_path), root=str(self.root), config_path=str(self.config_path) if self.config_path else None)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError as exc:
            # WAL can fail on some network/readonly filesystems; log but keep the backend usable.
            self._log("warning", "sqlite.pragma.wal_failed", error_type=type(exc).__name__, error=str(exc))
        self.init_schema()
        self._preload_model_providers()
        self._log("info", "backend.init.done", db_path=str(self.db_path))

    @classmethod
    def from_env(cls) -> "DocGraphBackend":
        root = Path(os.environ.get("DOCGRAPH_ROOT", ".")).resolve()
        db = Path(os.environ.get("DOCGRAPH_DB", root / "docs" / "docgraph.sqlite"))
        if not db.is_absolute():
            db = root / db
        config = os.environ.get("DOCGRAPH_CONFIG")
        return cls(db_path=db, root=root, config_path=config)

    def close(self) -> None:
        self._log("debug", "backend.close", db_path=str(self.db_path))
        self.conn.close()

    def _log(self, level: str | int, event: str, **fields: Any) -> None:
        safe_fields = {
            key: sanitize_for_log(
                value,
                include_payloads=bool(self.logging_settings.get("include_payloads", False)),
                preview_chars=int(self.logging_settings.get("payload_preview_chars", 500)),
            )
            for key, value in fields.items()
        }
        log_event(self.logger, level, event, **safe_fields)

    def log_tool_start(self, tool_name: str, params: dict[str, Any]) -> None:
        self._log("info", "mcp.tool.start", tool=tool_name, params=params)

    def log_tool_done(self, tool_name: str, result: Any, elapsed_ms: float) -> None:
        self._log("info", "mcp.tool.done", tool=tool_name, elapsed_ms=round(elapsed_ms, 3), result=result)

    def log_tool_error(self, tool_name: str, exc: BaseException, elapsed_ms: float) -> None:
        self._log("error", "mcp.tool.error", tool=tool_name, elapsed_ms=round(elapsed_ms, 3), error_type=type(exc).__name__, error=str(exc))

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources(
              source_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL,
              uri TEXT NOT NULL UNIQUE,
              name TEXT,
              current_hash TEXT,
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes(
              episode_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
              episode_type TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              raw_text TEXT,
              status TEXT NOT NULL DEFAULT 'active',
              supersedes_episode_id TEXT REFERENCES episodes(episode_id),
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id TEXT PRIMARY KEY,
              episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
              source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
              locator TEXT,
              text TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nodes(
              node_id TEXT PRIMARY KEY,
              node_type TEXT NOT NULL,
              canonical_name TEXT NOT NULL,
              summary TEXT,
              visibility TEXT NOT NULL DEFAULT 'local',
              finder_role TEXT,
              audience_roles_json TEXT NOT NULL DEFAULT '[]',
              interface_tags_json TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS aliases(
              alias_id TEXT PRIMARY KEY,
              node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
              alias TEXT NOT NULL,
              normalized_alias TEXT NOT NULL,
              alias_kind TEXT DEFAULT 'name',
              confidence TEXT DEFAULT 'medium',
              created_at TEXT NOT NULL,
              UNIQUE(node_id, normalized_alias)
            );

            CREATE TABLE IF NOT EXISTS edges(
              edge_id TEXT PRIMARY KEY,
              from_node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
              relation TEXT NOT NULL,
              to_node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
              summary TEXT,
              visibility TEXT NOT NULL DEFAULT 'local',
              finder_role TEXT,
              audience_roles_json TEXT NOT NULL DEFAULT '[]',
              interface_tags_json TEXT NOT NULL DEFAULT '[]',
              confidence TEXT DEFAULT 'medium',
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL,
              UNIQUE(from_node_id, relation, to_node_id)
            );

            CREATE TABLE IF NOT EXISTS claims(
              claim_id TEXT PRIMARY KEY,
              target_node_id TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
              target_edge_id TEXT REFERENCES edges(edge_id) ON DELETE SET NULL,
              claim_text TEXT NOT NULL,
              classification TEXT NOT NULL DEFAULT 'Fact',
              confidence TEXT DEFAULT 'medium',
              visibility TEXT NOT NULL DEFAULT 'local',
              finder_role TEXT,
              audience_roles_json TEXT NOT NULL DEFAULT '[]',
              interface_tags_json TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL DEFAULT 'active',
              superseded_by_claim_id TEXT REFERENCES claims(claim_id),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_evidence(
              claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
              chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
              evidence_role TEXT NOT NULL DEFAULT 'supports',
              strength TEXT DEFAULT 'medium',
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL,
              PRIMARY KEY(claim_id, chunk_id, evidence_role)
            );

            CREATE TABLE IF NOT EXISTS proposals(
              proposal_id TEXT PRIMARY KEY,
              reason TEXT NOT NULL,
              mutations_json TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_by TEXT,
              created_at TEXT NOT NULL,
              committed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS commits(
              commit_id TEXT PRIMARY KEY,
              proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
              before_revision TEXT NOT NULL,
              after_revision TEXT NOT NULL,
              mutation_digest TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
              chunk_id UNINDEXED,
              source_id UNINDEXED,
              locator,
              text
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
              claim_id UNINDEXED,
              claim_text
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
              node_id UNINDEXED,
              canonical_name,
              summary
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS aliases_fts USING fts5(
              alias_id UNINDEXED,
              node_id UNINDEXED,
              alias,
              normalized_alias
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
              edge_id UNINDEXED,
              from_node_id UNINDEXED,
              to_node_id UNINDEXED,
              relation,
              summary
            );

            CREATE TABLE IF NOT EXISTS node_terms(
              node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
              term TEXT NOT NULL,
              weight REAL NOT NULL DEFAULT 1.0,
              source TEXT NOT NULL DEFAULT 'auto',
              created_at TEXT NOT NULL,
              PRIMARY KEY(node_id, term, source)
            );

            CREATE TABLE IF NOT EXISTS edge_terms(
              edge_id TEXT NOT NULL REFERENCES edges(edge_id) ON DELETE CASCADE,
              term TEXT NOT NULL,
              weight REAL NOT NULL DEFAULT 1.0,
              source TEXT NOT NULL DEFAULT 'auto',
              created_at TEXT NOT NULL,
              PRIMARY KEY(edge_id, term, source)
            );

            CREATE TABLE IF NOT EXISTS embedding_items(
              item_type TEXT NOT NULL,
              target_id TEXT NOT NULL,
              text_hash TEXT NOT NULL,
              embedding_model TEXT NOT NULL,
              index_ref TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(item_type, target_id, embedding_model)
            );

            CREATE TABLE IF NOT EXISTS retrieval_runs(
              run_id TEXT PRIMARY KEY,
              query TEXT,
              anchors_json TEXT NOT NULL,
              mode TEXT NOT NULL,
              role TEXT,
              budget TEXT NOT NULL,
              result_summary_json TEXT NOT NULL,
              trace_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_migrations(
              version TEXT PRIMARY KEY,
              applied_at TEXT NOT NULL,
              description TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source_status ON chunks(source_id, status);
            CREATE INDEX IF NOT EXISTS idx_episodes_source_status ON episodes(source_id, status);
            CREATE INDEX IF NOT EXISTS idx_claims_target_node_status ON claims(target_node_id, status);
            CREATE INDEX IF NOT EXISTS idx_claims_target_edge_status ON claims(target_edge_id, status);
            CREATE INDEX IF NOT EXISTS idx_claims_visibility_status ON claims(visibility, status);
            CREATE INDEX IF NOT EXISTS idx_edges_from_status ON edges(from_node_id, status);
            CREATE INDEX IF NOT EXISTS idx_edges_to_status ON edges(to_node_id, status);
            CREATE INDEX IF NOT EXISTS idx_edges_visibility_status ON edges(visibility, status);
            CREATE INDEX IF NOT EXISTS idx_aliases_normalized ON aliases(normalized_alias);
            CREATE INDEX IF NOT EXISTS idx_evidence_chunk_status ON claim_evidence(chunk_id, status);
            CREATE INDEX IF NOT EXISTS idx_evidence_claim_status ON claim_evidence(claim_id, status);
            """
        )
        self._migrate_schema()
        self.conn.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES('graph_revision', '0')")
        self.conn.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', '6')")
        self.conn.execute("UPDATE metadata SET value='6' WHERE key='schema_version' AND CAST(value AS INTEGER) < 6")
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at, description) VALUES(?,?,?)",
            ("5", now_ts(), "hardening: indexes, schema_migrations, safer rendering, node/edge FTS search"),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at, description) VALUES(?,?,?)",
            ("6", now_ts(), "retrieval diagnostics: compact trace storage for read-only inspection"),
        )
        self.conn.commit()
        self._log("debug", "schema.init.done", schema_version="6")

    def _migrate_schema(self) -> None:
        self._log("debug", "schema.migrate.start")
        for table in ("nodes", "edges", "claims"):
            self._ensure_column(table, "visibility", "TEXT NOT NULL DEFAULT 'local'")
            self._ensure_column(table, "finder_role", "TEXT")
            self._ensure_column(table, "audience_roles_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(table, "interface_tags_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("edges", "summary", "TEXT")
        self._ensure_column("retrieval_runs", "trace_json", "TEXT NOT NULL DEFAULT '{}'")
        self._cleanup_orphan_fts_rows()
        self._log("debug", "schema.migrate.done")

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        existing = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._log("info", "schema.column_added", table=table, column=column, ddl=ddl)

    def _cleanup_orphan_fts_rows(self) -> None:
        for fts_table, id_col, base_table in (
            ("aliases_fts", "alias_id", "aliases"),
            ("edges_fts", "edge_id", "edges"),
        ):
            row = self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {fts_table} WHERE {id_col} NOT IN (SELECT {id_col} FROM {base_table})"
            ).fetchone()
            orphan_count = int(row["n"] if row else 0)
            if orphan_count <= 0:
                continue
            self.conn.execute(
                f"DELETE FROM {fts_table} WHERE {id_col} NOT IN (SELECT {id_col} FROM {base_table})"
            )
            self._log("warning", "schema.fts_orphans_cleaned", fts_table=fts_table, id_column=id_col, removed_rows=orphan_count)

    # ---------- ingestion ----------

    def _safe_resolve_uri(self, uri: str) -> Path:
        path = Path(uri)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Refusing to read outside DOCGRAPH_ROOT: {uri}") from exc
        return resolved

    def _safe_read_uri(self, uri: str) -> str:
        resolved = self._safe_resolve_uri(uri)
        return resolved.read_text(encoding="utf-8", errors="replace")

    def _existing_repo_file_for_uri(self, uri: str) -> Path | None:
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", uri):
            return None
        path = Path(uri)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved if resolved.is_file() else None

    def _text_size_bytes(self, text: str) -> int:
        return len(text.encode("utf-8", errors="replace"))

    def _source_limit_bytes(self, config_key: str) -> int:
        limit = int(cfg_number(self.config, f"source_handling.{config_key}", 0))
        return max(0, limit)

    def _enforce_source_size_limit(self, *, uri: str, source_kind: str, size_bytes: int, config_key: str) -> None:
        limit = self._source_limit_bytes(config_key)
        if limit <= 0 or size_bytes <= limit:
            return
        self._log(
            "warning",
            "ingest.rejected",
            reason="source_too_large",
            uri=uri,
            source_kind=source_kind,
            content_bytes=size_bytes,
            limit_bytes=limit,
        )
        raise ValueError(
            f"refusing to ingest {source_kind} source {uri!r}: "
            f"{size_bytes} bytes exceeds source_handling.{config_key}={limit}; "
            "ingest only focused evidence or raise the configured limit deliberately"
        )

    def _source_id_for_uri(self, uri: str) -> str:
        return "src_" + hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]

    def _chunk_content(self, content: str, source_type: str) -> list[tuple[str, str]]:
        lines = content.splitlines()
        line_source_types = set(cfg_list(self.config, "source_handling.line_chunk_source_types"))
        if source_type in line_source_types or len(lines) > 40:
            chunks: list[tuple[str, str]] = []
            step = max(1, int(cfg_get(self.config, "source_handling.lines_per_chunk", 60)))
            overlap = max(0, int(cfg_get(self.config, "source_handling.line_chunk_overlap", 5)))
            i = 0
            while i < len(lines):
                j = min(len(lines), i + step)
                text = "\n".join(lines[i:j]).strip()
                if text:
                    chunks.append((f"lines {i + 1}-{j}", text))
                if j == len(lines):
                    break
                i = max(j - overlap, i + 1)
            return chunks
        # paragraph chunking for docs/reports
        paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        out: list[tuple[str, str]] = []
        buf: list[str] = []
        idx = 1
        for p in paras:
            if sum(len(x) for x in buf) + len(p) > int(cfg_get(self.config, "source_handling.paragraph_max_chars", 1600)) and buf:
                out.append((f"paragraph group {idx}", "\n\n".join(buf)))
                idx += 1
                buf = []
            buf.append(p)
        if buf:
            out.append((f"paragraph group {idx}", "\n\n".join(buf)))
        if not out and content.strip():
            out.append(("content", content.strip()))
        return out

    def _reject_generated_doc_source(self, uri: str, content: str | None = None) -> None:
        """Prevent DocGraph from ingesting its own rendered output.

        Rendered docs are views of the graph, not independent evidence. If they
        are re-ingested, the graph can cite itself and reinforce stale or wrong
        claims. Keep generated docs as human-readable artifacts only.
        """
        norm_uri = uri.replace("\\", "/")
        basename = Path(norm_uri).name.lower()
        rendered_markers = (
            "/docs/rendered/",
            "docs/rendered/",
            "/rendered/",
        )
        content_prefix = (content or "")[:2000].lower()
        generated_content = "generated by render_docs" in content_prefix or "generated by docgraph" in content_prefix
        rendered_path = any(marker in norm_uri for marker in rendered_markers)
        generated_architecture_md = basename == "architecture.md" and generated_content
        if rendered_path or generated_architecture_md:
            raise ValueError(
                "refusing to ingest generated DocGraph output as evidence; "
                "exclude docs/rendered/* and generated architecture.md, and ingest original source docs/logs/code instead"
            )

    def ingest_source(
        self,
        source_type: str,
        uri: str,
        content: str | None = None,
        episode_type: str = "snapshot",
        name: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        has_inline_content = content is not None
        inline_content_bytes = self._text_size_bytes(content) if content is not None else None
        self._log(
            "info",
            "ingest.start",
            source_type=source_type,
            uri=uri,
            episode_type=episode_type,
            has_inline_content=has_inline_content,
            inline_content_bytes=inline_content_bytes,
            name=name,
        )
        if content is None:
            resolved = self._safe_resolve_uri(uri)
            file_size_bytes = resolved.stat().st_size
            self._enforce_source_size_limit(
                uri=uri,
                source_kind="file-backed",
                size_bytes=file_size_bytes,
                config_key="max_file_ingest_bytes",
            )
            content = resolved.read_text(encoding="utf-8", errors="replace")
        else:
            if bool(cfg_get(self.config, "source_handling.reject_inline_content_for_repo_files", True)):
                repo_file = self._existing_repo_file_for_uri(uri)
                if repo_file is not None:
                    self._log(
                        "warning",
                        "ingest.rejected",
                        reason="inline_content_for_repo_file",
                        uri=uri,
                        repo_path=str(repo_file),
                        content_bytes=inline_content_bytes,
                    )
                    raise ValueError(
                        f"refusing inline content for repository-local source {uri!r}; "
                        "omit content and let dg_ingest_source read the repo-relative uri"
                    )
            self._enforce_source_size_limit(
                uri=uri,
                source_kind="inline",
                size_bytes=inline_content_bytes or 0,
                config_key="max_inline_content_bytes",
            )
        content_bytes = self._text_size_bytes(content)
        self._reject_generated_doc_source(uri, content)
        source_id = self._source_id_for_uri(uri)
        h = sha256_text(content)
        ts = now_ts()
        cur = self.conn.cursor()
        existing = cur.execute("SELECT * FROM sources WHERE uri=?", (uri,)).fetchone()
        if existing and existing["current_hash"] == h:
            chunk_refs = self._chunk_refs_for_source(existing["source_id"])
            result = {
                "status": "unchanged",
                "source_id": existing["source_id"],
                "current_hash": h,
                "chunks": len(chunk_refs),
                "content_bytes": content_bytes,
                "chunk_ids": [c["chunk_id"] for c in chunk_refs],
                "chunk_refs": chunk_refs,
                "evidence_note": "Use these exact chunk_ids for claim evidence; do not infer chunk IDs from source_id or episode_id.",
            }
            self._log(
                "info",
                "ingest.unchanged",
                source_id=existing["source_id"],
                uri=uri,
                content_bytes=content_bytes,
                chunks=len(chunk_refs),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            return result

        old_chunk_ids = []
        supersedes_episode_id = None
        if existing:
            source_id = existing["source_id"]
            old_eps = cur.execute(
                "SELECT episode_id FROM episodes WHERE source_id=? AND status='active' ORDER BY created_at DESC",
                (source_id,),
            ).fetchall()
            if old_eps:
                supersedes_episode_id = old_eps[0]["episode_id"]
            old_chunk_ids = [
                r["chunk_id"]
                for r in cur.execute(
                    "SELECT chunk_id FROM chunks WHERE source_id=? AND status='active'", (source_id,)
                ).fetchall()
            ]
        with self.conn:
            if existing:
                self.conn.execute(
                    "UPDATE sources SET current_hash=?, status='active', updated_at=? WHERE source_id=?",
                    (h, ts, source_id),
                )
                self.conn.execute(
                    "UPDATE episodes SET status='superseded' WHERE source_id=? AND status='active'",
                    (source_id,),
                )
                self.conn.execute(
                    "UPDATE chunks SET status='stale' WHERE source_id=? AND status='active'",
                    (source_id,),
                )
                if old_chunk_ids:
                    marks = ",".join("?" for _ in old_chunk_ids)
                    self.conn.execute(
                        f"UPDATE claim_evidence SET status='stale' WHERE chunk_id IN ({marks})",
                        old_chunk_ids,
                    )
            else:
                self.conn.execute(
                    "INSERT INTO sources(source_id, source_type, uri, name, current_hash, status, created_at, updated_at) VALUES(?,?,?,?,?,'active',?,?)",
                    (source_id, source_type, uri, name or uri, h, ts, ts),
                )
            episode_id = new_id("ep")
            self.conn.execute(
                "INSERT INTO episodes(episode_id, source_id, episode_type, content_hash, raw_text, status, supersedes_episode_id, created_at) VALUES(?,?,?,?,?,'active',?,?)",
                (episode_id, source_id, episode_type, h, content, supersedes_episode_id, ts),
            )
            chunk_count = 0
            for locator, text in self._chunk_content(content, source_type):
                chunk_id = new_id("chunk")
                self.conn.execute(
                    "INSERT INTO chunks(chunk_id, episode_id, source_id, locator, text, content_hash, status, created_at) VALUES(?,?,?,?,?,?,'active',?)",
                    (chunk_id, episode_id, source_id, locator, text, sha256_text(text), ts),
                )
                self._upsert_chunk_fts(chunk_id, source_id, locator, text)
                chunk_count += 1
            review_update = self._mark_claims_without_active_support(old_chunk_ids)
        chunk_refs = self._chunk_refs_for_source(source_id)
        result = {
            "status": "ingested",
            "source_id": source_id,
            "episode_id": episode_id,
            "current_hash": h,
            "chunks": chunk_count,
            "content_bytes": content_bytes,
            "chunk_ids": [c["chunk_id"] for c in chunk_refs],
            "chunk_refs": chunk_refs,
            "evidence_note": "Use these exact chunk_ids for claim evidence; do not infer chunk IDs from source_id or episode_id.",
            "superseded_episode_id": supersedes_episode_id,
            "stale_chunk_ids": old_chunk_ids,
            "affected_claim_ids": review_update["affected_claim_ids"],
            "claims_marked_needs_review": review_update["claims_marked_needs_review"],
        }
        self._log(
            "info",
            "ingest.done",
            source_id=source_id,
            episode_id=episode_id,
            uri=uri,
            content_bytes=content_bytes,
            chunks=chunk_count,
            stale_chunks=len(old_chunk_ids),
            affected_claims=len(review_update["affected_claim_ids"]),
            claims_marked_needs_review=len(review_update["claims_marked_needs_review"]),
            superseded_episode_id=supersedes_episode_id,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return result

    def _chunk_refs_for_source(self, source_id: str, preview_chars: int = 180) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT chunk_id, episode_id, locator, text, status
            FROM chunks
            WHERE source_id=? AND status='active'
            ORDER BY rowid
            """,
            (source_id,),
        ).fetchall()
        refs: list[dict[str, Any]] = []
        for row in rows:
            preview = " ".join(str(row["text"]).split())
            if len(preview) > preview_chars:
                preview = preview[: preview_chars - 3] + "..."
            refs.append(
                {
                    "chunk_id": row["chunk_id"],
                    "episode_id": row["episode_id"],
                    "locator": row["locator"],
                    "status": row["status"],
                    "preview": preview,
                }
            )
        return refs

    def _mark_claims_without_active_support(self, affected_chunk_ids: list[str]) -> dict[str, list[str]]:
        if not affected_chunk_ids:
            return {"affected_claim_ids": [], "claims_marked_needs_review": []}
        marks = ",".join("?" for _ in affected_chunk_ids)
        affected_claims = [
            r["claim_id"]
            for r in self.conn.execute(
                f"SELECT DISTINCT claim_id FROM claim_evidence WHERE chunk_id IN ({marks})",
                affected_chunk_ids,
            ).fetchall()
        ]
        marked_needs_review: list[str] = []
        for claim_id in affected_claims:
            active_support = self.conn.execute(
                """
                SELECT 1
                FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id = ce.chunk_id
                WHERE ce.claim_id=?
                  AND ce.evidence_role='supports'
                  AND ce.status='active'
                  AND ch.status='active'
                LIMIT 1
                """,
                (claim_id,),
            ).fetchone()
            if not active_support:
                cur = self.conn.execute(
                    "UPDATE claims SET status='needs_review', updated_at=? WHERE claim_id=? AND status='active' AND classification <> 'OpenQuestion'",
                    (now_ts(), claim_id),
                )
                if cur.rowcount:
                    marked_needs_review.append(claim_id)
                self._log("warning", "claim.needs_review.no_active_support", claim_id=claim_id)
        return {
            "affected_claim_ids": affected_claims,
            "claims_marked_needs_review": marked_needs_review,
        }

    def suggest_evidence_relinks(self, claim_id: str, limit: int = 10) -> dict[str, Any]:
        """Suggest active replacement chunks for stale claim evidence.

        This is intentionally read-only. It helps a curator repair evidence links
        after a source was re-ingested, but it does not attach evidence or mark a
        claim active. Exact/near code-equivalence is treated as stronger than
        semantic similarity because evidence relinking changes proof, not search.
        """
        started = time.perf_counter()
        limit = max(1, min(int(limit), 50))
        claim = self.conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        if not claim:
            raise ValueError(f"unknown claim_id: {claim_id}")
        claim_d = dict(claim)
        stale_supports = self._stale_support_chunks_for_claim(claim_id)
        active_supports = self._active_support_chunks_for_claim(claim_id)
        candidates_by_chunk: dict[str, dict[str, Any]] = {}

        for stale in stale_supports:
            for candidate in self._active_chunks_for_source(stale["source_id"]):
                if candidate["chunk_id"] == stale["chunk_id"]:
                    continue
                scored = self._score_relink_candidate(claim_d, stale, candidate, same_source=True)
                self._merge_relink_candidate(candidates_by_chunk, scored, stale)

        if len(candidates_by_chunk) < limit:
            for stale in stale_supports:
                query = fts_query(f"{claim_d.get('claim_text', '')} {stale.get('text', '')}")
                if not query:
                    continue
                try:
                    rows = self.conn.execute(
                        """
                        SELECT ch.*, bm25(chunks_fts) AS rank, s.uri, s.source_type
                        FROM chunks_fts f
                        JOIN chunks ch ON ch.chunk_id=f.chunk_id
                        JOIN sources s ON s.source_id=ch.source_id
                        WHERE chunks_fts MATCH ?
                          AND ch.status='active'
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, limit * 4),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    self._log("debug", "suggest_evidence_relinks.fts_error", claim_id=claim_id, error_type=type(exc).__name__, error=str(exc))
                    rows = []
                for row in rows:
                    candidate = dict(row)
                    if candidate["chunk_id"] == stale["chunk_id"]:
                        continue
                    scored = self._score_relink_candidate(
                        claim_d,
                        stale,
                        candidate,
                        same_source=candidate.get("source_id") == stale.get("source_id"),
                        fts_rank=float(candidate.get("rank") or 0.0),
                    )
                    self._merge_relink_candidate(candidates_by_chunk, scored, stale)

        candidates = sorted(candidates_by_chunk.values(), key=lambda c: (-float(c["score"]), c["chunk_id"]))[:limit]
        equivalent = [c for c in candidates if c["support_level"] == "equivalent"]
        recommendation = "no_candidate_found"
        if equivalent:
            recommendation = "safe_equivalent_relink_candidate"
        elif candidates:
            recommendation = "review_candidates"

        draft_mutations = [
            {
                "op": "attach_evidence",
                "claim_id": claim_id,
                "chunk_id": c["chunk_id"],
                "evidence_role": "supports",
                "strength": "high" if c["support_level"] == "equivalent" else "medium",
            }
            for c in equivalent[:1]
        ]

        result = {
            "claim": {
                "claim_id": claim_d["claim_id"],
                "status": claim_d["status"],
                "classification": claim_d["classification"],
                "confidence": claim_d["confidence"],
                "claim_text": claim_d["claim_text"],
            },
            "active_support_count": len(active_supports),
            "stale_support_count": len(stale_supports),
            "stale_supports": [self._chunk_relink_summary(c) for c in stale_supports],
            "candidates": candidates,
            "recommendation": recommendation,
            "draft_mutations": draft_mutations,
            "rules": [
                "Read-only suggestion only; no database mutation was performed.",
                "Equivalent candidates still require curator review, proposal validation, commit, and render.",
                "If current source changed behavior, create contradiction/supersession instead of relinking.",
            ],
        }
        self._log(
            "info",
            "suggest_evidence_relinks.done",
            claim_id=claim_id,
            stale_supports=len(stale_supports),
            active_supports=len(active_supports),
            candidates=len(candidates),
            recommendation=recommendation,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(TRACE_LEVEL, "suggest_evidence_relinks.result", result=result)
        return result

    def suggest_source_relinks(
        self,
        source_id: str | None = None,
        uri: str | None = None,
        limit_per_claim: int = 5,
        max_claims: int = 500,
    ) -> dict[str, Any]:
        """Batch stale-evidence relink suggestions for one source.

        This is the source-level wrapper around ``suggest_evidence_relinks``.
        It keeps the repetitive per-claim loop inside the MCP backend so the
        agent/curator receives one grouped, read-only result.
        """
        started = time.perf_counter()
        limit_per_claim = max(1, min(int(limit_per_claim), 25))
        max_claims = max(1, min(int(max_claims), 2000))
        source = self._source_for_relink(source_id=source_id, uri=uri)
        source_id = source["source_id"]
        affected_claim_ids = self._affected_claim_ids_for_source(source_id, max_claims=max_claims)

        safe_relinks: list[dict[str, Any]] = []
        review_candidates: list[dict[str, Any]] = []
        unresolved_claims: list[dict[str, Any]] = []
        already_supported: list[dict[str, Any]] = []
        claim_results: list[dict[str, Any]] = []
        draft_mutations: list[dict[str, Any]] = []

        for claim_id in affected_claim_ids:
            stale_supports_for_source = [c for c in self._stale_support_chunks_for_claim(claim_id) if c["source_id"] == source_id]
            stale_chunk_ids_for_source = {c["chunk_id"] for c in stale_supports_for_source}
            suggestion = self.suggest_evidence_relinks(claim_id=claim_id, limit=limit_per_claim)
            source_candidates = [
                c
                for c in suggestion["candidates"]
                if c.get("source_id") == source_id and stale_chunk_ids_for_source.intersection(c.get("matched_stale_chunk_ids", []))
            ]
            safe_candidates = [c for c in source_candidates if c.get("support_level") == "equivalent"]
            review_only_candidates = [c for c in source_candidates if c.get("support_level") != "equivalent"]
            claim_summary = {
                "claim": suggestion["claim"],
                "active_support_count": suggestion["active_support_count"],
                "stale_support_count_for_source": len(stale_supports_for_source),
                "stale_supports_for_source": [self._chunk_relink_summary(c) for c in stale_supports_for_source],
                "candidate_count_for_source": len(source_candidates),
                "recommendation": "no_candidate_found",
            }

            if safe_candidates:
                best = safe_candidates[0]
                mutation = {
                    "op": "attach_evidence",
                    "claim_id": claim_id,
                    "chunk_id": best["chunk_id"],
                    "evidence_role": "supports",
                    "strength": "high",
                }
                item = claim_summary | {
                    "recommendation": "safe_equivalent_relink_candidate",
                    "candidate": best,
                    "draft_mutation": mutation,
                }
                safe_relinks.append(item)
                draft_mutations.append(mutation)
                claim_results.append(item)
            elif review_only_candidates:
                item = claim_summary | {
                    "recommendation": "review_candidates",
                    "candidates": review_only_candidates[:limit_per_claim],
                }
                review_candidates.append(item)
                claim_results.append(item)
            elif suggestion["active_support_count"] > 0:
                item = claim_summary | {
                    "recommendation": "already_has_active_support",
                }
                already_supported.append(item)
                claim_results.append(item)
            else:
                item = claim_summary | {
                    "recommendation": "no_candidate_found",
                }
                unresolved_claims.append(item)
                claim_results.append(item)

        result = {
            "source": {
                "source_id": source["source_id"],
                "uri": source["uri"],
                "source_type": source["source_type"],
                "name": source["name"],
                "status": source["status"],
                "current_hash": source["current_hash"],
            },
            "summary": {
                "affected_claim_count": len(affected_claim_ids),
                "safe_relink_claim_count": len(safe_relinks),
                "review_candidate_claim_count": len(review_candidates),
                "unresolved_claim_count": len(unresolved_claims),
                "already_supported_claim_count": len(already_supported),
                "draft_mutation_count": len(draft_mutations),
                "truncated": len(affected_claim_ids) >= max_claims,
            },
            "safe_relinks": safe_relinks,
            "review_candidates": review_candidates,
            "unresolved_claims": unresolved_claims,
            "already_supported": already_supported,
            "claim_results": claim_results,
            "draft_mutations": draft_mutations,
            "rules": [
                "Read-only batch suggestion only; no database mutation was performed.",
                "Safe relinks require exact chunk content-hash equality or same-source normalized-token hash equality.",
                "Review candidates from difflib/token/locator/FTS/LLM signals must be inspected before proposing evidence changes.",
                "Draft mutations attach replacement evidence only; curator must decide whether to mark claims active after validation.",
            ],
        }
        self._log(
            "info",
            "suggest_source_relinks.done",
            source_id=source_id,
            uri=source["uri"],
            affected_claims=len(affected_claim_ids),
            safe_relinks=len(safe_relinks),
            review_candidates=len(review_candidates),
            unresolved_claims=len(unresolved_claims),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        self._log(TRACE_LEVEL, "suggest_source_relinks.result", result=result)
        return result

    def _source_for_relink(self, source_id: str | None = None, uri: str | None = None) -> dict[str, Any]:
        if bool(source_id) == bool(uri):
            raise ValueError("provide exactly one of source_id or uri")
        if source_id:
            row = self.conn.execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
        else:
            row = self.conn.execute("SELECT * FROM sources WHERE uri=?", (uri,)).fetchone()
        if not row:
            ref = source_id if source_id else uri
            raise ValueError(f"unknown source: {ref}")
        return dict(row)

    def _affected_claim_ids_for_source(self, source_id: str, max_claims: int) -> list[str]:
        return [
            str(r["claim_id"])
            for r in self.conn.execute(
                """
                SELECT DISTINCT ce.claim_id
                FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ch.source_id=?
                  AND ce.evidence_role='supports'
                  AND (ce.status <> 'active' OR ch.status <> 'active' OR ep.status <> 'active' OR s.status <> 'active')
                ORDER BY ce.claim_id
                LIMIT ?
                """,
                (source_id, max_claims),
            )
        ]

    def _active_support_chunks_for_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return self._support_chunks_for_claim(claim_id, active=True)

    def _stale_support_chunks_for_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return self._support_chunks_for_claim(claim_id, active=False)

    def _support_chunks_for_claim(self, claim_id: str, active: bool) -> list[dict[str, Any]]:
        if active:
            status_filter = "ce.status='active' AND ch.status='active' AND ep.status='active' AND s.status='active'"
        else:
            status_filter = "(ce.status <> 'active' OR ch.status <> 'active' OR ep.status <> 'active' OR s.status <> 'active')"
        return [
            dict(r)
            for r in self.conn.execute(
                f"""
                SELECT ce.evidence_role, ce.strength, ce.status AS evidence_status,
                       ch.chunk_id, ch.episode_id, ch.source_id, ch.locator, ch.text,
                       ch.content_hash, ch.status AS chunk_status,
                       ep.status AS episode_status,
                       s.uri, s.source_type, s.status AS source_status
                FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ce.claim_id=?
                  AND ce.evidence_role='supports'
                  AND {status_filter}
                ORDER BY ce.created_at DESC
                """,
                (claim_id,),
            )
        ]

    def _active_chunks_for_source(self, source_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.conn.execute(
                """
                SELECT ch.*, s.uri, s.source_type
                FROM chunks ch
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ch.source_id=?
                  AND ch.status='active'
                  AND ep.status='active'
                  AND s.status='active'
                ORDER BY ch.rowid
                """,
                (source_id,),
            )
        ]

    def _score_relink_candidate(
        self,
        claim: dict[str, Any],
        stale: dict[str, Any],
        candidate: dict[str, Any],
        *,
        same_source: bool,
        fts_rank: float | None = None,
    ) -> dict[str, Any]:
        stale_norm = self._evidence_normalized_text(stale.get("text", ""))
        candidate_norm = self._evidence_normalized_text(candidate.get("text", ""))
        stale_norm_hash = sha256_text(stale_norm) if stale_norm else ""
        candidate_norm_hash = sha256_text(candidate_norm) if candidate_norm else ""
        stale_tokens = self._evidence_token_set(stale.get("text", ""))
        candidate_tokens = self._evidence_token_set(candidate.get("text", ""))
        claim_tokens = self._evidence_token_set(claim.get("claim_text", ""))

        reasons: list[str] = []
        score = 0.0
        support_level = "weak"
        if same_source:
            score += 25.0
            reasons.append("same source identity")

        if stale.get("content_hash") and stale.get("content_hash") == candidate.get("content_hash"):
            score += 180.0
            if same_source:
                support_level = "equivalent"
                reasons.append("exact chunk content hash matches in the same source")
            else:
                support_level = "strong_candidate"
                reasons.append("exact chunk content hash matches in a different source")
        elif stale_norm_hash and stale_norm_hash == candidate_norm_hash:
            score += 120.0
            if same_source:
                support_level = "equivalent"
                reasons.append("normalized token hash matches in the same source")
            else:
                support_level = "strong_candidate"
                reasons.append("normalized token hash matches in a different source")
        elif stale_norm and candidate_norm and (stale_norm in candidate_norm or candidate_norm in stale_norm):
            score += 85.0
            support_level = "strong_candidate"
            reasons.append("normalized text/code contains the stale evidence")

        sequence_ratio = 0.0
        if stale_norm and candidate_norm:
            sequence_ratio = difflib.SequenceMatcher(None, stale_norm, candidate_norm, autojunk=False).ratio()
            if sequence_ratio >= 0.90:
                score += sequence_ratio * 35.0
                reasons.append(f"normalized sequence similarity {sequence_ratio:.2f}")
                if support_level == "weak":
                    support_level = "review_candidate"

        token_overlap = self._jaccard(stale_tokens, candidate_tokens)
        if token_overlap:
            score += token_overlap * 60.0
            reasons.append(f"stale/candidate token overlap {token_overlap:.2f}")
        if support_level == "weak" and token_overlap >= 0.82:
            support_level = "strong_candidate"
        elif support_level == "weak" and token_overlap >= 0.55:
            support_level = "review_candidate"

        claim_overlap = self._jaccard(claim_tokens, candidate_tokens)
        if claim_overlap:
            score += claim_overlap * 25.0
            reasons.append(f"claim/candidate token overlap {claim_overlap:.2f}")

        line_distance = self._locator_line_distance(stale.get("locator"), candidate.get("locator"))
        if line_distance is not None:
            bonus = max(0.0, 20.0 - min(float(line_distance), 200.0) / 10.0)
            if bonus:
                score += bonus
                reasons.append(f"near previous locator ({line_distance} lines)")

        if fts_rank is not None:
            score += max(0.0, 8.0 - min(abs(float(fts_rank)), 8.0))
            reasons.append("FTS candidate")

        return {
            "chunk_id": candidate["chunk_id"],
            "source_id": candidate["source_id"],
            "uri": candidate.get("uri"),
            "source_type": candidate.get("source_type"),
            "locator": candidate.get("locator"),
            "score": round(score, 3),
            "support_level": support_level,
            "reasons": reasons,
            "text_preview": self._preview(candidate.get("text", "")),
            "matched_stale_chunk_ids": [stale["chunk_id"]],
        }

    def _merge_relink_candidate(self, candidates_by_chunk: dict[str, dict[str, Any]], candidate: dict[str, Any], stale: dict[str, Any]) -> None:
        existing = candidates_by_chunk.get(candidate["chunk_id"])
        if existing is None:
            candidates_by_chunk[candidate["chunk_id"]] = candidate
            return
        existing["matched_stale_chunk_ids"] = sorted(set(existing["matched_stale_chunk_ids"]) | {stale["chunk_id"]})
        existing["reasons"] = sorted(set(existing["reasons"]) | set(candidate["reasons"]))
        if float(candidate["score"]) > float(existing["score"]):
            existing["score"] = candidate["score"]
            existing["support_level"] = candidate["support_level"]
            existing["locator"] = candidate["locator"]
            existing["text_preview"] = candidate["text_preview"]

    def _chunk_relink_summary(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "uri": chunk.get("uri"),
            "source_type": chunk.get("source_type"),
            "locator": chunk.get("locator"),
            "evidence_status": chunk.get("evidence_status"),
            "chunk_status": chunk.get("chunk_status"),
            "episode_status": chunk.get("episode_status"),
            "source_status": chunk.get("source_status"),
            "text_preview": self._preview(chunk.get("text", "")),
        }

    def _evidence_normalized_text(self, text: str) -> str:
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = re.sub(r"//.*?$", " ", text, flags=re.M)
        tokens = re.findall(
            r"0x[0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||<<|>>|->|::|[{}()\[\];,.:?~!%^&*+=|/<>-]",
            text,
        )
        return " ".join(tokens)

    def _evidence_token_set(self, text: str) -> set[str]:
        norm = self._evidence_normalized_text(text).lower()
        return {t for t in norm.split() if len(t) > 1}

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _locator_line_distance(self, left: str | None, right: str | None) -> int | None:
        left_line = self._locator_start_line(left)
        right_line = self._locator_start_line(right)
        if left_line is None or right_line is None:
            return None
        return abs(left_line - right_line)

    def _locator_start_line(self, locator: str | None) -> int | None:
        if not locator:
            return None
        match = re.search(r"(?:line|lines)\s+(\d+)", locator, flags=re.I)
        return int(match.group(1)) if match else None

    # ---------- FTS maintenance ----------

    def _upsert_chunk_fts(self, chunk_id: str, source_id: str, locator: str | None, text: str) -> None:
        self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        self.conn.execute(
            "INSERT INTO chunks_fts(chunk_id, source_id, locator, text) VALUES(?,?,?,?)",
            (chunk_id, source_id, locator or "", text),
        )

    def _upsert_claim_fts(self, claim_id: str, claim_text: str) -> None:
        self.conn.execute("DELETE FROM claims_fts WHERE claim_id=?", (claim_id,))
        self.conn.execute("INSERT INTO claims_fts(claim_id, claim_text) VALUES(?,?)", (claim_id, claim_text))

    def _upsert_node_fts(self, node_id: str, canonical_name: str, summary: str | None) -> None:
        self.conn.execute("DELETE FROM nodes_fts WHERE node_id=?", (node_id,))
        self.conn.execute(
            "INSERT INTO nodes_fts(node_id, canonical_name, summary) VALUES(?,?,?)",
            (node_id, canonical_name, summary or ""),
        )

    def _upsert_alias_fts(self, alias_id: str, node_id: str, alias: str, normalized_alias: str) -> None:
        self.conn.execute(
            "DELETE FROM aliases_fts WHERE node_id=? AND normalized_alias=?",
            (node_id, normalized_alias),
        )
        self.conn.execute("DELETE FROM aliases_fts WHERE alias_id=?", (alias_id,))
        self.conn.execute(
            "INSERT INTO aliases_fts(alias_id, node_id, alias, normalized_alias) VALUES(?,?,?,?)",
            (alias_id, node_id, alias, normalized_alias),
        )

    def _upsert_edge_fts(self, edge_id: str, from_node_id: str, relation: str, to_node_id: str, summary: str | None) -> None:
        self.conn.execute(
            "DELETE FROM edges_fts WHERE from_node_id=? AND relation=? AND to_node_id=?",
            (from_node_id, relation, to_node_id),
        )
        self.conn.execute("DELETE FROM edges_fts WHERE edge_id=?", (edge_id,))
        self.conn.execute(
            "INSERT INTO edges_fts(edge_id, from_node_id, to_node_id, relation, summary) VALUES(?,?,?,?,?)",
            (edge_id, from_node_id, to_node_id, relation, summary or ""),
        )

    def _replace_node_terms(self, node_id: str, text: str, source: str = "auto") -> None:
        ts = now_ts()
        self.conn.execute("DELETE FROM node_terms WHERE node_id=? AND source=?", (node_id, source))
        for term in extract_terms(text):
            self.conn.execute(
                "INSERT OR REPLACE INTO node_terms(node_id, term, weight, source, created_at) VALUES(?,?,?,?,?)",
                (node_id, term, 1.0, source, ts),
            )

    def _replace_edge_terms(self, edge_id: str, text: str, source: str = "auto") -> None:
        ts = now_ts()
        self.conn.execute("DELETE FROM edge_terms WHERE edge_id=? AND source=?", (edge_id, source))
        for term in extract_terms(text):
            self.conn.execute(
                "INSERT OR REPLACE INTO edge_terms(edge_id, term, weight, source, created_at) VALUES(?,?,?,?,?)",
                (edge_id, term, 1.0, source, ts),
            )

    # ---------- read/search ----------

    def resolve(self, text: str, limit: int = 10, include_stale: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        self._log("debug", "resolve.start", text=text, limit=limit, include_stale=include_stale)
        norm = normalize_name(text)
        rows: list[dict[str, Any]] = []
        status_filter = "" if include_stale else "AND n.status='active'"
        for r in self.conn.execute(
            f"""
            SELECT n.*, a.alias, a.alias_kind, a.confidence AS alias_confidence
            FROM aliases a JOIN nodes n ON n.node_id=a.node_id
            WHERE a.normalized_alias=? {status_filter}
            LIMIT ?
            """,
            (norm, limit),
        ).fetchall():
            rows.append({"match_type": "exact_alias", "score": self._rank_weight("exact_alias", 100.0), **dict(r)})
        for r in self.conn.execute(f"SELECT * FROM nodes n WHERE lower(canonical_name)=lower(?) {status_filter} LIMIT ?", (text, limit)).fetchall():
            rows.append({"match_type": "exact_node", "score": self._rank_weight("exact_node", 95.0), **dict(r)})
        if len(rows) < limit:
            fq = fts_query(text)
            if fq:
                try:
                    for r in self.conn.execute(
                        """
                        SELECT a.alias_id, a.node_id, a.alias, a.normalized_alias, bm25(aliases_fts) AS rank,
                               n.node_type, n.canonical_name, n.summary, n.status
                        FROM aliases_fts af
                        JOIN aliases a ON a.alias_id=af.alias_id
                        JOIN nodes n ON n.node_id=a.node_id
                        WHERE aliases_fts MATCH ?
                        ORDER BY rank LIMIT ?
                        """,
                        (fq, limit),
                    ).fetchall():
                        if not include_stale and r["status"] != "active":
                            continue
                        rows.append({"match_type": "fts_alias", "score": self._rank_weight("fts_alias", 60.0) - float(r["rank"]), **dict(r)})
                except sqlite3.OperationalError as exc:
                    self._log("debug", "resolve.alias_fts.error", text=text, error_type=type(exc).__name__, error=str(exc))
            if len(rows) < limit:
                try:
                    for r in self.conn.execute(
                        """
                        SELECT n.*, bm25(nodes_fts) AS rank
                        FROM nodes_fts nf JOIN nodes n ON n.node_id=nf.node_id
                        WHERE nodes_fts MATCH ?
                        ORDER BY rank LIMIT ?
                        """,
                        (fq, limit),
                    ).fetchall():
                        if not include_stale and r["status"] != "active":
                            continue
                        rows.append({"match_type": "fts_node", "score": self._rank_weight("node_fts", 55.0) - float(r["rank"]), **dict(r)})
                except sqlite3.OperationalError as exc:
                    self._log("debug", "resolve.node_fts.error", text=text, error_type=type(exc).__name__, error=str(exc))
        # de-duplicate by node_id/match_type preference
        seen = set()
        out = []
        for r in sorted(rows, key=lambda x: -x.get("score", 0)):
            key = r.get("node_id")
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= limit:
                break
        result = {"query": text, "normalized": norm, "matches": out}
        self._log("info", "resolve.done", text=text, normalized=norm, match_count=len(out), elapsed_ms=(time.perf_counter() - started) * 1000)
        self._log(TRACE_LEVEL, "resolve.matches", text=text, matches=out)
        return result

    def search(
        self,
        query: str,
        role: str | None = None,
        intent: str | None = None,
        limit: int = 10,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        """Public retrieval search API delegated to ``RetrievalFlow``."""
        return self._retrieval_flow.search(query=query, role=role, intent=intent, limit=limit, include_stale=include_stale)

    def context(
        self,
        anchors: list[str] | None = None,
        query: str | None = None,
        role: str | None = None,
        intent: str | None = None,
        budget: str = "small",
        mode: str | None = None,
        include_stale_warnings: bool | None = None,
    ) -> ContextPacket:
        """Build a context packet via ``RetrievalFlow``."""
        return self._retrieval_flow.context(
            anchors=anchors,
            query=query,
            role=role,
            intent=intent,
            budget=budget,
            mode=mode,
            include_stale_warnings=include_stale_warnings,
        )

    def _normalize_context_mode(self, mode: str | None) -> str:
        """Compatibility wrapper for retrieval-mode validation."""
        return self._retrieval_flow.normalize_context_mode(mode)

    def _resolve_context_nodes(self, anchors: list[str] | None, query: str | None, role: str | None, intent: str | None, limit: int) -> list[str]:
        """Compatibility wrapper for anchor/query node resolution."""
        return self._retrieval_flow.resolve_context_nodes(anchors, query, role, intent, limit)

    def _local_context_sections(self, node_ids: list[str], role: str | None, limits: dict[str, int], include_stale_warnings: bool) -> dict[str, list[dict[str, Any]]]:
        """Compatibility wrapper for local context section construction."""
        return self._retrieval_flow.local_context_sections(node_ids, role, limits, include_stale_warnings)

    def _global_context(self, node_ids: list[str], query: str | None, role: str | None, limits: dict[str, int]) -> list[dict[str, Any]]:
        """Compatibility wrapper for global frame retrieval."""
        return self._retrieval_flow.global_context(node_ids, query, role, limits)

    def _bridge_context(self, node_ids: list[str], query: str | None, role: str | None, limits: dict[str, int]) -> dict[str, Any]:
        """Compatibility wrapper for bridge-path retrieval."""
        return self._retrieval_flow.bridge_context(node_ids, query, role, limits)

    def _find_bridge_paths(self, start: str, goal: str, role: str | None, max_depth: int, max_paths: int) -> list[dict[str, Any]]:
        """Compatibility wrapper for pairwise bridge-path search."""
        return self._retrieval_flow.find_bridge_paths(start, goal, role, max_depth, max_paths)

    def _format_bridge_path(self, start: str, goal: str, edge_path: list[dict[str, Any]], role: str | None) -> dict[str, Any]:
        """Compatibility wrapper for bridge-path scoring/formatting."""
        return self._retrieval_flow.format_bridge_path(start, goal, edge_path, role)

    def _semantic_context(self, query: str | None, role: str | None, *, lexical_anchor_count: int = 0) -> SemanticCandidates:
        """Compatibility wrapper for optional semantic retrieval."""
        return self._retrieval_flow.semantic_context(query, role, lexical_anchor_count=lexical_anchor_count)

    def _semantic_candidate_texts(self, role: str | None) -> list[dict[str, Any]]:
        """Compatibility wrapper for semantic candidate materialization."""
        return self._retrieval_flow.semantic_candidate_texts(role)

    def _maybe_rerank(self, query: str, candidates: list[dict[str, Any]], *, lexical_anchor_count: int = 0) -> list[dict[str, Any]]:
        """Compatibility wrapper for optional reranking."""
        ranked, _trace = self._retrieval_flow.maybe_rerank(query, candidates, lexical_anchor_count=lexical_anchor_count)
        return ranked

    def _record_retrieval_run(self, packet: ContextPacket, retrieval_trace: dict[str, Any] | None = None) -> None:
        """Compatibility wrapper for retrieval telemetry persistence."""
        self._retrieval_flow.record_retrieval_run(packet, retrieval_trace)

    @property
    def last_retrieval_run_id(self) -> str | None:
        """Return the retrieval telemetry row written by the last context call."""
        return self._retrieval_flow.last_recorded_run_id

    def _preload_model_providers(self) -> None:
        """Optionally warm model providers at backend startup."""
        if bool(cfg_get(self.config, "retrieval_models.embeddings.preload_on_boot", True)):
            should_preload_embedding = True
            embed_provider = str(cfg_get(self.config, "retrieval_models.embeddings.provider", "sentence_transformers"))
            embed_model = str(cfg_get(self.config, "retrieval_models.embeddings.model", "") or "").strip()
            if embed_provider == "sentence_transformers":
                model_path = Path(embed_model)
                if not model_path.is_absolute():
                    model_path = (self.root / model_path).resolve()
                if not model_path.exists():
                    self._log(
                        "info",
                        "models.embedding.preload.skipped",
                        reason="sentence_transformers model path is not local",
                        configured_model=embed_model,
                    )
                    should_preload_embedding = False
            if should_preload_embedding:
                provider, state = self._make_embedding_provider()
                self._log(
                    "info",
                    "models.embedding.preload",
                    enabled=state.enabled,
                    available=state.available,
                    reason=state.reason,
                    provider_type=(type(provider).__name__ if provider is not None else None),
                )
        if bool(cfg_get(self.config, "retrieval_models.reranker.preload_on_boot", False)):
            provider, state = self._make_reranker_provider()
            self._log(
                "info",
                "models.reranker.preload",
                enabled=state.enabled,
                available=state.available,
                reason=state.reason,
                provider_type=(type(provider).__name__ if provider is not None else None),
            )

    def _config_with_resolved_model_path(self, model_group: str) -> dict[str, Any]:
        """Resolve project-relative local model paths before provider construction."""
        resolved = copy.deepcopy(self.config)
        cfg = resolved.get("retrieval_models", {}).get(model_group, {})
        if not isinstance(cfg, dict):
            return resolved
        provider = str(cfg.get("provider", ""))
        if provider not in {"sentence_transformers", "sentence_transformers_cross_encoder"}:
            return resolved
        model = str(cfg.get("model", "") or "").strip()
        if not model:
            return resolved
        model_path = Path(model)
        if not model_path.is_absolute():
            local_path = (self.root / model_path).resolve()
            if local_path.exists():
                cfg["model"] = str(local_path)
        return resolved

    def _make_embedding_provider(self) -> tuple[Any | None, ProviderState]:
        """Return cached embedding provider; retry only if enabled but unavailable."""
        cached = self._embedding_provider_cache
        if cached is not None:
            _, state = cached
            if state.available or not state.enabled:
                return cached
        fresh = make_embedding_provider(self._config_with_resolved_model_path("embeddings"))
        self._embedding_provider_cache = fresh
        return fresh

    def _make_reranker_provider(self) -> tuple[Any | None, ProviderState]:
        """Return cached reranker provider; retry only if enabled but unavailable."""
        cached = self._reranker_provider_cache
        if cached is not None:
            _, state = cached
            if state.available or not state.enabled:
                return cached
        fresh = make_reranker_provider(self._config_with_resolved_model_path("reranker"))
        self._reranker_provider_cache = fresh
        return fresh

    def _make_llm_reranker_provider(self) -> tuple[Any | None, ProviderState]:
        """Return cached LLM reranker provider; retry only if enabled but unavailable."""
        cached = self._llm_reranker_provider_cache
        if cached is not None:
            _, state = cached
            if state.available or not state.enabled:
                return cached
        fresh = make_llm_reranker_provider(self.config)
        self._llm_reranker_provider_cache = fresh
        return fresh

    def related_context_check(
        self,
        finding: str,
        finder_role: str | None = None,
        interface_tags: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Bounded cross-role check before classifying a finding as local.

        This deliberately returns candidates and recommendations only. It does
        not create edges or claims from lexical similarity alone.
        """
        started = time.perf_counter()
        self._log("info", "related_context_check.start", finding=finding, finder_role=finder_role, interface_tags=interface_tags or [], limit=limit)
        tags = json_list(interface_tags or [])
        bad_tags = [t for t in tags if t not in self._allowed_interface_tags()]
        if bad_tags:
            raise ValueError(f"bad interface_tags: {bad_tags}; allowed tags: {self._allowed_interface_tags()}")
        if finder_role and finder_role not in self._allowed_roles():
            raise ValueError(f"bad finder_role: {finder_role}; allowed roles: {self._allowed_roles()}")

        high_level_types = set(cfg_list(self.config, "shared_knowledge.high_level_node_types"))
        trigger_tags = set(cfg_list(self.config, "shared_knowledge.cross_role_trigger_tags"))
        search_text = " ".join([finding] + tags)
        sr = self.search(search_text, role="architecture", intent="architecture", limit=max(limit * 3, limit))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for n in sr.get("anchors", []):
            if not n or n.get("node_id") in seen:
                continue
            if n.get("node_type") in high_level_types:
                seen.add(n["node_id"])
                candidates.append(n)
        for r in sr.get("results", []):
            node = r.get("node")
            if node and node.get("node_type") in high_level_types and node.get("node_id") not in seen:
                seen.add(node["node_id"])
                candidates.append(node)
        impact_likely = bool(trigger_tags.intersection(tags)) or bool(re.search(r"\b(register|config|vsync|frame|channel|interrupt|status|memory|header|test|debug|build)\b", finding, re.I))
        if candidates and impact_likely:
            suggested_visibility = "shared"
        elif impact_likely:
            suggested_visibility = "shared_candidate"
        else:
            suggested_visibility = "local"
        result = {
            "finding": finding,
            "finder_role": finder_role,
            "interface_tags": tags,
            "suggested_visibility": suggested_visibility,
            "high_level_candidates": candidates[:limit],
            "rules": [
                "Use candidates as possible related context, not proof.",
                "Create an edge only when evidence supports the relation, not from lexical similarity alone.",
                "If impact is likely but relation is unproven, use shared_candidate.",
            ],
        }
        self._log("info", "related_context_check.done", suggested_visibility=suggested_visibility, candidate_count=len(candidates[:limit]), impact_likely=impact_likely, elapsed_ms=(time.perf_counter() - started) * 1000)
        self._log(TRACE_LEVEL, "related_context_check.result", result=result)
        return result

    # ---------- mutation flow ----------

    def mutation_schema(self) -> dict[str, Any]:
        """Return the exact mutation contract accepted by propose/commit.

        Agents often guess operation names such as ``upsert_edge`` or ``node``.
        The backend normalizes common aliases, but this schema is the canonical
        contract the curator should follow.
        """
        return {
            "canonical_ops": sorted(CANONICAL_MUTATION_OPS),
            "op_aliases": dict(sorted(MUTATION_OP_ALIASES.items())),
            "taxonomy_contract": {
                "node_types": sorted(self._allowed_node_types()),
                "relation_types": sorted(self._allowed_relation_types()),
                "claim_classes": sorted(self._allowed_claim_classes()),
                "claim_statuses": sorted(self._allowed_claim_statuses()),
                "evidence_roles": sorted(self._allowed_evidence_roles()),
            },
            "visibility_contract": {
                "visibility_values": sorted(self._allowed_visibility_values()),
                "roles": sorted(self._allowed_roles()),
                "interface_tags": sorted(self._allowed_interface_tags()),
                "meaning": {
                    "local": "visible mainly to finder_role and explicit audience_roles",
                    "shared": "verified cross-role/interface knowledge",
                    "global": "broad system-level knowledge relevant to all roles",
                    "shared_candidate": "likely cross-role but relation/impact is not fully proven yet",
                },
            },
            "rules": [
                "Call dg_mutation_schema before proposing updates so node/relation/tag/taxonomy constraints are explicit.",
                "Every mutation should include op. If op is omitted, the backend tries to infer it from fields, but curator agents should not rely on inference.",
                "Use add_alias, not upsert_alias.",
                "Use add_edge, not upsert_edge.",
                "Use upsert_node for nodes and upsert_claim for claims.",
                "Evidence can be linked explicitly with attach_evidence, or compactly with upsert_claim chunk_ids.",
                "Do not guess chunk IDs. Use exact chunk_ids/chunk_refs returned by dg_ingest_source or dg_ingest_investigation_report.",
                "Do not ingest a source merely because it was inspected; ingest only evidence needed for a durable claim or explicit open-question context.",
                "For repository-local sources, call dg_ingest_source with a repo-relative uri and omit inline content; inline content is for external/archive/temporary evidence only.",
                "Active Fact/Inference/Hypothesis/Contradiction claims must have active supporting evidence before commit.",
                "Proposal references are preflighted: missing target_node_id, target_edge_id, claim_id, or chunk_id will fail before commit with a typed error.",
                "Do not ingest generated render_docs output such as docs/rendered/* or generated architecture.md as source evidence.",
                "Every durable node/edge/claim may include visibility, finder_role, audience_roles, and interface_tags. Use shared/shared_candidate/global for cross-role knowledge.",
                "Do not classify a finding local until a bounded related-context check was considered.",
            ],
            "ops": {
                "upsert_node": {
                    "required": ["op", "node_type", "canonical_name"],
                    "optional": ["node_id", "summary", "visibility", "finder_role", "audience_roles", "interface_tags", "status"],
                    "aliases": {"name": "canonical_name", "type": "node_type"},
                    "example": {"op": "upsert_node", "node_id": "flow.build_flow", "node_type": "flow", "canonical_name": "build_flow", "summary": "Build flow."},
                },
                "add_alias": {
                    "required": ["op", "node_id", "alias"],
                    "optional": ["alias_id", "alias_kind", "confidence"],
                    "aliases": {"name": "alias"},
                    "example": {"op": "add_alias", "node_id": "flow.build_flow", "alias": "build process"},
                },
                "add_edge": {
                    "required": ["op", "from_node_id", "relation", "to_node_id"],
                    "optional": ["edge_id", "summary", "visibility", "finder_role", "audience_roles", "interface_tags", "confidence", "status"],
                    "aliases": {"from": "from_node_id", "source_node_id": "from_node_id", "to": "to_node_id", "target_node_id": "to_node_id", "relation_type": "relation"},
                    "example": {"op": "add_edge", "from_node_id": "function.init", "relation": "implements", "to_node_id": "flow.build_flow"},
                },
                "upsert_claim": {
                    "required": ["op", "claim_text"],
                    "optional": ["claim_id", "target_node_id", "target_edge_id", "classification", "confidence", "visibility", "finder_role", "audience_roles", "interface_tags", "status", "superseded_by_claim_id", "chunk_ids", "evidence_role", "evidence_strength"],
                    "aliases": {"text": "claim_text", "statement": "claim_text", "evidence_chunk_ids": "chunk_ids", "supporting_chunk_ids": "chunk_ids", "chunks": "chunk_ids"},
                    "example": {"op": "upsert_claim", "claim_id": "claim.build_flow.doc", "target_node_id": "flow.build_flow", "claim_text": "The old docs describe the build flow.", "classification": "Hypothesis", "chunk_ids": ["chunk_id_from_ingest_chunk_refs"]},
                    "expansion": "chunk_ids is expanded internally into attach_evidence mutations; canonical explicit attach_evidence is still preferred for complex evidence roles.",
                },
                "attach_evidence": {
                    "required": ["op", "claim_id", "chunk_id"],
                    "optional": ["evidence_role", "strength", "status"],
                    "aliases": {"role": "evidence_role"},
                    "example": {"op": "attach_evidence", "claim_id": "claim.build_flow.doc", "chunk_id": "chunk_id_from_ingest_chunk_refs", "evidence_role": "supports"},
                },
                "mark_claim_status": {
                    "required": ["op", "claim_id", "status"],
                    "optional": ["superseded_by_claim_id"],
                    "example": {"op": "mark_claim_status", "claim_id": "claim.old", "status": "needs_review"},
                },
            },
        }

    def _normalize_mutations(self, mutations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compatibility wrapper for mutation normalization and expansion."""
        return self._mutation_flow.normalize_mutations(mutations)

    def _evidence_mutations_from_claim_shortcut(self, m: dict[str, Any]) -> list[dict[str, Any]]:
        """Compatibility wrapper for compact claim-evidence expansion."""
        return self._mutation_flow.evidence_mutations_from_claim_shortcut(m)

    def _infer_mutation_op(self, m: dict[str, Any]) -> str | None:
        """Compatibility wrapper for inferring canonical mutation ops."""
        return self._mutation_flow.infer_mutation_op(m)

    def _normalize_mutation(self, mutation: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wrapper for single mutation normalization."""
        return self._mutation_flow.normalize_mutation(mutation)

    def propose_update(self, reason: str, mutations: list[dict[str, Any]], created_by: str = "curator") -> dict[str, Any]:
        started = time.perf_counter()
        self._log("info", "proposal.start", reason=reason, mutation_count=len(mutations) if isinstance(mutations, list) else None, created_by=created_by)
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("mutations must be a non-empty list")
        mutations = self._normalize_mutations(mutations)
        self._validate_mutation_shapes(mutations)
        self._validate_mutation_references(mutations)
        proposal_id = new_id("prop")
        ts = now_ts()
        with self.conn:
            self.conn.execute(
                "INSERT INTO proposals(proposal_id, reason, mutations_json, status, created_by, created_at) VALUES(?,?,?,?,?,?)",
                (proposal_id, reason, json.dumps(mutations, sort_keys=True), "pending", created_by, ts),
            )
        result = {"proposal_id": proposal_id, "status": "pending", "mutations": len(mutations)}
        self._log("info", "proposal.done", proposal_id=proposal_id, mutation_count=len(mutations), elapsed_ms=(time.perf_counter() - started) * 1000)
        self._log(TRACE_LEVEL, "proposal.mutations", proposal_id=proposal_id, mutations=mutations)
        return result

    def commit_update(self, proposal_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        self._log("info", "commit.start", proposal_id=proposal_id)
        row = self.conn.execute("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown proposal_id: {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(f"proposal is not pending: {row['status']}")
        mutations = self._normalize_mutations(json.loads(row["mutations_json"]))
        self._validate_mutation_shapes(mutations)
        self._validate_mutation_references(mutations)
        before_rev = self._revision()
        digest = sha256_text(row["mutations_json"])
        commit_id = new_id("commit")
        ts = now_ts()
        try:
            with self.conn:
                for m in mutations:
                    self._apply_mutation(m)
                validation = self.validate()
                if validation["errors"]:
                    self._log("error", "commit.validation_failed", proposal_id=proposal_id, errors=validation["errors"])
                    raise ValueError("validation failed: " + json.dumps(validation["errors"], indent=2))
                after_rev = str(int(before_rev) + 1)
                self.conn.execute("UPDATE metadata SET value=? WHERE key='graph_revision'", (after_rev,))
                self.conn.execute(
                    "UPDATE proposals SET status='committed', committed_at=? WHERE proposal_id=?",
                    (ts, proposal_id),
                )
                self.conn.execute(
                    "INSERT INTO commits(commit_id, proposal_id, before_revision, after_revision, mutation_digest, applied_at) VALUES(?,?,?,?,?,?)",
                    (commit_id, proposal_id, before_rev, after_rev, digest, ts),
                )
        except Exception as exc:
            self._log("error", "commit.error", proposal_id=proposal_id, error_type=type(exc).__name__, error=str(exc), elapsed_ms=(time.perf_counter() - started) * 1000)
            raise
        result = {"commit_id": commit_id, "proposal_id": proposal_id, "before_revision": before_rev, "after_revision": str(int(before_rev) + 1)}
        self._log("info", "commit.done", **result, mutation_count=len(mutations), elapsed_ms=(time.perf_counter() - started) * 1000)
        return result

    def _validate_mutation_shapes(self, mutations: list[dict[str, Any]]) -> None:
        """Compatibility wrapper for mutation shape/taxonomy checks."""
        self._mutation_flow.validate_mutation_shapes(mutations)

    def _validate_visibility_metadata(self, m: dict[str, Any]) -> None:
        """Compatibility wrapper for visibility metadata validation."""
        self._mutation_flow.validate_visibility_metadata(m)

    def _visibility_values_for_insert(self, m: dict[str, Any]) -> tuple[str, str | None, str, str]:
        """Compatibility wrapper for SQL-ready visibility field normalization."""
        return self._mutation_flow.visibility_values_for_insert(m)

    def _apply_mutation(self, m: dict[str, Any]) -> None:
        """Compatibility wrapper for applying one canonical mutation."""
        self._mutation_flow.apply_mutation(m)

    def _validate_mutation_references(self, mutations: list[dict[str, Any]]) -> None:
        """Preflight FK-like references so agents get actionable errors."""
        proposed_nodes: set[str] = set()
        proposed_edges: set[str] = set()
        proposed_claims: set[str] = set()
        required_nodes: list[tuple[str, int, str]] = []
        required_edges: list[tuple[str, int, str]] = []
        required_claims: list[tuple[str, int, str]] = []
        required_chunks: list[tuple[str, int, str]] = []
        support_by_claim: dict[str, set[str]] = {}

        for idx, mutation in enumerate(mutations):
            op = mutation["op"]
            if op == "upsert_node":
                proposed_nodes.add(self._node_id_for_upsert(mutation))
            elif op == "add_edge":
                proposed_edges.add(self._edge_id_for_add(mutation))
            elif op == "upsert_claim" and mutation.get("claim_id"):
                proposed_claims.add(str(mutation["claim_id"]))

        for idx, mutation in enumerate(mutations):
            op = mutation["op"]
            if op == "add_alias":
                required_nodes.append((mutation["node_id"], idx, "add_alias.node_id"))
            elif op == "add_edge":
                required_nodes.append((mutation["from_node_id"], idx, "add_edge.from_node_id"))
                required_nodes.append((mutation["to_node_id"], idx, "add_edge.to_node_id"))
            elif op == "upsert_claim":
                if mutation.get("target_node_id"):
                    required_nodes.append((mutation["target_node_id"], idx, "upsert_claim.target_node_id"))
                if mutation.get("target_edge_id"):
                    required_edges.append((mutation["target_edge_id"], idx, "upsert_claim.target_edge_id"))
                if mutation.get("superseded_by_claim_id"):
                    required_claims.append((mutation["superseded_by_claim_id"], idx, "upsert_claim.superseded_by_claim_id"))
            elif op == "attach_evidence":
                required_claims.append((mutation["claim_id"], idx, "attach_evidence.claim_id"))
                required_chunks.append((mutation["chunk_id"], idx, "attach_evidence.chunk_id"))
                if mutation.get("evidence_role", "supports") == "supports" and mutation.get("status", "active") == "active":
                    support_by_claim.setdefault(str(mutation["claim_id"]), set()).add(str(mutation["chunk_id"]))
            elif op == "mark_claim_status":
                required_claims.append((mutation["claim_id"], idx, "mark_claim_status.claim_id"))
                if mutation.get("superseded_by_claim_id"):
                    required_claims.append((mutation["superseded_by_claim_id"], idx, "mark_claim_status.superseded_by_claim_id"))

        node_ids = {x[0] for x in required_nodes}
        edge_ids = {x[0] for x in required_edges}
        claim_ids = {x[0] for x in required_claims} | set(support_by_claim)
        chunk_ids = {x[0] for x in required_chunks}

        existing_nodes = self._existing_ids("nodes", "node_id", node_ids)
        existing_edges = self._existing_ids("edges", "edge_id", edge_ids)
        existing_claims = self._existing_ids("claims", "claim_id", claim_ids)
        existing_chunks = self._existing_ids("chunks", "chunk_id", chunk_ids)
        active_chunks = self._active_chunk_ids(chunk_ids)

        errors: list[dict[str, Any]] = []
        for node_id, idx, field in required_nodes:
            if node_id not in existing_nodes and node_id not in proposed_nodes:
                errors.append({"type": "missing_node_reference", "mutation_index": idx, "field": field, "node_id": node_id})
        for edge_id, idx, field in required_edges:
            if edge_id not in existing_edges and edge_id not in proposed_edges:
                errors.append({"type": "missing_edge_reference", "mutation_index": idx, "field": field, "edge_id": edge_id})
        for claim_id, idx, field in required_claims:
            if claim_id not in existing_claims and claim_id not in proposed_claims:
                errors.append({"type": "missing_claim_reference", "mutation_index": idx, "field": field, "claim_id": claim_id})
        for chunk_id, idx, field in required_chunks:
            if chunk_id not in existing_chunks:
                errors.append(
                    {
                        "type": "missing_chunk_reference",
                        "mutation_index": idx,
                        "field": field,
                        "chunk_id": chunk_id,
                        "hint": "Use chunk_ids/chunk_refs returned by dg_ingest_source; chunk IDs are random and must not be inferred or constructed.",
                    }
                )

        for idx, mutation in enumerate(mutations):
            if mutation["op"] != "upsert_claim":
                continue
            classification = mutation.get("classification", "Fact")
            status = mutation.get("status", "active")
            if classification != "OpenQuestion" and not mutation.get("target_node_id") and not mutation.get("target_edge_id"):
                errors.append(
                    {
                        "type": "claim_without_target",
                        "mutation_index": idx,
                        "claim_id": mutation.get("claim_id"),
                        "classification": classification,
                        "hint": "Set target_node_id/target_edge_id, or use OpenQuestion for untargeted uncertainty.",
                    }
                )
            if status != "active" or classification == "OpenQuestion":
                continue
            claim_id = mutation.get("claim_id")
            if not claim_id:
                errors.append(
                    {
                        "type": "active_claim_requires_claim_id_for_evidence",
                        "mutation_index": idx,
                        "classification": classification,
                        "hint": "Provide claim_id plus chunk_ids/attach_evidence, or classify as OpenQuestion.",
                    }
                )
                continue
            proposal_support = bool(support_by_claim.get(str(claim_id), set()) & active_chunks)
            existing_support = self._claim_has_active_support(str(claim_id))
            if not proposal_support and not existing_support:
                errors.append(
                    {
                        "type": "active_claim_without_active_support",
                        "mutation_index": idx,
                        "claim_id": claim_id,
                        "classification": classification,
                        "hint": "Attach an active supporting chunk with chunk_ids/attach_evidence, or classify as OpenQuestion.",
                    }
                )

        if errors:
            raise ValueError("proposal reference validation failed: " + json.dumps(errors, indent=2))

    def _node_id_for_upsert(self, mutation: dict[str, Any]) -> str:
        return str(mutation.get("node_id") or f"{mutation['node_type']}.{normalize_name(mutation['canonical_name'])}")

    def _edge_id_for_add(self, mutation: dict[str, Any]) -> str:
        row = self.conn.execute(
            "SELECT edge_id FROM edges WHERE from_node_id=? AND relation=? AND to_node_id=?",
            (mutation["from_node_id"], mutation["relation"], mutation["to_node_id"]),
        ).fetchone()
        if row:
            return str(row["edge_id"])
        return str(
            mutation.get("edge_id")
            or "edge_"
            + hashlib.sha1(f"{mutation['from_node_id']}:{mutation['relation']}:{mutation['to_node_id']}".encode()).hexdigest()[:16]
        )

    def _existing_ids(self, table: str, column: str, ids: set[str]) -> set[str]:
        if not ids:
            return set()
        out: set[str] = set()
        ordered = sorted(ids)
        for start in range(0, len(ordered), 500):
            batch = ordered[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rows = self.conn.execute(f"SELECT {column} FROM {table} WHERE {column} IN ({marks})", batch).fetchall()
            out.update(str(row[column]) for row in rows)
        return out

    def _active_chunk_ids(self, chunk_ids: set[str]) -> set[str]:
        if not chunk_ids:
            return set()
        out: set[str] = set()
        ordered = sorted(chunk_ids)
        for start in range(0, len(ordered), 500):
            batch = ordered[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"""
                SELECT ch.chunk_id
                FROM chunks ch
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ch.chunk_id IN ({marks})
                  AND ch.status='active'
                  AND ep.status='active'
                  AND s.status='active'
                """,
                batch,
            ).fetchall()
            out.update(str(row["chunk_id"]) for row in rows)
        return out

    def _claim_has_active_support(self, claim_id: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM claim_evidence ce
            JOIN chunks ch ON ch.chunk_id=ce.chunk_id
            JOIN episodes ep ON ep.episode_id=ch.episode_id
            JOIN sources s ON s.source_id=ch.source_id
            WHERE ce.claim_id=?
              AND ce.evidence_role='supports'
              AND ce.status='active'
              AND ch.status='active'
              AND ep.status='active'
              AND s.status='active'
            LIMIT 1
            """,
            (claim_id,),
        ).fetchone()
        return row is not None

    # ---------- validation/render/stale ----------

    def validate(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._log("debug", "validate.start")
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for r in self.conn.execute("SELECT alias_id, node_id FROM aliases WHERE node_id NOT IN (SELECT node_id FROM nodes)"):
            errors.append({"type": "alias_missing_node", **dict(r)})
        for r in self.conn.execute("SELECT edge_id, from_node_id, to_node_id FROM edges WHERE from_node_id NOT IN (SELECT node_id FROM nodes) OR to_node_id NOT IN (SELECT node_id FROM nodes)"):
            errors.append({"type": "edge_missing_node", **dict(r)})
        for r in self.conn.execute("SELECT claim_id, classification, claim_text FROM claims WHERE target_node_id IS NULL AND target_edge_id IS NULL AND classification <> 'OpenQuestion'"):
            errors.append({"type": "claim_without_target", **dict(r)})
        for r in self.conn.execute(
            """
            SELECT c.claim_id, c.classification, c.claim_text
            FROM claims c
            WHERE c.status='active'
              AND c.classification <> 'OpenQuestion'
              AND NOT EXISTS (
                SELECT 1 FROM claim_evidence ce
                JOIN chunks ch ON ch.chunk_id=ce.chunk_id
                JOIN episodes ep ON ep.episode_id=ch.episode_id
                JOIN sources s ON s.source_id=ch.source_id
                WHERE ce.claim_id=c.claim_id
                  AND ce.evidence_role='supports'
                  AND ce.status='active'
                  AND ch.status='active'
                  AND ep.status='active'
                  AND s.status='active'
              )
            """
        ):
            errors.append({"type": "active_claim_without_active_support", **dict(r)})
        for r in self.conn.execute("SELECT claim_id, chunk_id FROM claim_evidence WHERE claim_id NOT IN (SELECT claim_id FROM claims) OR chunk_id NOT IN (SELECT chunk_id FROM chunks)"):
            errors.append({"type": "evidence_broken_link", **dict(r)})
        for r in self.conn.execute(
            """
            SELECT ch.chunk_id, ch.episode_id, ch.source_id
            FROM chunks ch
            LEFT JOIN episodes ep ON ep.episode_id=ch.episode_id
            LEFT JOIN sources s ON s.source_id=ch.source_id
            WHERE ch.status='active'
              AND (ep.episode_id IS NULL OR s.source_id IS NULL OR ep.status <> 'active' OR s.status <> 'active')
            """
        ):
            errors.append({"type": "active_chunk_without_active_episode_source", **dict(r)})

        allowed_visibility = self._allowed_visibility_values()
        allowed_roles = self._allowed_roles()
        allowed_tags = self._allowed_interface_tags()
        for table, id_col in (("nodes", "node_id"), ("edges", "edge_id"), ("claims", "claim_id")):
            for r in self.conn.execute(f"SELECT {id_col}, visibility, finder_role, audience_roles_json, interface_tags_json FROM {table}"):
                item = dict(r)
                if item.get("visibility") not in allowed_visibility:
                    errors.append({"type": "invalid_visibility", "table": table, **item})
                finder = item.get("finder_role")
                if finder and finder not in allowed_roles:
                    errors.append({"type": "invalid_finder_role", "table": table, **item})
                bad_roles = [role for role in json_list(item.get("audience_roles_json")) if role not in allowed_roles]
                if bad_roles:
                    errors.append({"type": "invalid_audience_roles", "table": table, "bad_roles": bad_roles, **item})
                bad_tags = [tag for tag in json_list(item.get("interface_tags_json")) if tag not in allowed_tags]
                if bad_tags:
                    errors.append({"type": "invalid_interface_tags", "table": table, "bad_tags": bad_tags, **item})

        for r in self.conn.execute(
            """
            SELECT normalized_alias, COUNT(DISTINCT node_id) AS node_count, GROUP_CONCAT(DISTINCT node_id) AS node_ids
            FROM aliases
            GROUP BY normalized_alias
            HAVING COUNT(DISTINCT node_id) > 1
            """
        ):
            warnings.append({"type": "alias_collision_across_nodes", **dict(r)})
        result = {"ok": not errors, "errors": errors, "warnings": warnings, "graph_revision": self._revision()}
        self._log("info", "validate.done", ok=result["ok"], error_count=len(errors), warning_count=len(warnings), graph_revision=result["graph_revision"], elapsed_ms=(time.perf_counter() - started) * 1000)
        self._log(TRACE_LEVEL, "validate.details", errors=errors, warnings=warnings)
        return result

    def _safe_output_dir(self, output_dir: str | Path | None) -> Path:
        out = Path(output_dir) if output_dir else self.root / "docs" / "rendered"
        if not out.is_absolute():
            out = self.root / out
        resolved = out.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Refusing to render docs outside DOCGRAPH_ROOT: {output_dir}") from exc
        return resolved

    def render_docs(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        self._log("info", "render_docs.start", output_dir=str(output_dir) if output_dir else None)
        out = self._safe_output_dir(output_dir)
        nodes_dir = out / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        index_lines = ["# Rendered DocGraph", "", "Generated from SQLite. Do not edit manually.", "", "## Nodes", ""]
        for n in self.conn.execute("SELECT * FROM nodes ORDER BY node_type, canonical_name"):
            node = dict(n)
            filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", node["node_id"]) + ".md"
            index_lines.append(f"- [{node['node_id']}](nodes/{filename})")
            text = self._render_node_md(node)
            (nodes_dir / filename).write_text(text, encoding="utf-8")
            count += 1
        (out / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        result = {"rendered_nodes": count, "output_dir": str(out)}
        self._log("info", "render_docs.done", **result, elapsed_ms=(time.perf_counter() - started) * 1000)
        return result

    def stale_scan(self, auto_ingest: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        self._log("info", "stale_scan.start", auto_ingest=auto_ingest)
        changed = []
        missing = []
        ingested = []
        file_types = cfg_list(self.config, "source_handling.file_backed_source_types")
        if not file_types:
            result = {"changed": [], "missing": [], "auto_ingested": []}
            self._log("info", "stale_scan.done", changed_count=0, missing_count=0, auto_ingested_count=0, reason="no file-backed source types configured", elapsed_ms=(time.perf_counter() - started) * 1000)
            return result
        marks = ",".join("?" for _ in file_types)
        for s in self.conn.execute(f"SELECT * FROM sources WHERE source_type IN ({marks})", file_types):
            uri = s["uri"]
            try:
                content = self._safe_read_uri(uri)
            except FileNotFoundError:
                missing.append({"source_id": s["source_id"], "uri": uri})
                continue
            except ValueError:
                continue
            h = sha256_text(content)
            if h != s["current_hash"]:
                changed.append({"source_id": s["source_id"], "uri": uri, "old_hash": s["current_hash"], "new_hash": h})
                if auto_ingest:
                    ingested.append(self.ingest_source(s["source_type"], uri, episode_type="snapshot", name=s["name"]))
        result = {"changed": changed, "missing": missing, "auto_ingested": ingested}
        self._log("info", "stale_scan.done", changed_count=len(changed), missing_count=len(missing), auto_ingested_count=len(ingested), elapsed_ms=(time.perf_counter() - started) * 1000)
        self._log(TRACE_LEVEL, "stale_scan.details", result=result)
        return result

    # ---------- helpers ----------

    def _revision(self) -> str:
        return self.conn.execute("SELECT value FROM metadata WHERE key='graph_revision'").fetchone()["value"]

    def _get_node(self, node_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return dict(r) if r else None

    def _edge_with_names(self, edge: dict[str, Any]) -> dict[str, Any]:
        out = dict(edge)
        from_node = self._get_node(out["from_node_id"])
        to_node = self._get_node(out["to_node_id"])
        if from_node:
            out["from_canonical_name"] = from_node.get("canonical_name")
            out["from_node_type"] = from_node.get("node_type")
        if to_node:
            out["to_canonical_name"] = to_node.get("canonical_name")
            out["to_node_type"] = to_node.get("node_type")
        return out

    def _node_degree(self, node_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM edges WHERE status='active' AND (from_node_id=? OR to_node_id=?)",
            (node_id, node_id),
        ).fetchone()
        return int(row["n"] if row else 0)

    def _aliases_for_node(self, node_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT alias, alias_kind, confidence FROM aliases WHERE node_id=? ORDER BY alias", (node_id,))]

    def _claim_target_nodes(self, claim_row: sqlite3.Row | dict[str, Any]) -> set[str]:
        c = dict(claim_row)
        out: set[str] = set()
        if c.get("target_node_id"):
            out.add(c["target_node_id"])
        if c.get("target_edge_id"):
            e = self.conn.execute("SELECT * FROM edges WHERE edge_id=?", (c["target_edge_id"],)).fetchone()
            if e:
                out.add(e["from_node_id"])
                out.add(e["to_node_id"])
        return out

    def _claims_for_chunk(self, chunk_id: str, include_stale: bool = False, role: str | None = None) -> list[dict[str, Any]]:
        filt = "" if include_stale else "AND c.status='active' AND ce.status='active'"
        rows = [
            dict(r)
            for r in self.conn.execute(
                f"""
                SELECT c.*
                FROM claim_evidence ce JOIN claims c ON c.claim_id=ce.claim_id
                WHERE ce.chunk_id=? {filt}
                """,
                (chunk_id,),
            )
        ]
        return [r for r in rows if self._claim_visible_to_role(r, role)]

    def _claims_for_node(self, node_id: str, include_stale: bool, limit: int, role: str | None = None) -> list[dict[str, Any]]:
        status = "" if include_stale else "AND status='active'"
        rows = [
            dict(r)
            for r in self.conn.execute(
                f"SELECT * FROM claims WHERE target_node_id=? {status} ORDER BY updated_at DESC LIMIT ?",
                (node_id, max(limit * 4, limit)),
            )
        ]
        return [r for r in rows if self._claim_visible_to_role(r, role)][:limit]

    def _edges_for_node(self, node_id: str, include_stale: bool, limit: int) -> list[dict[str, Any]]:
        status = "" if include_stale else "AND status='active'"
        return [
            self._edge_with_names(dict(r))
            for r in self.conn.execute(
                f"SELECT * FROM edges WHERE (from_node_id=? OR to_node_id=?) {status} LIMIT ?",
                (node_id, node_id, limit),
            )
        ]

    def _claims_for_edge(self, edge_id: str, include_stale: bool, limit: int, role: str | None = None) -> list[dict[str, Any]]:
        status = "" if include_stale else "AND status='active'"
        rows = [
            dict(r)
            for r in self.conn.execute(
                f"SELECT * FROM claims WHERE target_edge_id=? {status} ORDER BY updated_at DESC LIMIT ?",
                (edge_id, max(limit * 4, limit)),
            )
        ]
        return [r for r in rows if self._claim_visible_to_role(r, role)][:limit]

    def _evidence_for_claim(self, claim_id: str, limit: int = 2) -> list[dict[str, Any]]:
        rows = []
        for r in self.conn.execute(
            """
            SELECT ce.evidence_role, ce.strength, ce.status AS evidence_status,
                   ch.chunk_id, ch.locator, ch.text, ch.status AS chunk_status,
                   s.uri, s.source_type
            FROM claim_evidence ce
            JOIN chunks ch ON ch.chunk_id=ce.chunk_id
            JOIN sources s ON s.source_id=ch.source_id
            WHERE ce.claim_id=?
            ORDER BY CASE ce.evidence_role WHEN 'supports' THEN 0 WHEN 'refutes' THEN 1 ELSE 2 END
            LIMIT ?
            """,
            (claim_id, limit),
        ):
            d = dict(r)
            d["text_preview"] = self._preview(d.pop("text"))
            rows.append(d)
        return rows

    def _short_chunk(self, d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        d["text_preview"] = self._preview(d.pop("text", ""))
        d.pop("raw_text", None)
        return d

    def _preview(self, text: str, n: int = 360) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= n else text[: n - 3] + "..."

    def _dedupe_by(self, items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for item in items:
            v = item.get(key)
            if v in seen:
                continue
            seen.add(v)
            out.append(item)
        return out

    def _rank_weight(self, name: str, default: float) -> float:
        return cfg_number(self.config, f"retrieval.ranking.{name}", default)

    def _budget_limits(self, budget: str) -> dict[str, int]:
        raw = cfg_get(self.config, f"retrieval.budgets.{budget}") or cfg_get(self.config, "retrieval.budgets.small") or {}
        defaults = {"nodes": 5, "claims": 4, "edges": 8, "evidence": 2}
        out = defaults | {k: int(v) for k, v in raw.items() if k in defaults}
        return out

    def _role_config(self, role: str | None) -> dict[str, Any]:
        """Compatibility wrapper for role config lookup."""
        return self._visibility_policy.role_config(role)

    def _role_node_bonus(self, node_id: str, role: str | None) -> float:
        """Compatibility wrapper for role node-type ranking bonuses."""
        return self._visibility_policy.role_node_bonus(node_id, role)

    def _role_nodes_bonus(self, node_ids: Iterable[str], role: str | None) -> float:
        """Compatibility wrapper for aggregate role node bonuses."""
        return self._visibility_policy.role_nodes_bonus(node_ids, role)

    def _node_visible_to_role(self, node_id: str, role: str | None) -> bool:
        """Compatibility wrapper for node visibility checks."""
        return self._visibility_policy.node_visible_to_role(node_id, role)

    def _role_relation_bonus(self, relation: str | None, role: str | None) -> float:
        """Compatibility wrapper for role relation ranking bonuses."""
        return self._visibility_policy.role_relation_bonus(relation, role)

    def _row_visibility(self, row: sqlite3.Row | dict[str, Any]) -> str:
        """Compatibility wrapper for row visibility resolution."""
        return self._visibility_policy.row_visibility(row)

    def _row_audience_roles(self, row: sqlite3.Row | dict[str, Any]) -> list[str]:
        """Compatibility wrapper for row audience role parsing."""
        return self._visibility_policy.row_audience_roles(row)

    def _row_interface_tags(self, row: sqlite3.Row | dict[str, Any]) -> list[str]:
        """Compatibility wrapper for row interface tag parsing."""
        return self._visibility_policy.row_interface_tags(row)

    def _row_visible_to_role(self, row: sqlite3.Row | dict[str, Any], role: str | None) -> bool:
        """Compatibility wrapper for role-aware row visibility checks."""
        return self._visibility_policy.row_visible_to_role(row, role)

    def _claim_visible_to_role(self, row: sqlite3.Row | dict[str, Any], role: str | None) -> bool:
        """Compatibility wrapper for claim visibility checks."""
        return self._visibility_policy.claim_visible_to_role(row, role)

    def _role_row_bonus(self, row: sqlite3.Row | dict[str, Any], role: str | None) -> float:
        """Compatibility wrapper for row role-affinity bonus scoring."""
        return self._visibility_policy.role_row_bonus(row, role)

    def _role_claim_bonus(self, row: sqlite3.Row | dict[str, Any], role: str | None) -> float:
        """Compatibility wrapper for claim role-affinity bonus scoring."""
        return self._visibility_policy.role_claim_bonus(row, role)

    def _cross_role_notes(self, claims: list[dict[str, Any]], edges: list[dict[str, Any]], role: str | None) -> list[dict[str, Any]]:
        """Compatibility wrapper for cross-role visibility note generation."""
        return self._visibility_policy.cross_role_notes(claims, edges, role)

    def _sort_edges_for_role(self, edges: list[dict[str, Any]], role: str | None) -> list[dict[str, Any]]:
        """Compatibility wrapper for role-aware edge sorting."""
        return self._visibility_policy.sort_edges_for_role(edges, role)

    def _allowed_node_types(self) -> set[str]:
        return set(cfg_list(self.config, "taxonomy.node_types"))

    def _allowed_relation_types(self) -> set[str]:
        return set(cfg_list(self.config, "taxonomy.relation_types"))

    def _allowed_claim_classes(self) -> set[str]:
        return set(cfg_list(self.config, "data_model.claim_classes"))

    def _allowed_claim_statuses(self) -> set[str]:
        return set(cfg_list(self.config, "data_model.claim_statuses"))

    def _allowed_evidence_roles(self) -> set[str]:
        return set(cfg_list(self.config, "data_model.evidence_roles"))

    def _allowed_visibility_values(self) -> set[str]:
        return set(cfg_list(self.config, "shared_knowledge.visibility_values")) or {"local", "shared", "global", "shared_candidate"}

    def _allowed_interface_tags(self) -> set[str]:
        return set(cfg_list(self.config, "shared_knowledge.interface_tags"))

    def _allowed_roles(self) -> set[str]:
        roles = cfg_get(self.config, "roles", {})
        return set(roles.keys()) if isinstance(roles, dict) else set()

    def _suggest_next_checks(self, nodes: list[dict[str, Any]], role: str | None, intent: str | None) -> list[str]:
        """Compatibility wrapper for role/intent next-check suggestion hints."""
        return self._visibility_policy.suggest_next_checks(nodes, role, intent)

    def _context_markdown(self, p: ContextPacket) -> str:
        """Compatibility wrapper for context packet markdown rendering."""
        return self._retrieval_flow.context_markdown(p)

    def _render_node_md(self, node: dict[str, Any]) -> str:
        lines = [f"# {node['node_id']}", "", "Generated from DocGraph SQLite. Do not edit manually.", ""]
        lines.append(f"Type: `{node['node_type']}`")
        lines.append(f"Canonical name: `{node['canonical_name']}`")
        lines.append(f"Visibility: `{node.get('visibility', 'local')}`")
        if node.get("finder_role"):
            lines.append(f"Finder role: `{node['finder_role']}`")
        if self._row_audience_roles(node):
            lines.append(f"Audience roles: `{', '.join(self._row_audience_roles(node))}`")
        if self._row_interface_tags(node):
            lines.append(f"Interface tags: `{', '.join(self._row_interface_tags(node))}`")
        if node.get("summary"):
            lines.append("")
            lines.append(node["summary"])
        aliases = self._aliases_for_node(node["node_id"])
        if aliases:
            lines.append("")
            lines.append("## Aliases")
            for a in aliases:
                lines.append(f"- {a['alias']} ({a['alias_kind']}, {a['confidence']})")
        claims = self._claims_for_node(node["node_id"], include_stale=True, limit=100)
        if claims:
            lines.append("")
            lines.append("## Claims")
            for c in claims:
                lines.append(f"- **{c['status']} / {c['classification']} / {c['confidence']} / {c.get('visibility', 'local')}**: {c['claim_text']}")
                for ev in self._evidence_for_claim(c["claim_id"], limit=3):
                    lines.append(f"  - evidence {ev['evidence_role']}: {ev['uri']} {ev.get('locator') or ''}")
        edges = self._edges_for_node(node["node_id"], include_stale=True, limit=100)
        if edges:
            lines.append("")
            lines.append("## Related edges")
            for e in edges:
                summary = f" — {e['summary']}" if e.get("summary") else ""
                lines.append(f"- {e['from_node_id']} --{e['relation']}--> {e['to_node_id']} ({e['status']}, {e['confidence']}, {e.get('visibility', 'local')}){summary}")
        return "\n".join(lines) + "\n"
