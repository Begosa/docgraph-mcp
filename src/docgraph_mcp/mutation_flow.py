from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from .config import cfg_get

CANONICAL_MUTATION_OPS = {
    "upsert_node",
    "add_alias",
    "add_edge",
    "upsert_claim",
    "attach_evidence",
    "mark_claim_status",
}

MUTATION_OP_ALIASES = {
    # node
    "node": "upsert_node",
    "add_node": "upsert_node",
    "create_node": "upsert_node",
    "update_node": "upsert_node",
    # alias
    "alias": "add_alias",
    "upsert_alias": "add_alias",
    "create_alias": "add_alias",
    # edge
    "edge": "add_edge",
    "upsert_edge": "add_edge",
    "create_edge": "add_edge",
    # claim
    "claim": "upsert_claim",
    "add_claim": "upsert_claim",
    "create_claim": "upsert_claim",
    # evidence
    "evidence": "attach_evidence",
    "claim_evidence": "attach_evidence",
    "add_evidence": "attach_evidence",
    "attach_claim_evidence": "attach_evidence",
    # status
    "status": "mark_claim_status",
    "claim_status": "mark_claim_status",
    "update_claim_status": "mark_claim_status",
}


class MutationFlow:
    """Normalize, validate, and apply canonical DocGraph mutations."""

    def __init__(
        self,
        backend: Any,
        *,
        id_factory: Callable[[str], str],
        ts_factory: Callable[[], str],
        normalize_name: Callable[[str], str],
        list_parser: Callable[[Any], list[str]],
        list_to_text: Callable[[Any], str],
    ) -> None:
        """Bind backend access and helper callbacks for mutation execution."""
        self.backend = backend
        self._new_id = id_factory
        self._now_ts = ts_factory
        self._normalize_name = normalize_name
        self._json_list = list_parser
        self._json_list_text = list_to_text

    def normalize_mutations(self, mutations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize every mutation and expand compact claim-evidence shortcuts."""
        out: list[dict[str, Any]] = []
        for mutation in mutations:
            normalized = self.normalize_mutation(mutation)
            out.append(normalized)
            out.extend(self.evidence_mutations_from_claim_shortcut(normalized))
        return out

    def evidence_mutations_from_claim_shortcut(self, mutation: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand ``upsert_claim.chunk_ids`` into canonical ``attach_evidence`` ops."""
        if mutation.get("op") != "upsert_claim":
            return []
        chunk_ids = mutation.pop("chunk_ids", None)
        if chunk_ids is None:
            chunk_ids = mutation.pop("evidence_chunk_ids", None)
        if chunk_ids is None:
            chunk_ids = mutation.pop("supporting_chunk_ids", None)
        if chunk_ids is None:
            return []
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        if not isinstance(chunk_ids, list):
            raise ValueError("upsert_claim chunk_ids must be a string or list of strings")
        claim_id = mutation.get("claim_id")
        if not claim_id:
            claim_id = self._new_id("claim")
            mutation["claim_id"] = claim_id
        evidence_role = mutation.pop("evidence_role", "supports")
        strength = mutation.pop("evidence_strength", mutation.pop("strength", "medium"))
        return [
            {
                "op": "attach_evidence",
                "claim_id": claim_id,
                "chunk_id": chunk_id,
                "evidence_role": evidence_role,
                "strength": strength,
            }
            for chunk_id in chunk_ids
            if chunk_id
        ]

    def infer_mutation_op(self, mutation: dict[str, Any]) -> str | None:
        """Infer canonical op from field shape when ``op`` is omitted."""
        if mutation.get("node_type") or mutation.get("canonical_name") or (mutation.get("type") and mutation.get("name")):
            return "upsert_node"
        if mutation.get("alias") and mutation.get("node_id"):
            return "add_alias"
        if (mutation.get("from_node_id") or mutation.get("from") or mutation.get("source_node_id")) and (
            mutation.get("to_node_id") or mutation.get("to") or mutation.get("target_node_id")
        ):
            return "add_edge"
        if mutation.get("claim_text") or mutation.get("text") or mutation.get("statement"):
            return "upsert_claim"
        if mutation.get("claim_id") and mutation.get("chunk_id"):
            return "attach_evidence"
        if mutation.get("claim_id") and mutation.get("status"):
            return "mark_claim_status"
        return None

    def normalize_mutation(self, mutation: dict[str, Any]) -> dict[str, Any]:
        """Map aliases and shorthand fields into canonical mutation keys."""
        if not isinstance(mutation, dict):
            raise ValueError("mutation must be object")
        normalized = dict(mutation)
        raw_op = normalized.get("op") or self.infer_mutation_op(normalized)
        op = MUTATION_OP_ALIASES.get(raw_op, raw_op)
        if op is not None:
            normalized["op"] = op

        if op == "upsert_node":
            if "canonical_name" not in normalized and "name" in normalized:
                normalized["canonical_name"] = normalized["name"]
            if "node_type" not in normalized and "type" in normalized:
                normalized["node_type"] = normalized["type"]
        elif op == "add_alias":
            if "alias" not in normalized and "name" in normalized:
                normalized["alias"] = normalized["name"]
        elif op == "add_edge":
            if "from_node_id" not in normalized:
                normalized["from_node_id"] = normalized.get("from") or normalized.get("source_node_id") or normalized.get("source")
            if "to_node_id" not in normalized:
                normalized["to_node_id"] = normalized.get("to") or normalized.get("target_node_id") or normalized.get("target")
            if "relation" not in normalized:
                normalized["relation"] = normalized.get("relation_type") or normalized.get("type")
        elif op == "upsert_claim":
            if "claim_text" not in normalized:
                normalized["claim_text"] = normalized.get("text") or normalized.get("statement")
            if "chunk_ids" not in normalized:
                for alias in ("evidence_chunk_ids", "supporting_chunk_ids", "chunks"):
                    if alias in normalized:
                        normalized["chunk_ids"] = normalized[alias]
                        break
        elif op == "attach_evidence":
            if "evidence_role" not in normalized and "role" in normalized:
                normalized["evidence_role"] = normalized["role"]
        return normalized

    def validate_mutation_shapes(self, mutations: list[dict[str, Any]]) -> None:
        """Enforce op-specific required fields and taxonomy/visibility constraints."""
        allowed_ops = CANONICAL_MUTATION_OPS
        for mutation in mutations:
            if not isinstance(mutation, dict):
                raise ValueError("mutation must be object")
            op = mutation.get("op")
            if op not in allowed_ops:
                raise ValueError(
                    f"unsupported mutation op: {op}; allowed ops: {sorted(allowed_ops)}; accepted aliases: {sorted(MUTATION_OP_ALIASES)}"
                )
            if op == "upsert_node":
                missing = [key for key in ("node_type", "canonical_name") if not mutation.get(key)]
                if missing:
                    raise ValueError(f"upsert_node requires {', '.join(missing)}")
                if mutation.get("node_type") not in self.backend._allowed_node_types():
                    raise ValueError(f"bad node_type: {mutation.get('node_type')}; allowed node_types: {self.backend._allowed_node_types()}")
                self.validate_visibility_metadata(mutation)
            if op == "add_alias":
                missing = [key for key in ("node_id", "alias") if not mutation.get(key)]
                if missing:
                    raise ValueError(f"add_alias requires {', '.join(missing)}")
            if op == "add_edge":
                missing = [key for key in ("from_node_id", "relation", "to_node_id") if not mutation.get(key)]
                if missing:
                    raise ValueError(f"add_edge requires {', '.join(missing)}")
                if mutation.get("relation") not in self.backend._allowed_relation_types():
                    raise ValueError(
                        f"bad relation: {mutation.get('relation')}; allowed relation_types: {self.backend._allowed_relation_types()}"
                    )
                self.validate_visibility_metadata(mutation)
            if op == "upsert_claim":
                if not mutation.get("claim_text"):
                    raise ValueError("upsert_claim requires claim_text")
                if mutation.get("classification", "Fact") not in self.backend._allowed_claim_classes():
                    raise ValueError(f"bad classification: {mutation.get('classification')}")
                if mutation.get("status", "active") not in self.backend._allowed_claim_statuses():
                    raise ValueError(f"bad claim status: {mutation.get('status')}")
                self.validate_visibility_metadata(mutation)
            if op == "attach_evidence":
                missing = [key for key in ("claim_id", "chunk_id") if not mutation.get(key)]
                if missing:
                    raise ValueError(f"attach_evidence requires {', '.join(missing)}")
                if mutation.get("evidence_role", "supports") not in self.backend._allowed_evidence_roles():
                    raise ValueError(
                        f"bad evidence_role: {mutation.get('evidence_role')}; allowed evidence_roles: {self.backend._allowed_evidence_roles()}"
                    )
            if op == "mark_claim_status":
                missing = [key for key in ("claim_id", "status") if not mutation.get(key)]
                if missing:
                    raise ValueError(f"mark_claim_status requires {', '.join(missing)}")
                if mutation.get("status") not in self.backend._allowed_claim_statuses():
                    raise ValueError(
                        f"bad claim status: {mutation.get('status')}; allowed claim_statuses: {self.backend._allowed_claim_statuses()}"
                    )

    def validate_visibility_metadata(self, mutation: dict[str, Any]) -> None:
        """Validate visibility, finder role, audience roles, and interface tags."""
        visibility = mutation.get("visibility", cfg_get(self.backend.config, "shared_knowledge.default_visibility", "local"))
        if visibility not in self.backend._allowed_visibility_values():
            raise ValueError(
                f"bad visibility: {visibility}; allowed visibility values: {self.backend._allowed_visibility_values()}"
            )
        finder_role = mutation.get("finder_role")
        if finder_role and finder_role not in self.backend._allowed_roles():
            raise ValueError(f"bad finder_role: {finder_role}; allowed roles: {self.backend._allowed_roles()}")
        bad_roles = [role for role in self._json_list(mutation.get("audience_roles")) if role not in self.backend._allowed_roles()]
        if bad_roles:
            raise ValueError(f"bad audience_roles: {bad_roles}; allowed roles: {self.backend._allowed_roles()}")
        bad_tags = [tag for tag in self._json_list(mutation.get("interface_tags")) if tag not in self.backend._allowed_interface_tags()]
        if bad_tags:
            raise ValueError(f"bad interface_tags: {bad_tags}; allowed tags: {self.backend._allowed_interface_tags()}")

    def visibility_values_for_insert(self, mutation: dict[str, Any]) -> tuple[str, str | None, str, str]:
        """Return normalized visibility fields ready for SQL insert/update."""
        return (
            mutation.get("visibility", cfg_get(self.backend.config, "shared_knowledge.default_visibility", "local")),
            mutation.get("finder_role"),
            self._json_list_text(mutation.get("audience_roles")),
            self._json_list_text(mutation.get("interface_tags")),
        )

    def apply_mutation(self, mutation: dict[str, Any]) -> None:
        """Apply one validated canonical mutation inside the caller transaction."""
        op = mutation["op"]
        ts = self._now_ts()
        self.backend._log("debug", "mutation.apply", op=op, mutation=mutation)

        if op == "upsert_node":
            node_id = mutation.get("node_id") or f"{mutation['node_type']}.{self._normalize_name(mutation['canonical_name'])}"
            visibility, finder_role, audience_roles_json, interface_tags_json = self.visibility_values_for_insert(mutation)
            self.backend.conn.execute(
                """
                INSERT INTO nodes(node_id, node_type, canonical_name, summary, visibility, finder_role, audience_roles_json, interface_tags_json, status, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(node_id) DO UPDATE SET
                  node_type=excluded.node_type,
                  canonical_name=excluded.canonical_name,
                  summary=excluded.summary,
                  visibility=excluded.visibility,
                  finder_role=excluded.finder_role,
                  audience_roles_json=excluded.audience_roles_json,
                  interface_tags_json=excluded.interface_tags_json,
                  status=excluded.status,
                  updated_at=excluded.updated_at
                """,
                (
                    node_id,
                    mutation["node_type"],
                    mutation["canonical_name"],
                    mutation.get("summary"),
                    visibility,
                    finder_role,
                    audience_roles_json,
                    interface_tags_json,
                    mutation.get("status", "active"),
                    ts,
                    ts,
                ),
            )
            self.backend._upsert_node_fts(node_id, mutation["canonical_name"], mutation.get("summary"))
            self.backend._replace_node_terms(node_id, f"{mutation['canonical_name']} {mutation.get('summary') or ''}")
            return

        if op == "add_alias":
            node_id = mutation["node_id"]
            alias = mutation["alias"]
            normalized_alias = self._normalize_name(alias)
            requested_alias_id = mutation.get("alias_id") or "alias_" + hashlib.sha1(
                f"{node_id}:{normalized_alias}".encode()
            ).hexdigest()[:16]
            self.backend.conn.execute(
                """
                INSERT INTO aliases(alias_id, node_id, alias, normalized_alias, alias_kind, confidence, created_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(node_id, normalized_alias) DO UPDATE SET
                  alias=excluded.alias,
                  alias_kind=excluded.alias_kind,
                  confidence=excluded.confidence
                """,
                (
                    requested_alias_id,
                    node_id,
                    alias,
                    normalized_alias,
                    mutation.get("alias_kind", "name"),
                    mutation.get("confidence", "medium"),
                    ts,
                ),
            )
            row = self.backend.conn.execute(
                "SELECT alias_id FROM aliases WHERE node_id=? AND normalized_alias=?",
                (node_id, normalized_alias),
            ).fetchone()
            if not row:
                raise RuntimeError(f"failed to persist alias for node_id={node_id} normalized_alias={normalized_alias}")
            alias_id = str(row["alias_id"])
            self.backend._upsert_alias_fts(alias_id, node_id, alias, normalized_alias)
            if alias_id != requested_alias_id:
                self.backend.conn.execute("DELETE FROM aliases_fts WHERE alias_id=?", (requested_alias_id,))
            return

        if op == "add_edge":
            requested_edge_id = mutation.get("edge_id") or "edge_" + hashlib.sha1(
                f"{mutation['from_node_id']}:{mutation['relation']}:{mutation['to_node_id']}".encode()
            ).hexdigest()[:16]
            visibility, finder_role, audience_roles_json, interface_tags_json = self.visibility_values_for_insert(mutation)
            self.backend.conn.execute(
                """
                INSERT INTO edges(edge_id, from_node_id, relation, to_node_id, summary, visibility, finder_role, audience_roles_json, interface_tags_json, confidence, status, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(from_node_id, relation, to_node_id) DO UPDATE SET
                  summary=excluded.summary,
                  visibility=excluded.visibility,
                  finder_role=excluded.finder_role,
                  audience_roles_json=excluded.audience_roles_json,
                  interface_tags_json=excluded.interface_tags_json,
                  confidence=excluded.confidence,
                  status=excluded.status
                """,
                (
                    requested_edge_id,
                    mutation["from_node_id"],
                    mutation["relation"],
                    mutation["to_node_id"],
                    mutation.get("summary"),
                    visibility,
                    finder_role,
                    audience_roles_json,
                    interface_tags_json,
                    mutation.get("confidence", "medium"),
                    mutation.get("status", "active"),
                    ts,
                ),
            )
            row = self.backend.conn.execute(
                "SELECT edge_id FROM edges WHERE from_node_id=? AND relation=? AND to_node_id=?",
                (mutation["from_node_id"], mutation["relation"], mutation["to_node_id"]),
            ).fetchone()
            if not row:
                raise RuntimeError(
                    f"failed to persist edge for from={mutation['from_node_id']} relation={mutation['relation']} to={mutation['to_node_id']}"
                )
            edge_id = str(row["edge_id"])
            self.backend._upsert_edge_fts(
                edge_id,
                mutation["from_node_id"],
                mutation["relation"],
                mutation["to_node_id"],
                mutation.get("summary"),
            )
            self.backend._replace_edge_terms(
                edge_id,
                f"{mutation['from_node_id']} {mutation['relation']} {mutation['to_node_id']} {mutation.get('summary') or ''}",
            )
            if edge_id != requested_edge_id:
                self.backend.conn.execute("DELETE FROM edges_fts WHERE edge_id=?", (requested_edge_id,))
            return

        if op == "upsert_claim":
            claim_id = mutation.get("claim_id") or self._new_id("claim")
            visibility, finder_role, audience_roles_json, interface_tags_json = self.visibility_values_for_insert(mutation)
            self.backend.conn.execute(
                """
                INSERT INTO claims(claim_id, target_node_id, target_edge_id, claim_text, classification, confidence, visibility, finder_role, audience_roles_json, interface_tags_json, status, superseded_by_claim_id, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(claim_id) DO UPDATE SET
                  target_node_id=excluded.target_node_id,
                  target_edge_id=excluded.target_edge_id,
                  claim_text=excluded.claim_text,
                  classification=excluded.classification,
                  confidence=excluded.confidence,
                  visibility=excluded.visibility,
                  finder_role=excluded.finder_role,
                  audience_roles_json=excluded.audience_roles_json,
                  interface_tags_json=excluded.interface_tags_json,
                  status=excluded.status,
                  superseded_by_claim_id=excluded.superseded_by_claim_id,
                  updated_at=excluded.updated_at
                """,
                (
                    claim_id,
                    mutation.get("target_node_id"),
                    mutation.get("target_edge_id"),
                    mutation["claim_text"],
                    mutation.get("classification", "Fact"),
                    mutation.get("confidence", "medium"),
                    visibility,
                    finder_role,
                    audience_roles_json,
                    interface_tags_json,
                    mutation.get("status", "active"),
                    mutation.get("superseded_by_claim_id"),
                    ts,
                    ts,
                ),
            )
            self.backend._upsert_claim_fts(claim_id, mutation["claim_text"])
            return

        if op == "attach_evidence":
            self.backend.conn.execute(
                """
                INSERT INTO claim_evidence(claim_id, chunk_id, evidence_role, strength, status, created_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(claim_id, chunk_id, evidence_role) DO UPDATE SET
                  strength=excluded.strength,
                  status=excluded.status
                """,
                (
                    mutation["claim_id"],
                    mutation["chunk_id"],
                    mutation.get("evidence_role", "supports"),
                    mutation.get("strength", "medium"),
                    mutation.get("status", "active"),
                    ts,
                ),
            )
            return

        if op == "mark_claim_status":
            self.backend.conn.execute(
                "UPDATE claims SET status=?, superseded_by_claim_id=?, updated_at=? WHERE claim_id=?",
                (mutation["status"], mutation.get("superseded_by_claim_id"), ts, mutation["claim_id"]),
            )
            return

        raise ValueError(f"unsupported mutation op in apply: {op}")
