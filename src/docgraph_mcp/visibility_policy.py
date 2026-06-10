from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .config import cfg_get


class VisibilityPolicy:
    """Role visibility and ranking policy extracted from ``DocGraphBackend``."""

    def __init__(self, backend: Any, *, list_parser: Callable[[Any], list[str]]) -> None:
        """Bind backend access and audience/tag list parsing helper."""
        self.backend = backend
        self._json_list = list_parser

    def role_config(self, role: str | None) -> dict[str, Any]:
        """Return role config map or empty map for unknown/empty role."""
        if not role:
            return {}
        cfg = cfg_get(self.backend.config, f"roles.{role}", {})
        return cfg if isinstance(cfg, dict) else {}

    def role_node_bonus(self, node_id: str, role: str | None) -> float:
        """Score bonus when node type matches role preferred node types."""
        node = self.backend._get_node(node_id)
        if not node:
            return 0.0
        prefs = set(self.role_config(role).get("preferred_node_types", []))
        return self.backend._rank_weight("role_preferred_node_bonus", 8.0) if node.get("node_type") in prefs else 0.0

    def role_nodes_bonus(self, node_ids: Iterable[str], role: str | None) -> float:
        """Best node bonus across a set of node ids."""
        return max((self.role_node_bonus(nid, role) for nid in node_ids), default=0.0)

    def node_visible_to_role(self, node_id: str, role: str | None) -> bool:
        """Check role visibility for one node id."""
        node = self.backend._get_node(node_id)
        return bool(node and self.row_visible_to_role(node, role))

    def role_relation_bonus(self, relation: str | None, role: str | None) -> float:
        """Score bonus when relation matches role preferred relations."""
        prefs = set(self.role_config(role).get("preferred_relations", []))
        return self.backend._rank_weight("role_preferred_relation_bonus", 6.0) if relation in prefs else 0.0

    def row_visibility(self, row: Any) -> str:
        """Return row visibility or configured default."""
        d = dict(row)
        return d.get("visibility") or cfg_get(self.backend.config, "shared_knowledge.default_visibility", "local")

    def row_audience_roles(self, row: Any) -> list[str]:
        """Return parsed audience roles from row payload."""
        return self._json_list(dict(row).get("audience_roles_json") or dict(row).get("audience_roles"))

    def row_interface_tags(self, row: Any) -> list[str]:
        """Return parsed interface tags from row payload."""
        return self._json_list(dict(row).get("interface_tags_json") or dict(row).get("interface_tags"))

    def row_visible_to_role(self, row: Any, role: str | None) -> bool:
        """Apply role/visibility policy for a row-like object."""
        if not role:
            return True
        d = dict(row)
        visibility = self.row_visibility(d)
        finder_role = d.get("finder_role")
        audiences = set(self.row_audience_roles(d))
        # Backward compatibility: old graph rows have no finder/audience metadata.
        # Treat unclassified local rows as generally visible until the curator
        # explicitly narrows them.
        if visibility == "local" and not finder_role and not audiences:
            return True
        if role == finder_role or role in audiences:
            return True
        if visibility == "global":
            return True
        if visibility in set(self.role_config(role).get("include_visibility", [])):
            # If no explicit audience is set, shared/global candidates are visible to all roles that opt in.
            if not audiences or role in audiences or visibility in {"global", "shared_candidate"}:
                return True
        return False

    def claim_visible_to_role(self, row: Any, role: str | None) -> bool:
        """Claims use the same visibility policy as other rows."""
        return self.row_visible_to_role(row, role)

    def role_row_bonus(self, row: Any, role: str | None) -> float:
        """Score role/visibility affinity bonus for a row."""
        if not role:
            return 0.0
        d = dict(row)
        visibility = self.row_visibility(d)
        bonus = 0.0
        if role in self.row_audience_roles(d) or role == d.get("finder_role"):
            bonus += self.backend._rank_weight("role_audience_match_bonus", 12.0)
        if visibility == "shared":
            bonus += self.backend._rank_weight("shared_visibility_bonus", 7.0)
        elif visibility == "global":
            bonus += self.backend._rank_weight("global_visibility_bonus", 9.0)
        elif visibility == "shared_candidate":
            bonus += self.backend._rank_weight("shared_candidate_bonus", 3.0)
        elif visibility == "local" and role != d.get("finder_role") and role not in self.row_audience_roles(d):
            bonus += self.backend._rank_weight("local_cross_role_penalty", -100.0)
        return bonus

    def role_claim_bonus(self, row: Any, role: str | None) -> float:
        """Claims currently use the same role bonus as generic rows."""
        return self.role_row_bonus(row, role)

    def cross_role_notes(self, claims: list[dict[str, Any]], edges: list[dict[str, Any]], role: str | None) -> list[dict[str, Any]]:
        """Build compact notes for shared/global/shared-candidate results."""
        _ = role  # retained for interface compatibility and future policy tweaks
        notes: list[dict[str, Any]] = []
        for item_type, items in (("claim", claims), ("edge", edges)):
            for item in items:
                visibility = self.row_visibility(item)
                if visibility in {"shared", "global", "shared_candidate"}:
                    notes.append({
                        "kind": item_type,
                        "id": item.get(f"{item_type}_id"),
                        "visibility": visibility,
                        "finder_role": item.get("finder_role"),
                        "audience_roles": self.row_audience_roles(item),
                        "interface_tags": self.row_interface_tags(item),
                        "note": "shared_candidate is a visibility warning, not a verified cross-role edge" if visibility == "shared_candidate" else "cross-role visible knowledge",
                    })
        return notes[:20]

    def sort_edges_for_role(self, edges: list[dict[str, Any]], role: str | None) -> list[dict[str, Any]]:
        """Sort edges by role relation affinity and visibility bonuses."""
        return sorted(edges, key=lambda e: -(self.role_relation_bonus(e.get("relation"), role) + self.role_row_bonus(e, role)))

    def suggest_next_checks(self, nodes: list[dict[str, Any]], role: str | None, intent: str | None) -> list[str]:
        """Build role/intent-specific checklist hints for context packets."""
        _ = nodes  # reserved for future node-aware checks
        checks: list[str] = []
        role_checks = self.role_config(role).get("suggested_checks", [])
        if isinstance(role_checks, list):
            checks.extend(str(x) for x in role_checks)
        intent_checks = cfg_get(self.backend.config, f"intents.{intent}.suggested_checks", []) if intent else []
        if isinstance(intent_checks, list):
            checks.extend(str(x) for x in intent_checks)
        if not checks:
            checks.append("Verify selected anchors against current source before using them as proof.")
        return list(dict.fromkeys(checks))
