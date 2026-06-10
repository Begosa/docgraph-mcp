from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:  # PyYAML is optional at runtime; JSON config still works without it.
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": {
        "enabled": True,
        "level": "info",
        "file": "docs/logs/docgraph-mcp.log",
        "max_bytes": 5000000,
        "backup_count": 3,
        "stderr": False,
        "include_payloads": False,
        "payload_preview_chars": 500,
    },
    "data_model": {
        "claim_classes": ["Fact", "Inference", "Hypothesis", "OpenQuestion", "Contradiction"],
        "claim_statuses": ["active", "needs_review", "stale", "superseded", "contradicted", "retired"],
        "evidence_roles": ["supports", "refutes", "weakens", "historical_context"],
    },
    "shared_knowledge": {
        "visibility_values": ["local", "shared", "global", "shared_candidate"],
        "default_visibility": "local",
        "interface_tags": [
            "config",
            "register",
            "timing",
            "vsync",
            "frame",
            "clock",
            "reset",
            "protocol",
            "power",
            "boot",
            "data_path",
            "channel_count",
            "interrupt",
            "status",
            "memory_layout",
            "generated_header",
            "build",
            "test",
            "simulation",
            "palladium",
            "seed",
            "waveform",
            "coverage",
            "debug",
            "runbook",
        ],
        "high_level_node_types": ["flow", "feature", "concept", "interface", "runbook"],
        "cross_role_trigger_tags": [
            "config",
            "register",
            "timing",
            "vsync",
            "frame",
            "clock",
            "reset",
            "protocol",
            "power",
            "boot",
            "data_path",
            "channel_count",
            "interrupt",
            "status",
            "memory_layout",
            "generated_header",
            "build",
            "test",
            "simulation",
            "palladium",
            "seed",
            "waveform",
            "coverage",
            "debug",
            "runbook",
        ],
    },
    "source_handling": {
        "line_chunk_source_types": ["code_file", "rtl_file", "build_file", "log"],
        "file_backed_source_types": ["code_file", "rtl_file", "build_file", "doc_file"],
        "lines_per_chunk": 60,
        "line_chunk_overlap": 5,
        "paragraph_max_chars": 1600,
        "max_file_ingest_bytes": 1048576,
        "max_inline_content_bytes": 262144,
        "reject_inline_content_for_repo_files": True,
    },
    "taxonomy": {
        "node_types": [
            "block",
            "clock_domain",
            "reset_domain",
            "register",
            "field",
            "protocol_endpoint",
            "function",
            "file",
            "flow",
            "feature",
            "concept",
            "runbook",
            "interface",
            "test",
            "testbench",
            "build_target",
            "rtl_signal",
            "fw_symbol",
        ],
        "relation_types": [
            "affects",
            "configures",
            "implements",
            "clocked_by",
            "reset_by",
            "mapped_to",
            "validated_in",
            "belongs_to",
            "part_of",
            "configured_during",
            "debugged_by",
            "reads",
            "writes",
            "calls",
            "depends_on",
            "produces",
            "consumes",
            "verifies",
            "documents",
        ],
    },
    "retrieval": {
        "default_mode": "hybrid",
        "default_budget": "small",
        "include_stale_warnings": True,
        "max_extracted_terms": 12,
        "max_anchor_expansion": 20,
        "neighbor_edges_per_anchor": 8,
        "anchor_filter": {
            "enabled": True,
            "relative_delta": 25.0,
            "min_score": 0.0,
            "min_anchors": 4,
        },
        "modes": {
            "local": {"enabled": True},
            "global": {"enabled": True, "max_depth": 3},
            "bridge": {"enabled": True, "max_depth": 4, "max_paths": 6, "max_anchors": 10},
            "hybrid": {"enabled": True},
            "mix": {
                "enabled": True,
                "semantic_anchor_promotion": {
                    "enabled": False,
                    "top_semantic_results": 16,
                    "max_promoted_anchors": 4,
                    "min_lexical_anchors": 3,
                    "min_score": 0.0,
                    "relative_delta": 12.0,
                    "lexical_weight": 1.0,
                    "semantic_weight": 1.2,
                    "rrf_k": 60.0,
                    "require_graph_coherence": True,
                    "coherence_min_lexical_anchors": 2,
                    "coherence_max_depth": 2,
                    "coherence_max_depth_by_budget": {
                        "small": 2,
                        "medium": 2,
                        "large": 3,
                    },
                },
            },
        },
        "global_scope": {
            "preferred_node_types": ["flow", "feature", "concept", "interface", "runbook"],
            "upward_relations": ["belongs_to", "part_of", "depends_on", "affects", "configured_during"],
        },
        "bridge_scope": {
            "preferred_relations": ["belongs_to", "part_of", "configured_during", "configures", "affects", "depends_on", "implements", "writes", "reads", "calls"],
        },
        "budgets": {
            "small": {"nodes": 5, "claims": 4, "edges": 8, "evidence": 2},
            "medium": {"nodes": 10, "claims": 8, "edges": 16, "evidence": 3},
            "large": {"nodes": 20, "claims": 16, "edges": 32, "evidence": 5},
        },
        "ranking": {
            "exact_alias": 100.0,
            "exact_node": 95.0,
            "fts_alias": 60.0,
            "node_fts": 55.0,
            "edge_fts": 52.0,
            "resolved_node_bonus": 20.0,
            "claim_fts": 70.0,
            "chunk_fts": 45.0,
            "edge_expansion": 35.0,
            "claim_target_anchor_bonus": -3.0,
            "edge_endpoint_anchor_bonus": -4.0,
            "chunk_target_anchor_bonus": -6.0,
            "expanded_anchor_step_bonus": -8.0,
            "role_preferred_node_bonus": 8.0,
            "role_preferred_relation_bonus": 6.0,
            "role_audience_match_bonus": 12.0,
            "shared_visibility_bonus": 7.0,
            "global_visibility_bonus": 9.0,
            "shared_candidate_bonus": 3.0,
            "local_cross_role_penalty": -100.0,
            "global_frame_bonus": 18.0,
            "bridge_path_base": 80.0,
            "bridge_path_step_penalty": -8.0,
            "generic_node_degree_penalty": -2.0,
            "semantic_candidate": 55.0,
        },
    },
    "retrieval_models": {
        "embeddings": {
            "enabled": False,
            "provider": "sentence_transformers",
            "model": "BAAI/bge-large-en-v1.5",
            "preload_on_boot": True,
            "incremental_cache_enabled": True,
            "normalize": True,
            "query_instruction": "Represent this sentence for searching relevant passages: ",
            "max_in_memory_items": 200,
            "top_k": 12,
        },
        "reranker": {
            "enabled": False,
            "provider": "sentence_transformers_cross_encoder",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "preload_on_boot": False,
            "top_k_input": 40,
            "top_k_output": 12,
            "min_relevance_score": 0.0,
        },
        "llm_reranker": {
            "enabled": False,
            "provider": "http_chat",
            "base_url_env": "DOCGRAPH_LLM_BASE_URL",
            "api_key_env": "DOCGRAPH_LLM_TOKEN",
            "models_endpoint": "/v1/models",
            "chat_endpoint": "/v1/chat/completions",
            "preferred_models": [],
            "model": "",
            "model_cache_ttl_seconds": 300,
            "timeout_seconds": 45,
            "temperature": 0.0,
            "max_candidate_chars": 400,
            "top_k_input": 12,
            "top_k_output": 6,
            "min_relevance_score": 0.0,
            "tool_name": "submit_rerank_scores",
            "trigger": "rescue_only",
            "rescue": {
                "max_lexical_anchors": 2,
                "min_top_score": 0.0,
                "min_embedding_top_score": 0.0,
            },
        },
    },
    "roles": {
        "firmware": {
            "include_visibility": ["shared", "global", "shared_candidate"],
            "preferred_node_types": ["function", "file", "register", "field", "flow", "fw_symbol"],
            "preferred_relations": ["configures", "writes", "reads", "calls", "affects", "implements"],
            "suggested_checks": [
                "Inspect the current FW configuration/write path for selected anchors.",
                "Verify active mode/configuration before assuming RTL behavior.",
            ],
        },
        "rtl": {
            "include_visibility": ["shared", "global", "shared_candidate"],
            "preferred_node_types": ["block", "interface", "rtl_signal", "register", "field", "flow", "clock_domain", "reset_domain", "protocol_endpoint"],
            "preferred_relations": ["implements", "affects", "depends_on", "produces", "consumes"],
            "suggested_checks": [
                "Inspect producer/consumer RTL path around selected anchors.",
                "Check reset, clock, handshake, and interface assumptions.",
            ],
        },
        "build": {
            "include_visibility": ["shared", "global", "shared_candidate"],
            "preferred_node_types": ["build_target", "file", "function", "concept"],
            "preferred_relations": ["depends_on", "implements", "documents"],
            "suggested_checks": [
                "Inspect the active build target, flags, generated files, and dependency path.",
            ],
        },
        "test_debug": {
            "include_visibility": ["shared", "global", "shared_candidate"],
            "preferred_node_types": ["test", "testbench", "runbook", "function", "file", "concept"],
            "preferred_relations": ["verifies", "debugged_by", "depends_on", "calls"],
            "suggested_checks": [
                "Compare active evidence against current logs/test reproduction before concluding.",
            ],
        },
        "architecture": {
            "include_visibility": ["shared", "global", "shared_candidate"],
            "preferred_node_types": ["flow", "feature", "concept", "block", "interface", "protocol_endpoint", "runbook"],
            "preferred_relations": ["affects", "depends_on", "belongs_to", "documents"],
            "suggested_checks": [
                "Separate durable architecture facts from local implementation observations.",
            ],
        },
    },
    "intents": {
        "debug": {
            "suggested_checks": [
                "Verify selected anchors against current reproduction evidence before concluding.",
            ]
        },
        "implementation": {
            "suggested_checks": [
                "Verify ownership and existing implementation path before changing code.",
            ]
        },
        "architecture": {
            "suggested_checks": [
                "Check whether claims are facts, inferences, or open questions before treating them as architecture.",
            ]
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config requires PyYAML. Use .json config or install pyyaml.")
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping/object: {p}")
    return _deep_merge(DEFAULT_CONFIG, loaded)


def cfg_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_list(config: dict[str, Any], path: str) -> list[Any]:
    value = cfg_get(config, path, [])
    return value if isinstance(value, list) else []


def cfg_number(config: dict[str, Any], path: str, default: float) -> float:
    value = cfg_get(config, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
