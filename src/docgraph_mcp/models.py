from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        ...


class RerankerProvider(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]:
        ...


@dataclass(frozen=True)
class ProviderState:
    enabled: bool
    available: bool
    reason: str | None = None


class SentenceTransformersEmbeddingProvider:
    def __init__(self, model_name: str, *, normalize: bool = True, query_instruction: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.model = SentenceTransformer(model_name)
        self.state = ProviderState(True, True)

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prepared = [self.query_instruction + t if is_query and self.query_instruction else t for t in texts]
        vectors = self.model.encode(prepared, normalize_embeddings=self.normalize)
        return [list(map(float, v)) for v in vectors]


class SentenceTransformersCrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        self.state = ProviderState(True, True)

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c) for c in candidates])
        return [float(s) for s in scores]


class HttpEmbeddingProvider:
    def __init__(self, *, endpoint: str, model: str, api_key_env: str | None = None) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.state = ProviderState(True, True)

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:  # noqa: ARG002
        import json
        import os
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - endpoint is explicit config
            payload = json.loads(resp.read().decode("utf-8"))
        return [list(map(float, item["embedding"])) for item in payload.get("data", [])]


def select_first_available_model(available: list[str], preferred: list[str]) -> str | None:
    """Pick the first preferred model id present in an availability list."""
    available_set = set(available)
    for model_id in preferred:
        if model_id in available_set:
            return model_id
    return None


RERANK_SCORES_TOOL_NAME = "submit_rerank_scores"


def build_rerank_tool_schema(candidate_count: int, *, tool_name: str = RERANK_SCORES_TOOL_NAME) -> dict[str, Any]:
    """Return an OpenAI-compatible tool schema for structured rerank scores."""
    count = max(1, int(candidate_count))
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Submit one relevance score per candidate, in the same order as provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "description": "Relevance scores from 0.0 (unrelated) to 1.0 (highly related).",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": count,
                        "maxItems": count,
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
        },
    }


def parse_rerank_scores_payload(payload: Any, expected: int) -> Optional[list[float]]:
    """Normalize a structured rerank payload into a fixed-length score list."""
    if expected <= 0:
        return []
    scores_raw: Any = None
    if isinstance(payload, dict):
        scores_raw = payload.get("scores")
        if scores_raw is None:
            scores_raw = payload.get("rankings")
    elif isinstance(payload, list):
        scores_raw = payload
    if not isinstance(scores_raw, list) or len(scores_raw) != expected:
        return None
    out: list[float] = []
    for item in scores_raw:
        if isinstance(item, dict):
            value = item.get("score", item.get("relevance"))
        else:
            value = item
        out.append(float(value))
    return out


def parse_rerank_tool_call(
    message: Optional[dict[str, Any]],
    expected: int,
    *,
    tool_name: str = RERANK_SCORES_TOOL_NAME,
) -> Optional[list[float]]:
    """Parse scores from a forced function/tool call response."""
    if not isinstance(message, dict):
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        if str(fn.get("name") or "") != tool_name:
            continue
        args_text = fn.get("arguments")
        if not isinstance(args_text, str):
            continue
        try:
            payload = json.loads(args_text)
        except json.JSONDecodeError:
            continue
        parsed = parse_rerank_scores_payload(payload, expected)
        if parsed is not None:
            return parsed
    return None


class HttpChatRerankerProvider:
    """OpenAI-compatible chat reranker with dynamic model selection."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.base_url = self._resolve_base_url(cfg)
        self.api_key_env = cfg.get("api_key_env")
        self.models_endpoint = str(cfg.get("models_endpoint", "/v1/models"))
        self.chat_endpoint = str(cfg.get("chat_endpoint", "/v1/chat/completions"))
        self.preferred_models = [str(item) for item in (cfg.get("preferred_models") or []) if str(item).strip()]
        self.fixed_model = str(cfg.get("model") or "").strip()
        self.timeout_seconds = max(5, int(cfg.get("timeout_seconds", 45)))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_candidate_chars = max(80, int(cfg.get("max_candidate_chars", 400)))
        self.model_cache_ttl_seconds = max(30, int(cfg.get("model_cache_ttl_seconds", 300)))
        self.tool_name = str(cfg.get("tool_name", RERANK_SCORES_TOOL_NAME)).strip() or RERANK_SCORES_TOOL_NAME
        self._model_cache: Optional[tuple[str, float]] = None
        if not self.base_url:
            self.state = ProviderState(True, False, "llm reranker base URL is not configured")
        else:
            self.state = ProviderState(True, True)

    def _resolve_base_url(self, cfg: dict[str, Any]) -> str:
        direct = str(cfg.get("base_url") or "").strip()
        if direct:
            return direct.rstrip("/")
        env_name = str(cfg.get("base_url_env") or "").strip()
        if env_name:
            env_value = str(os.environ.get(env_name) or "").strip()
            if env_value:
                return env_value.rstrip("/")
        return ""

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            token = str(os.environ.get(self.api_key_env) or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _join_url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        base = self.base_url.rstrip("/")
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{base}{path}"

    def _http_json(self, *, url: str, method: str = "GET", body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._auth_headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"llm reranker HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llm reranker request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("llm reranker returned non-object JSON")
        return payload

    def fetch_available_models(self) -> list[str]:
        payload = self._http_json(url=self._join_url(self.models_endpoint), method="GET")
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        out: list[str] = []
        for item in data:
            if isinstance(item, dict):
                model_id = str(item.get("id") or "").strip()
                if model_id:
                    out.append(model_id)
        return out

    def resolve_model(self) -> Optional[str]:
        if self.fixed_model:
            return self.fixed_model
        now = time.time()
        if self._model_cache and self._model_cache[1] > now:
            return self._model_cache[0]
        if not self.preferred_models:
            return None
        available = self.fetch_available_models()
        chosen = select_first_available_model(available, self.preferred_models)
        if chosen:
            self._model_cache = (chosen, now + float(self.model_cache_ttl_seconds))
        return chosen

    def _build_prompt(self, query: str, candidates: list[str]) -> str:
        lines = [
            "You are a relevance judge for architecture documentation retrieval.",
            "Score each candidate from 0.0 (unrelated) to 1.0 (highly related) for the query.",
            f"Call the `{self.tool_name}` tool with one score per candidate in order.",
            "",
            f"Query: {query.strip()}",
            "",
            "Candidates:",
        ]
        for idx, candidate in enumerate(candidates):
            text = candidate.replace("\n", " ").strip()
            if len(text) > self.max_candidate_chars:
                text = text[: self.max_candidate_chars - 3] + "..."
            lines.append(f"{idx}: {text}")
        return "\n".join(lines)

    def _chat_request_body(self, *, model: str, prompt: str, candidate_count: int) -> dict[str, Any]:
        return {
            "model": model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "tools": [build_rerank_tool_schema(candidate_count, tool_name=self.tool_name)],
            "tool_choice": {"type": "function", "function": {"name": self.tool_name}},
        }

    def _parse_chat_response(self, payload: dict[str, Any], expected: int) -> list[float]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llm reranker returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("llm reranker returned no message")
        parsed = parse_rerank_tool_call(message, expected, tool_name=self.tool_name)
        if parsed is not None:
            return parsed
        raise RuntimeError(f"llm reranker did not return required tool call `{self.tool_name}`")

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        model = self.resolve_model()
        if not model:
            raise RuntimeError("no llm reranker model is configured or available")
        prompt = self._build_prompt(query, candidates)
        payload = self._http_json(
            url=self._join_url(self.chat_endpoint),
            method="POST",
            body=self._chat_request_body(model=model, prompt=prompt, candidate_count=len(candidates)),
        )
        return self._parse_chat_response(payload, len(candidates))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0.0 or rn == 0.0:
        return 0.0
    return dot / (ln * rn)


def make_embedding_provider(config: dict[str, Any]) -> tuple[EmbeddingProvider | None, ProviderState]:
    cfg = config.get("retrieval_models", {}).get("embeddings", {})
    if not cfg.get("enabled", False):
        return None, ProviderState(False, False, "embeddings disabled")
    provider = cfg.get("provider", "sentence_transformers")
    model = cfg.get("model")
    try:
        if provider == "sentence_transformers":
            if not model:
                return None, ProviderState(True, False, "embedding model is not configured")
            p = SentenceTransformersEmbeddingProvider(
                str(model),
                normalize=bool(cfg.get("normalize", True)),
                query_instruction=str(cfg.get("query_instruction", "")),
            )
            return p, p.state
        if provider == "http":
            endpoint = cfg.get("endpoint")
            if not endpoint or not model:
                return None, ProviderState(True, False, "http embedding endpoint/model is not configured")
            p = HttpEmbeddingProvider(endpoint=str(endpoint), model=str(model), api_key_env=cfg.get("api_key_env"))
            return p, p.state
        return None, ProviderState(True, False, f"unsupported embedding provider: {provider}")
    except RuntimeError as exc:
        return None, ProviderState(True, False, str(exc))


def make_reranker_provider(config: dict[str, Any]) -> tuple[RerankerProvider | None, ProviderState]:
    cfg = config.get("retrieval_models", {}).get("reranker", {})
    if not cfg.get("enabled", False):
        return None, ProviderState(False, False, "reranker disabled")
    provider = cfg.get("provider", "sentence_transformers_cross_encoder")
    model = cfg.get("model")
    try:
        if provider == "sentence_transformers_cross_encoder":
            if not model:
                return None, ProviderState(True, False, "reranker model is not configured")
            p = SentenceTransformersCrossEncoderReranker(str(model))
            return p, p.state
        return None, ProviderState(True, False, f"unsupported reranker provider: {provider}")
    except RuntimeError as exc:
        return None, ProviderState(True, False, str(exc))


def make_llm_reranker_provider(config: dict[str, Any]) -> tuple[RerankerProvider | None, ProviderState]:
    cfg = config.get("retrieval_models", {}).get("llm_reranker", {})
    if not cfg.get("enabled", False):
        return None, ProviderState(False, False, "llm reranker disabled")
    provider = cfg.get("provider", "http_chat")
    if provider != "http_chat":
        return None, ProviderState(True, False, f"unsupported llm reranker provider: {provider}")
    try:
        llm = HttpChatRerankerProvider(cfg)
        return llm, llm.state
    except Exception as exc:  # pragma: no cover - defensive provider bootstrap
        return None, ProviderState(True, False, str(exc))
