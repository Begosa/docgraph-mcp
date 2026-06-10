#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docgraph_mcp.logging_utils import sanitize_for_log  # noqa: E402
from docgraph_mcp.models import (  # noqa: E402
    HttpChatRerankerProvider,
    build_rerank_tool_schema,
    parse_rerank_tool_call,
)


def main() -> None:
    backend = ROOT / "src" / "docgraph_mcp"
    union_isinstance_offenders = []
    typing_notrequired_offenders = []
    for path in backend.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "isinstance(" in line and " | " in line:
                union_isinstance_offenders.append(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")
            stripped = line.strip()
            if stripped.startswith("from typing import") and "NotRequired" in stripped:
                typing_notrequired_offenders.append(f"{path.relative_to(ROOT)}:{lineno}:{stripped}")
    assert not union_isinstance_offenders, union_isinstance_offenders
    assert not typing_notrequired_offenders, typing_notrequired_offenders

    assert sanitize_for_log(1, include_payloads=False, preview_chars=10) == 1
    assert sanitize_for_log(1.5, include_payloads=False, preview_chars=10) == 1.5
    assert sanitize_for_log(True, include_payloads=False, preview_chars=10) is True
    assert sanitize_for_log(None, include_payloads=False, preview_chars=10) is None
    assert build_rerank_tool_schema(2)["function"]["name"] == "submit_rerank_scores"
    assert parse_rerank_tool_call(
        {"tool_calls": [{"function": {"name": "submit_rerank_scores", "arguments": '{"scores":[1.0,0.0]}'}}]},
        2,
    ) == [1.0, 0.0]
    provider = HttpChatRerankerProvider({"base_url": "https://example.com"})
    body = provider._chat_request_body(model="demo-model", prompt="query and candidates", candidate_count=2)
    assert body["messages"][0] == {"role": "system", "content": ""}
    assert body["messages"][1] == {"role": "user", "content": "query and candidates"}
    assert "tools" in body and body["tool_choice"]["function"]["name"] == "submit_rerank_scores"
    print("PYTHON_COMPATIBILITY_OK")


if __name__ == "__main__":
    main()
