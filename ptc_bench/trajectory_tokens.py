"""Count each trajectory message and tool schema once, excluding context resend overhead.

These metrics are a lower bound on billed tokens, which depend on provider caching."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import tiktoken

from iptc.agent import ToolDef, _tool_defs_to_openai


@lru_cache(maxsize=8)
def _encoder_for(model: str):
    """Select the model encoding with an o200k_base fallback.

    Route known model families explicitly when installed tiktoken tables lag releases."""
    name = (model or "").lower()
    if any(name.startswith(p) for p in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")):
        return tiktoken.get_encoding("o200k_base")
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _tok(text: str, model: str) -> int:
    """Count tokens in ``text`` using the encoder for ``model``."""
    if not text:
        return 0
    return len(_encoder_for(model).encode(text, disallowed_special=()))


def _tool_schema_json(
    task: dict,
    mode: str,
    *,
    expose_all_apis: bool = False,
    tool_inspection: bool = False,
) -> str:
    """Reconstruct the schema selected by run_eval._make_executor_and_tools.

    Honor expose_all_apis and exclude inspected methods: their documentation
    already appears in counted tool results."""
    from iptc.ptc_tool import (
        get_ptc_cell_tool_def,
        get_ptc_tool_def,
        load_api_tool_defs,
    )
    from ptc_bench.constants.executable_backend_config import ALL_API_CLASSES

    if expose_all_apis:
        involved = list(ALL_API_CLASSES)
    else:
        involved = task.get("involved_classes", [])

    tools: list[ToolDef] = []
    # Inspected method docs are counted in cell stdout, not the tool schema.
    if not tool_inspection:
        tools.extend(load_api_tool_defs(involved))
    if mode == "ptc_cell_encouraged":
        tools.append(get_ptc_cell_tool_def(involved))
    elif mode == "ptc_encouraged":
        tools.append(get_ptc_tool_def(involved))

    openai_tools, _ = _tool_defs_to_openai(tools)
    return json.dumps(openai_tools)


def compute_trajectory_tokens(
    trace: dict,
    task: dict | None = None,
    model: str | None = None,
) -> dict[str, int]:
    """Count trajectory input/output tokens once per message, schema, or result.

    Return totals plus input-component diagnostics. Reconstruct tool schemas
    from the trace configuration when task data is available."""
    model = model or trace.get("model") or "gpt-5-mini"
    rc = trace.get("run_config") or {}
    expose_all_apis = bool(rc.get("expose_all_apis", False))
    tool_inspection = bool(rc.get("tool_inspection", False))
    # Normalize stale mode labels in older Claude PTC traces.
    if rc.get("backend") == "claude_ptc":
        mode = "native_ptc"
    else:
        mode = rc.get("mode", "ptc_encouraged")

    sp_tokens = _tok(trace.get("system_prompt") or "", model)

    turns = trace.get("turns", [])
    user_tokens = sum(
        _tok(t["content"], model)
        for t in turns
        if t.get("role") == "user" and t.get("content")
    )

    # Task data is needed to reconstruct tool schemas; omit them when unavailable.
    if task is not None:
        try:
            tool_tokens = _tok(
                _tool_schema_json(
                    task, mode,
                    expose_all_apis=expose_all_apis,
                    tool_inspection=tool_inspection,
                ),
                model,
            )
        except Exception:
            tool_tokens = 0
    else:
        tool_tokens = 0

    # Count each result once, regardless of re-sends.
    tool_result_tokens = 0
    for t in turns:
        for tc in t.get("tool_calls", []) or []:
            r = tc.get("result")
            if r is None:
                continue
            if isinstance(r, (dict, list)):
                serialized = json.dumps(r, default=str)
            else:
                serialized = str(r)
            tool_result_tokens += _tok(serialized, model)

    input_tokens = sp_tokens + tool_tokens + user_tokens + tool_result_tokens

    assistant_content_tokens = 0
    assistant_tool_call_tokens = 0
    for t in turns:
        if t.get("role") != "assistant":
            continue
        if t.get("content"):
            assistant_content_tokens += _tok(t["content"], model)
        for tc in t.get("tool_calls", []) or []:
            args = tc.get("arguments")
            if args is None:
                continue
            if isinstance(args, (dict, list)):
                serialized = json.dumps(args, default=str)
            else:
                serialized = str(args)
            assistant_tool_call_tokens += _tok(tc.get("tool_name", ""), model)
            assistant_tool_call_tokens += _tok(serialized, model)

    output_tokens = assistant_content_tokens + assistant_tool_call_tokens

    return {
        "trajectory_input_tokens": input_tokens,
        "trajectory_output_tokens": output_tokens,
        "trajectory_total_tokens": input_tokens + output_tokens,
        "tj_system_prompt_tokens": sp_tokens,
        "tj_tool_schema_tokens": tool_tokens,
        "tj_user_question_tokens": user_tokens,
        "tj_tool_result_tokens": tool_result_tokens,
        "tj_assistant_content_tokens": assistant_content_tokens,
        "tj_assistant_tool_call_tokens": assistant_tool_call_tokens,
    }
