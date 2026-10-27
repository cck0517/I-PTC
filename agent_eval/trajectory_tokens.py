"""Trajectory-token counting.

The raw ``input_tokens`` recorded per turn counts every byte of re-sent context,
so a trajectory with N turns pays N × the conversation size. That overcounts
the actual work the model did — a property of the stateless chat API protocol,
not of the agent's strategy.

This module computes a fairer per-trajectory metric: each unique piece of
content (system prompt, tool descriptions, user question, each tool result,
each assistant-generated message) is tokenized exactly once. The sum is a
clean measure of the "information processed / produced" by the trajectory,
independent of turn count.

It is a *lower bound* on billed cost — real billing still depends on the
provider's prompt-cache hit rate on re-sent prefixes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import tiktoken

from agent_eval.agent import ToolDef, _tool_defs_to_openai


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _encoder_for(model: str):
    """Return the tiktoken encoder matching ``model``, falling back to o200k_base.

    tiktoken's ``encoding_for_model`` only knows about models that were in
    its table at install time; newer model families (gpt-5*, gpt-5.4*, etc.)
    raise ``KeyError`` even though they all use the o200k_base encoding.
    Route those families explicitly so we don't depend on tiktoken's fuzzy
    prefix matching catching up to the latest releases.
    """
    name = (model or "").lower()
    # Known o200k_base families (gpt-4o, gpt-4.1, gpt-5*, o-series reasoning models).
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


# ---------------------------------------------------------------------------
# Tool-description reconstruction (mode-dependent)
# ---------------------------------------------------------------------------

def _tool_schema_json(
    task: dict,
    mode: str,
    *,
    expose_all_apis: bool = False,
    tool_inspection: bool = False,
) -> str:
    """Reconstruct and serialize the tool schema the agent sent for this task.

    Matches the branching in ``run_eval._make_executor_and_tools``:
      - ``ptc_cell_encouraged``:       API tools + ptc_cell
      - ``ptc_encouraged``:            API tools + ptc_execute
      - ``tool_only`` / ``native_ptc``: API tools only

    When ``expose_all_apis`` is True (read from the trace's
    ``run_config.expose_all_apis``), the agent saw schemas for *every*
    supported API class regardless of what the task declared in
    ``involved_classes``. Reconstruct the same union here.

    When ``tool_inspection`` is True (read from ``run_config.tool_inspection``),
    the per-method API tool defs are *hidden* from
    the agent's tool list — the agent sees only the cell tool. They show
    up in the trace as ``tool_result`` content from ``<api>??`` /
    ``<api>.<method>??`` inspections, which are already tokenized via
    ``tj_tool_result_tokens``. Skip them here to avoid double-counting.
    """
    from agent_eval.ptc_tool import (
        get_ptc_cell_tool_def,
        get_ptc_tool_def,
        load_api_tool_defs,
    )
    from agent_eval.run_eval import _ALL_API_CLASSES

    if expose_all_apis:
        involved = list(_ALL_API_CLASSES)
    else:
        involved = task.get("involved_classes", [])

    tools: list[ToolDef] = []
    # Per-method API tool defs: only in the schema when inspection is OFF.
    # Under inspection mode they exist behind the inspector and surface as
    # cell stdout (counted in tj_tool_result_tokens), not as tools the
    # agent's chat schema carries.
    if not tool_inspection:
        tools.extend(load_api_tool_defs(involved))
    # The cell/execute tool itself is in the schema regardless of mode.
    if mode == "ptc_cell_encouraged":
        tools.append(get_ptc_cell_tool_def(involved))
    elif mode == "ptc_encouraged":
        tools.append(get_ptc_tool_def(involved))
    # else: tool_only / native_ptc → API tools only (handled above)

    openai_tools, _ = _tool_defs_to_openai(tools)
    return json.dumps(openai_tools)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_trajectory_tokens(
    trace: dict,
    task: dict | None = None,
    model: str | None = None,
) -> dict[str, int]:
    """Compute trajectory-based token metrics for a single trace.

    Unlike ``total_input_tokens`` (which sums per-turn raw counts and therefore
    multiplies the conversation prefix by N turns), these metrics tokenize each
    piece of unique content exactly once:

    Input side:
      - System prompt
      - Tool descriptions / schemas (fixed per task)
      - Each user message (the question and any multi-turn follow-ups)
      - Each tool-call result

    Output side:
      - Each assistant message (content + tool-call arguments)

    Returns a dict with ``trajectory_input_tokens``, ``trajectory_output_tokens``,
    and their sum. Also breaks out the input-side components for diagnostics.

    Reads ``run_config.expose_all_apis`` from the trace and reconstructs the
    full 10-API schema when it's True (matching what the agent actually saw,
    not just ``task['involved_classes']``).
    """
    model = model or trace.get("model") or "gpt-5-mini"
    rc = trace.get("run_config") or {}
    expose_all_apis = bool(rc.get("expose_all_apis", False))
    tool_inspection = bool(rc.get("tool_inspection", False))
    # Backward-compat: claude_ptc traces are conceptually native_ptc (API tools
    # only) regardless of the mode label stored on disk. Force the effective
    # mode so old traces with stale mode strings still reconstruct the right
    # tool schema.
    if rc.get("backend") == "claude_ptc":
        mode = "native_ptc"
    else:
        mode = rc.get("mode", "ptc_encouraged")

    # ---- Input side ----
    sp_tokens = _tok(trace.get("system_prompt") or "", model)

    # User messages: the first question plus any follow-up user turns in
    # multi-turn tasks, each counted once.
    turns = trace.get("turns", [])
    user_tokens = sum(
        _tok(t["content"], model)
        for t in turns
        if t.get("role") == "user" and t.get("content")
    )

    # Tool schema tokens — best-effort. Needs the task config to know which
    # API classes were loaded. If ``task`` isn't provided we skip this (0).
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

    # Tool-call results (one count per result, not per re-send)
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

    # ---- Output side ----
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
            # The tool name itself is also emitted by the model; count it.
            assistant_tool_call_tokens += _tok(tc.get("tool_name", ""), model)
            assistant_tool_call_tokens += _tok(serialized, model)

    output_tokens = assistant_content_tokens + assistant_tool_call_tokens

    return {
        "trajectory_input_tokens": input_tokens,
        "trajectory_output_tokens": output_tokens,
        "trajectory_total_tokens": input_tokens + output_tokens,
        # Diagnostic breakdown
        "tj_system_prompt_tokens": sp_tokens,
        "tj_tool_schema_tokens": tool_tokens,
        "tj_user_question_tokens": user_tokens,
        "tj_tool_result_tokens": tool_result_tokens,
        "tj_assistant_content_tokens": assistant_content_tokens,
        "tj_assistant_tool_call_tokens": assistant_tool_call_tokens,
    }
