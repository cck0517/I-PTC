#!/usr/bin/env python3
"""Run PTC-Bench tasks and save structured agent traces. See --help for options."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from iptc.agent import Agent, ToolDef
from iptc.doc_learner import DocLearner
from iptc.ptc_tool import (
    PROMPT_MODES,
    PTCExecutor,
    _build_inspect_fn,
    build_system_prompt,
    get_ptc_cell_tool_def,
    get_ptc_tool_def,
    load_api_tool_defs,
)
from ptc_bench.constants.executable_backend_config import (
    ALL_API_CLASSES as _ALL_API_CLASSES,
    CLASS_FILE_PATH_MAPPING,
)
from ptc_bench.utils import load_dataset_entry


def _instantiate_api(
    class_name: str,
    scenario: dict,
    flakiness_config: dict | None = None,
) -> Any:
    """Load an API scenario and optional per-base-class transient-error thresholds."""
    entry = CLASS_FILE_PATH_MAPPING[class_name]
    # Tuple mappings let alt config keys reuse the base class attribute name.
    if isinstance(entry, tuple):
        module_path, class_attr = entry
    else:
        module_path, class_attr = entry, class_name
    module = importlib.import_module(module_path)
    cls = getattr(module, class_attr)
    api = cls()
    if flakiness_config is not None:
        spec = flakiness_config.get(class_attr)
        if spec is None and class_name != class_attr:
            spec = flakiness_config.get(class_name)
        if spec is not None:
            api._flaky_thresholds = spec
    api._load_scenario(scenario)
    return api


# Native backends use provider code tools and ignore the mode flag.
_NATIVE_BACKENDS = ("claude_ptc", "openai_native", "gemini_native")

_MODE_ALIASES = {"ptc": "ptc_encouraged", "raw_iptc": "ptc_cell_encouraged", "iptc": "ptc_cell_encouraged"}


def _make_executor_and_tools(
    task: dict,
    mode: str,
    minimal_print: bool = False,
    tool_inspection: bool = False,
    expose_all_apis: bool = False,
    record_metrics: bool = False,
) -> tuple[PTCExecutor, list[ToolDef], list[str], list[ToolDef] | None]:
    """Build task APIs and return (executor, tools, involved_classes, inspector_defs).

    Mode selects stateless code, persistent cells, or direct tools. Inspection
    hides API methods from tools and exposes them through inspector_defs;
    callers must apply learned addenda to both lists. expose_all_apis adds APIs
    with empty scenarios without changing the task data."""
    task_involved = task["involved_classes"]
    config = task["initial_config"]

    if expose_all_apis:
        effective_classes = list(_ALL_API_CLASSES)
        # Replace base classes with task-declared Alt classes so scenario data and flaky overrides load.
        for declared in task_involved:
            if declared.endswith("Alt") and declared not in effective_classes:
                base = declared[: -len("Alt")]
                if base in effective_classes:
                    idx = effective_classes.index(base)
                    effective_classes[idx] = declared
                else:
                    effective_classes.append(declared)
    else:
        effective_classes = list(task_involved)

    enterprise_calendar_api = None
    org_chart_api = None
    yelp_api = None
    maps_api = None
    uber_api = None
    grocery_api = None
    personal_calendar_api = None
    doordash_api = None
    spotify_api = None
    venmo_api = None

    flakiness = task.get("flakiness_config")
    for cls_name in effective_classes:
        api = _instantiate_api(
            cls_name,
            config.get(cls_name, {}),
            flakiness_config=flakiness,
        )
        if cls_name in ("EnterpriseCalendarAPI", "EnterpriseCalendarAPIAlt"):
            enterprise_calendar_api = api
        elif cls_name in ("OrgChartAPI", "OrgChartAPIAlt"):
            org_chart_api = api
        elif cls_name in ("YelpAPI", "YelpAPIAlt"):
            yelp_api = api
        elif cls_name == "GoogleMapsAPI":
            maps_api = api
        elif cls_name in ("UberAPI", "UberAPIAlt"):
            uber_api = api
        elif cls_name in ("GroceryDeliveryAPI", "GroceryDeliveryAPIAlt"):
            grocery_api = api
        elif cls_name == "PersonalCalendarAPI":
            personal_calendar_api = api
        elif cls_name in ("DoorDashAPI", "DoorDashAPIAlt"):
            doordash_api = api
        elif cls_name in ("SpotifyAPI", "SpotifyAPIAlt"):
            spotify_api = api
        elif cls_name in ("VenmoAPI", "VenmoAPIAlt"):
            venmo_api = api

    executor = PTCExecutor(
        enterprise_calendar_api=enterprise_calendar_api,
        org_chart_api=org_chart_api,
        yelp_api=yelp_api,
        maps_api=maps_api,
        uber_api=uber_api,
        grocery_api=grocery_api,
        personal_calendar_api=personal_calendar_api,
        doordash_api=doordash_api,
        spotify_api=spotify_api,
        venmo_api=venmo_api,
        record_metrics=record_metrics,
    )

    render_classes = list(effective_classes)

    # Load method definitions even when only the inspector exposes them.
    api_tool_defs = load_api_tool_defs(effective_classes)

    # inspector_defs holds methods hidden from the agent's tool schema.
    inspector_defs: list[ToolDef] | None = None
    if mode == "ptc_cell_encouraged":
        cell_tool = get_ptc_cell_tool_def(
            render_classes,
            minimal_print=minimal_print,
            tool_inspection=tool_inspection,
        )
        if tool_inspection:
            all_tool_defs = [cell_tool]
            inspector_defs = list(api_tool_defs)
            executor.set_inspector(
                _build_inspect_fn(api_tool_defs, render_classes)
            )
        else:
            all_tool_defs = api_tool_defs + [cell_tool]
    elif mode == "ptc_encouraged":
        execute_tool = get_ptc_tool_def(
            render_classes,
            minimal_print=minimal_print,
            tool_inspection=tool_inspection,
        )
        if tool_inspection:
            # Stateless inspection uses the same discovery output without namespace persistence.
            all_tool_defs = [execute_tool]
            inspector_defs = list(api_tool_defs)
            executor.set_inspector(
                _build_inspect_fn(api_tool_defs, render_classes)
            )
        else:
            all_tool_defs = api_tool_defs + [execute_tool]
    else:  # tool_only
        all_tool_defs = api_tool_defs

    return executor, all_tool_defs, render_classes, inspector_defs


def _serialize_api_state(executor: PTCExecutor) -> dict[str, Any]:
    """Snapshot the current in-memory state of all API objects."""
    state: dict[str, Any] = {}
    if executor.enterprise_calendar_api is not None:
        api = executor.enterprise_calendar_api
        state["EnterpriseCalendarAPI"] = {
            "metadata": dict(api.metadata),
            "employees": list(api.employees),
            "events": list(api.events),
        }
    if executor.org_chart_api is not None:
        api = executor.org_chart_api
        state["OrgChartAPI"] = {
            "employees": list(api.employees),
            "reporting_history": list(api.reporting_history),
        }
    if executor.yelp_api is not None:
        api = executor.yelp_api
        state["YelpAPI"] = {
            "businesses": dict(api.businesses),
            "reviews": dict(api.reviews),
            "saved_places": dict(api.saved_places),
            "bookmarks": list(api.bookmarks),
        }
    if executor.maps_api is not None:
        api = executor.maps_api
        state["GoogleMapsAPI"] = {
            "geocode_cache": dict(api.geocode_cache),
        }
    if executor.uber_api is not None:
        api = executor.uber_api
        state["UberAPI"] = {
            "pricing": dict(api.pricing),
            "rides": dict(api.rides),
            "current_ride": api.current_ride,
            "ride_counter": api.ride_counter,
        }
    if executor.grocery_api is not None:
        api = executor.grocery_api
        state["GroceryDeliveryAPI"] = {
            "shopping_list": list(api.shopping_list),
            "stores": dict(api.stores),
            "gift_cards": dict(api.gift_cards),
        }
    if executor.personal_calendar_api is not None:
        api = executor.personal_calendar_api
        state["PersonalCalendarAPI"] = {
            "events": dict(api.events),
            "current_date": api.current_date,
            "event_counter": api.event_counter,
        }
    if executor.doordash_api is not None:
        api = executor.doordash_api
        state["DoorDashAPI"] = {
            "restaurants": dict(api.restaurants),
            "menus": dict(api.menus),
            "orders": list(api.orders),
            "order_counter": api.order_counter,
        }
    if executor.spotify_api is not None:
        api = executor.spotify_api
        state["SpotifyAPI"] = {
            "playlists": dict(api.playlists),
            "tracks": dict(api.tracks),
            "recently_played": list(api.recently_played),
            "saved_playlists": list(api.saved_playlists),
        }
    if executor.venmo_api is not None:
        api = executor.venmo_api
        state["VenmoAPI"] = {
            "balance": api.balance,
            "friends": list(api.friends),
            "transactions": list(api.transactions),
            "pending_requests": list(api.pending_requests),
        }
    return state


def _dispatch_api_call(
    executor: PTCExecutor, tool_name: str, args: dict[str, Any]
) -> Any:
    """
    Dispatch a direct API method tool call (e.g. ``enterprise_calendar_api.book_event``).

    The tool name is ``<api_prefix>.<method_name>``.
    """
    api_prefix, method_name = tool_name.split(".", 1)

    api_map: dict[str, Any] = {}
    if executor.enterprise_calendar_api is not None:
        api_map["enterprise_calendar_api"] = executor.enterprise_calendar_api
    if executor.org_chart_api is not None:
        api_map["org_chart_api"] = executor.org_chart_api
    if executor.yelp_api is not None:
        api_map["yelp_api"] = executor.yelp_api
    if executor.maps_api is not None:
        api_map["maps_api"] = executor.maps_api
    if executor.uber_api is not None:
        api_map["uber_api"] = executor.uber_api
    if executor.grocery_api is not None:
        api_map["grocery_api"] = executor.grocery_api
    if executor.personal_calendar_api is not None:
        api_map["personal_calendar_api"] = executor.personal_calendar_api
    if executor.doordash_api is not None:
        api_map["doordash_api"] = executor.doordash_api
    if executor.spotify_api is not None:
        api_map["spotify_api"] = executor.spotify_api
    if executor.venmo_api is not None:
        api_map["venmo_api"] = executor.venmo_api

    api_obj = api_map.get(api_prefix)
    if api_obj is None:
        return {"error": f"Unknown API: {api_prefix}"}

    method = getattr(api_obj, method_name, None)
    if method is None:
        return {"error": f"Unknown method: {method_name} on {api_prefix}"}

    try:
        return method(**args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_single_task(
    task: dict,
    backend: str,
    model: str,
    mode: str,
    max_turns: int,
    temperature: float,
    max_token_budget: int | None = None,
    multi_turn: bool = False,
    minimal_print: bool = False,
    doc_learner: DocLearner | None = None,
    tool_inspection: bool = False,
    expose_all_apis: bool = True,
    record_runtime_metrics: bool = False,
) -> dict:
    """Run one task and return its trace.

    Keep expose_all_apis aligned with the CLI default so subprocess and
    in-thread runs expose the same tools."""
    task_id = task["id"]
    question_turns = task["question"]
    user_message = question_turns[0][0]["content"]

    extra_user_messages: list[str] | None = None
    if multi_turn and len(question_turns) > 1:
        extra_user_messages = [t[0]["content"] for t in question_turns[1:]]

    _t_init = time.perf_counter() if record_runtime_metrics else None
    executor, tool_defs, involved, inspector_defs = _make_executor_and_tools(
        task, mode,
        minimal_print=minimal_print,
        tool_inspection=tool_inspection,
        expose_all_apis=expose_all_apis,
        record_metrics=record_runtime_metrics,
    )
    executor_init_time_s = (
        round(time.perf_counter() - _t_init, 6) if _t_init is not None else None
    )

    if doc_learner is not None:
        doc_learner.apply_improvements(tool_defs)
        # Inspection-only definitions need learned addenda too.
        if inspector_defs is not None:
            doc_learner.apply_improvements(inspector_defs)

    # Claude PTC prompts name its native server tool, not the benchmark's code tools.
    if backend in _NATIVE_BACKENDS:
        system_prompt = build_system_prompt(involved, mode="native_ptc")
    else:
        system_prompt = build_system_prompt(involved, mode=mode)

    def tool_callback(name: str, args: dict[str, Any]) -> Any:
        if name == "ptc_execute":
            return executor.execute(args.get("code", ""))
        if name == "ptc_cell":
            return executor.execute_cell(args.get("code", ""))
        if "." in name:
            return _dispatch_api_call(executor, name, args)
        return {"error": f"Unknown tool: {name}"}

    agent = Agent(
        backend=backend,
        model=model,
        tool_defs=tool_defs,
        tool_executor=tool_callback,
        system_prompt=system_prompt,
        max_turns=max_turns,
        temperature=temperature,
        max_token_budget=max_token_budget,
    )

    print(f"  Running task: {task_id}")
    trace = agent.run(user_message=user_message, task_id=task_id,
                       extra_user_messages=extra_user_messages)
    trace_dict = trace.to_dict()
    trace_dict["final_state"] = _serialize_api_state(executor)
    from ptc_bench.eval_checker.multi_turn_eval.api_method_types import (
        compute_trace_api_type_distribution,
    )
    trace_dict["api_type_distribution"] = compute_trace_api_type_distribution(
        trace_dict
    )
    # Persist the effective native mode for downstream analysis.
    metadata_mode = "native_ptc" if backend in _NATIVE_BACKENDS else mode
    trace_dict["run_config"] = {
        "backend": backend,
        "model": model,
        "mode": metadata_mode,
        "max_turns": max_turns,
        "max_token_budget": max_token_budget,
        "temperature": temperature,
        "multi_turn": multi_turn,
        "minimal_print": minimal_print,
        "tool_inspection": tool_inspection,
        "expose_all_apis": expose_all_apis,
    }
    # Metrics stay in a side channel and are attached only when enabled.
    if record_runtime_metrics:
        trace_dict["run_config"]["record_runtime_metrics"] = True
        rm = executor.runtime_metrics_summary()
        rm["executor_init_time_s"] = executor_init_time_s
        llm_time = sum(
            t.get("latency_s") or 0.0 for t in trace_dict.get("turns", [])
            if t.get("role") == "assistant"
        )
        rm["total_llm_latency_s"] = round(llm_time, 3)
        trace_dict["runtime_metrics"] = rm
    return trace_dict


def _build_timeout_trace(
    task_id: str,
    backend: str,
    model: str,
    mode: str,
    max_turns: int,
    max_token_budget: int | None,
    temperature: float | None,
    multi_turn: bool,
    minimal_print: bool,
    timeout_s: float,
    elapsed_s: float | None = None,
) -> dict:
    """Build a synthetic trace dict for a task that exceeded its wall-clock budget."""
    metadata_mode = "native_ptc" if backend in _NATIVE_BACKENDS else mode
    wall = round(elapsed_s if elapsed_s is not None else float(timeout_s), 2)
    return {
        "task_id": task_id,
        "model": model,
        "system_prompt": "",
        "turns": [],
        "final_answer": None,
        "stop_reason": "time_exceeded",
        "total_time_s": wall,
        "wall_time_s": wall,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "error": f"Task exceeded wall-clock timeout of {int(timeout_s)}s",
        "run_config": {
            "backend": backend,
            "model": model,
            "mode": metadata_mode,
            "max_turns": max_turns,
            "max_token_budget": max_token_budget,
            "temperature": temperature,
            "multi_turn": multi_turn,
            "minimal_print": minimal_print,
        },
    }


def _summary_row_from_trace(trace: dict, task_id: str | None = None) -> dict:
    """Build a summary row. top_level_tool_calls excludes calls inside cells;
    use eval_results.json num_api_calls for the full API call count."""
    turns = trace.get("turns", [])
    n_tool_calls = sum(len(t.get("tool_calls", [])) for t in turns)
    n_llm_turns = sum(1 for t in turns if t.get("role") == "assistant")
    n_ptc_calls = sum(
        1 for t in turns for tc in t.get("tool_calls", [])
        if tc.get("tool_name") == "ptc_execute"
    )
    n_ptc_cell_calls = sum(
        1 for t in turns for tc in t.get("tool_calls", [])
        if tc.get("tool_name") == "ptc_cell"
    )
    n_top_level_tool_calls = n_tool_calls - n_ptc_calls - n_ptc_cell_calls
    total_in_tok = trace.get("total_input_tokens", 0)
    total_out_tok = trace.get("total_output_tokens", 0)
    row = {
        "task_id": task_id or trace.get("task_id"),
        "stop_reason": trace.get("stop_reason"),
        "tool_calls": n_tool_calls,
        "ptc_execute_calls": n_ptc_calls,
        "ptc_cell_calls": n_ptc_cell_calls,
        "top_level_tool_calls": n_top_level_tool_calls,
        "llm_turns": n_llm_turns,
        "has_final_answer": trace.get("final_answer") is not None,
        "wall_time_s": trace.get("wall_time_s", 0),
        "input_tokens": total_in_tok,
        "output_tokens": total_out_tok,
        "total_tokens": total_in_tok + total_out_tok,
        "error": trace.get("error"),
    }
    rm = trace.get("runtime_metrics")
    if rm:
        row.update({
            "total_exec_time_s": rm.get("total_exec_time_s"),
            "total_setup_time_s": rm.get("total_setup_time_s"),
            "total_llm_latency_s": rm.get("total_llm_latency_s"),
            "peak_namespace_bytes": rm.get("peak_namespace_bytes"),
            "final_namespace_vars": rm.get("final_namespace_vars"),
            "n_code_invocations": rm.get("n_code_invocations"),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PTC-Bench agent evaluation.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "gemini", "claude", "claude_ptc", "qwen", "glm", "kimi", "openrouter", "openai_native", "gemini_native"],
        required=True,
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (e.g. gpt-4o, gemini-2.0-flash, claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODE_ALIASES) + list(PROMPT_MODES),
        default="ptc",
        help=(
            "Interface (paper names). 'tool_only': API tools only, no code "
            "execution. 'ptc': stateless ptc_execute tool. 'raw_iptc': "
            "persistent ptc_cell, without inspection or doc learning. "
            "'iptc': full I-PTC = ptc_cell + --tool-inspection + doc "
            "learning (--learning-batch-size 10, --learner-model = --model "
            "unless given). Legacy values ptc_encouraged (= ptc) and "
            "ptc_cell_encouraged (= raw_iptc) are still accepted. Ignored "
            "when --backend=claude_ptc (native PTC). Default: ptc."
        ),
    )
    parser.add_argument(
        "--queries-file", default=None,
        help=(
            "Path to a custom JSONL queries file. If omitted, uses the "
            "default ptc_bench/data/main.jsonl. Useful for running ablations with "
            "modified initial configs."
        ),
    )
    parser.add_argument(
        "--task-id", default=None,
        help="Run a single task by ID. If omitted, runs all tasks.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for traces. Defaults to experiments/traces/<model>_<mode>_<timestamp>/.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=30,
        help="Maximum agent turns per task (default: 30).",
    )
    parser.add_argument(
        "--max-token-budget", type=int, default=0,
        help=(
            "Maximum total tokens (input+output) per task before the agent "
            "is stopped. Default: 0 (disabled). Set to a positive integer "
            "(e.g. 200000) to enable a per-task ceiling — useful for "
            "capping runaway cost on tool_only-style workflows."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=(
            "Sampling temperature. Default: 0.0 for most models. "
            "Omit or leave as default for reasoning models (o-series, "
            "gpt-5-mini) which do not support this parameter."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel worker threads for running tasks. "
            "Default: 1 (sequential). Higher values run multiple tasks "
            "concurrently, which is useful for I/O-bound API calls."
        ),
    )
    parser.add_argument(
        "--multi-turn", action="store_true", default=False,
        help=(
            "Enable multi-turn mode. When set, the pipeline sends all "
            "user turns from the query sequentially, checking correctness "
            "after each turn."
        ),
    )
    parser.add_argument(
        "--minimal-print", action="store_true", default=False,
        help=(
            "Side experiment: append instructions to the tool descriptions "
            "asking the model to minimize print output in code execution. "
            "Applies to all modes. Does not change any other pipeline behavior."
        ),
    )
    parser.add_argument(
        "--tool-inspection", action="store_true", default=False,
        help=(
            "Any code-execution mode (--mode ptc or raw_iptc; implied by "
            "iptc), not tool_only. Remove the per-method API tool defs from "
            "the tool list; the agent sees only the code tool (ptc_execute "
            "or ptc_cell) and must discover methods via ``<api>??`` / "
            "``<api>.<method>??`` inspection syntax inside the code it "
            "writes. Multiple inspections per snippet are supported. In "
            "ptc mode the inspector is installed on every fresh namespace, "
            "so discovery works there too, but what is learned survives "
            "only in the conversation, not in variables."
        ),
    )
    parser.add_argument(
        "--expose-all-apis", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Instantiate and expose ALL supported APIs (not just those in "
            "the task's ``involved_classes``). APIs the task didn't "
            "configure are loaded with empty/default state — the "
            "benchmark itself is unchanged, only the tool catalog "
            "visible to the agent widens. Default: ON, so the value of "
            "tool inspection (lazy method discovery) is visible against a "
            "realistic 10-API tool surface. Pass --no-expose-all-apis to "
            "scope APIs to the task's ``involved_classes`` instead."
        ),
    )
    parser.add_argument(
        "--record-runtime-metrics", action="store_true", default=False,
        help=(
            "Record per-invocation code-"
            "execution wall time, namespace size (vars + deep-size bytes), "
            "and process RSS into a ``runtime_metrics`` block on each trace "
            "(plus summary fields). Metrics are captured in a side channel; "
            "tool results shown to the model are untouched. Default OFF: "
            "behavior and trace schema are byte-identical to legacy runs."
        ),
    )
    parser.add_argument(
        "--learning-batch-size", type=int, default=None,
        help=(
            "Enable LLM-based doc improvement between task batches. Tasks "
            "are split into batches of this size; after each batch, errors "
            "are collected and an LLM generates improved tool descriptions "
            "for subsequent batches. 0 = disabled (default; 10 with --mode iptc). 1 = update "
            "after every task (sequential). Set to --workers value for "
            "natural parallelism boundaries."
        ),
    )
    parser.add_argument(
        "--learning-max-patterns", type=int, default=None,
        help=(
            "Maximum number of new error patterns the doc learner processes "
            "after each batch (one LLM call each). When a batch has more, "
            "the most frequent are kept; the rest are not marked as seen and "
            "can still be learned later if they recur. Default: no cap."
        ),
    )
    parser.add_argument(
        "--learner-model", type=str, default=None,
        help=(
            "Model to use for the doc-improvement LLM calls when "
            "--learning-batch-size is enabled. Default: --model with --mode iptc, else gpt-5-mini."
        ),
    )
    parser.add_argument(
        "--learner-backend", default=None,
        choices=["openai", "claude", "gemini", "qwen", "glm", "kimi", "openrouter"],
        help=(
            "Provider for the doc-learner calls. Default: the run's own "
            "--backend when --learner-model equals --model; otherwise chosen "
            "from the learner model's name (claude-* -> claude, gemini-* -> "
            "gemini, qwen*/glm*/kimi*/moonshot* -> those providers, anything "
            "else -> openai)."
        ),
    )
    parser.add_argument(
        "--prompt-overrides", type=str, default=None,
        help=(
            "Path to a JSON file mapping override-keys to replacement prompt "
            "templates. When "
            "omitted (the default), the built-in prompts are used and "
            "behavior is identical to pre-registry runs. See "
            "iptc.ptc_tool.OVERRIDE_KEYS for valid keys."
        ),
    )
    parser.add_argument(
        "--preload-learned-docs", type=str, default=None,
        help=(
            "Path to a JSON file of previously-learned tool-doc addenda. "
            "When set, a DocLearner is constructed in apply-only mode and "
            "the addenda are injected into tool descriptions at run start. "
            "Omit (default) to run with stock docs."
        ),
    )
    parser.add_argument(
        "--task-timeout-s", type=int, default=0,
        help=(
            "Per-task wall-clock timeout in seconds. 0 (default) disables it, "
            "which keeps the in-process thread pool path and a 1200s outer "
            "safety bound. >0 enables process-based isolation: each task runs "
            "in a subprocess that is killed if it exceeds the budget, and a "
            "synthetic trace with stop_reason='time_exceeded' is written so "
            "the task is counted as a failure rather than silently dropped."
        ),
    )
    args = parser.parse_args()

    # Paper interface names map to internal modes; iptc also enables inspection and learning.
    is_full_iptc = args.mode == "iptc"
    args.mode = _MODE_ALIASES.get(args.mode, args.mode)
    if is_full_iptc:
        args.tool_inspection = True
    if args.learning_batch_size is None:
        args.learning_batch_size = 10 if is_full_iptc else 0
    if args.learner_model is None:
        args.learner_model = args.model if is_full_iptc else "gpt-5-mini"

    # Inspection would hide all API tools from native backends, which have no inspector.
    if args.backend in _NATIVE_BACKENDS and args.tool_inspection:
        parser.error(
            f"--backend {args.backend} cannot be combined with on-demand tool "
            "inspection (--tool-inspection, or --mode iptc which turns it on): "
            "the native backend would be left with no API tools. Use the "
            "default --mode ptc for native runs."
        )

    if args.prompt_overrides:
        from iptc.ptc_tool import set_prompt_overrides
        with open(args.prompt_overrides) as f:
            set_prompt_overrides(json.load(f))
    # Inspection needs a code-execution surface for ?? syntax.
    if args.learning_max_patterns is not None and args.learning_max_patterns < 1:
        print("Error: --learning-max-patterns must be at least 1.")
        sys.exit(1)
    if args.tool_inspection and args.mode == "tool_only":
        print(
            "Error: --tool-inspection requires a code-execution mode (ptc, "
            "raw_iptc or iptc). "
            "tool_only has no code surface to write ``??`` syntax in."
        )
        sys.exit(1)

    effective_mode = "native_ptc" if args.backend in _NATIVE_BACKENDS else args.mode

    if args.queries_file:
        from ptc_bench.utils import load_file
        tasks = load_file(args.queries_file)
    else:
        tasks = load_dataset_entry("main")
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Error: task '{args.task_id}' not found.")
            sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(":", "_")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / "experiments" / "traces" / f"{model_slug}_{effective_mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A learner is needed for both new addenda and preloaded improvements.
    doc_learner: DocLearner | None = None
    learning_batch = args.learning_batch_size
    preloaded_count = 0
    if learning_batch > 0 or args.preload_learned_docs:
        learner_backend = args.learner_backend or (
            args.backend if args.learner_model == args.model else None
        )
        doc_learner = DocLearner(model=args.learner_model, backend=learner_backend)
        if args.preload_learned_docs:
            preloaded_count = doc_learner.preload(args.preload_learned_docs)
        if learning_batch > 0 and learning_batch < args.workers:
            print(
                f"  Note: learning-batch-size ({learning_batch}) < workers "
                f"({args.workers}); effective parallelism is {learning_batch}"
            )

    print(f"PTC-Bench Agent Eval")
    print(f"  Backend:  {args.backend}")
    print(f"  Model:    {args.model}")
    print(f"  Mode:     {effective_mode}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Workers:  {args.workers}")
    if doc_learner:
        learn_state = (
            f"batch-size={learning_batch}"
            if learning_batch > 0 else "apply-only (no new learning)"
        )
        print(f"  Learning: {learn_state}, learner-model={args.learner_model}, learner-backend={doc_learner._detect_backend()}")
        if preloaded_count:
            print(f"  Preload:  {preloaded_count} method(s) "
                  f"from {args.preload_learned_docs}")
    print(f"  Output:   {out_dir}")
    print()

    token_budget = args.max_token_budget if args.max_token_budget else None

    # Workers finishing after timeout must not overwrite synthetic timeout traces.
    # The lock makes the timeout check and write atomic.
    timed_out_task_ids: set[str] = set()
    trace_write_lock = threading.Lock()

    def _write_trace(task_id: str, trace: dict) -> None:
        with trace_write_lock:
            if task_id in timed_out_task_ids:
                print(f"    [{task_id}] finished after the run-level timeout; "
                      f"keeping the recorded timeout trace")
                return
            with open(out_dir / f"{task_id}.json", "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
    # Track worker start times for synthetic timeout traces; CPython permits per-key writes.
    task_start_times: dict[str, float] = {}

    def _process_task(
        index_task: tuple[int, dict],
        _doc_learner: DocLearner | None = None,
    ) -> tuple[int, dict, dict]:
        """Run a single task in-process and return (index, trace_dict, summary_row)."""
        i, task = index_task
        print(f"[{i}/{len(tasks)}]  Running task: {task['id']}")
        t0 = time.time()
        task_start_times[task["id"]] = t0
        try:
            trace = _run_single_task(
                task, args.backend, args.model, args.mode,
                args.max_turns, args.temperature, token_budget,
                multi_turn=args.multi_turn,
                minimal_print=args.minimal_print,
                doc_learner=_doc_learner,
                tool_inspection=args.tool_inspection,
                expose_all_apis=args.expose_all_apis,
                record_runtime_metrics=args.record_runtime_metrics,
            )
        except Exception as exc:
            trace = {
                "task_id": task["id"],
                "model": args.model,
                "error": f"Runner error: {exc}",
                "run_config": {
                    "backend": args.backend,
                    "model": args.model,
                    "mode": effective_mode,
                    "max_turns": args.max_turns,
                    "temperature": args.temperature,
                },
            }
            print(f"    ERROR on {task['id']}: {exc}")

        elapsed = time.time() - t0
        trace["wall_time_s"] = round(elapsed, 2)

        if _doc_learner is not None:
            trace["doc_version"] = _doc_learner.version

        _write_trace(task["id"], trace)
        summary_row = _summary_row_from_trace(trace, task["id"])
        tok = summary_row["total_tokens"]
        tok_str = f", {tok} tok" if tok else ""
        print(
            f"    [{task['id']}] Done ({elapsed:.1f}s, "
            f"{summary_row['tool_calls']} tool calls, "
            f"{summary_row['llm_turns']} turns{tok_str})"
        )
        return i, trace, summary_row

    def _process_task_subprocess(
        index_task: tuple[int, dict],
    ) -> tuple[int, dict, dict]:
        """Run a task in a killable subprocess; write a synthetic time_exceeded trace on timeout."""
        i, task = index_task
        task_id = task["id"]
        print(f"[{i}/{len(tasks)}]  Running task: {task_id} (timeout={args.task_timeout_s}s)")
        t0 = time.time()

        with tempfile.TemporaryDirectory() as td:
            tmp_out = Path(td) / f"{task_id}.json"
            cmd = [
                sys.executable, "-m", "experiments._single_task",
                "--task-id", task_id,
                "--out-file", str(tmp_out),
                "--backend", args.backend,
                "--model", args.model,
                "--mode", args.mode,
                "--max-turns", str(args.max_turns),
                "--max-token-budget", str(args.max_token_budget),
            ]
            if args.temperature is not None:
                cmd += ["--temperature", str(args.temperature)]
            if args.multi_turn:
                cmd += ["--multi-turn"]
            if args.minimal_print:
                cmd += ["--minimal-print"]
            if args.queries_file:
                cmd += ["--queries-file", args.queries_file]
            if args.prompt_overrides:
                cmd += ["--prompt-overrides", args.prompt_overrides]
            # Send both forms explicitly so subprocess defaults cannot diverge from the parent.
            if args.tool_inspection:
                cmd += ["--tool-inspection"]
            if args.record_runtime_metrics:
                cmd += ["--record-runtime-metrics"]
            cmd += [
                "--expose-all-apis" if args.expose_all_apis else "--no-expose-all-apis"
            ]
            # Only preload artifacts are forwarded; batched learning disables subprocess mode.
            if args.preload_learned_docs:
                cmd += ["--preload-learned-docs", args.preload_learned_docs]
            if args.learner_model:
                cmd += ["--learner-model", args.learner_model]

            timed_out = False
            try:
                subprocess.run(
                    cmd, timeout=args.task_timeout_s,
                    cwd=str(ROOT), check=False,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"    [TIMEOUT] {task_id} killed after {args.task_timeout_s}s")

            elapsed = time.time() - t0
            if timed_out:
                trace = _build_timeout_trace(
                    task_id=task_id,
                    backend=args.backend, model=args.model, mode=args.mode,
                    max_turns=args.max_turns, max_token_budget=token_budget,
                    temperature=args.temperature, multi_turn=args.multi_turn,
                    minimal_print=args.minimal_print,
                    timeout_s=args.task_timeout_s, elapsed_s=elapsed,
                )
            elif tmp_out.exists():
                with open(tmp_out) as f:
                    trace = json.load(f)
            else:
                effective = "native_ptc" if args.backend in _NATIVE_BACKENDS else args.mode
                trace = {
                    "task_id": task_id,
                    "model": args.model,
                    "system_prompt": "",
                    "turns": [],
                    "final_answer": None,
                    "stop_reason": "runner_error",
                    "wall_time_s": round(elapsed, 2),
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "error": "Subprocess exited without producing a trace file",
                    "run_config": {
                        "backend": args.backend, "model": args.model,
                        "mode": effective, "max_turns": args.max_turns,
                        "max_token_budget": token_budget,
                        "temperature": args.temperature,
                        "multi_turn": args.multi_turn,
                        "minimal_print": args.minimal_print,
                    },
                }

            _write_trace(task_id, trace)

        summary_row = _summary_row_from_trace(trace, task_id)
        print(
            f"    [{task_id}] Done ({elapsed:.1f}s, stop_reason={trace.get('stop_reason')})"
        )
        return i, trace, summary_row

    def _record_timeouts(
        futures_map: dict, timeout_s: int,
    ) -> list[str]:
        """Write failure traces for uncollected tasks and return their task IDs.

        Elapsed time comes from worker start timestamps. Partial token counts
        are unavailable and remain zero."""
        now = time.time()
        recorded: list[str] = []
        for future, (i, task) in futures_map.items():
            if i in results_by_index:
                continue
            try:
                future.cancel()
            except Exception:
                pass
            # Queued tasks have not started and therefore have zero elapsed time.
            start = task_start_times.get(task["id"])
            elapsed = round(now - start, 2) if start is not None else 0.0
            err_msg = (
                f"Timed out: task did not complete within the "
                f"run-level timeout of {timeout_s}s."
            )
            trace = {
                "task_id": task["id"],
                "model": args.model,
                "stop_reason": "timeout",
                "error": err_msg,
                "turns": [],
                "wall_time_s": elapsed,
                "run_config": {
                    "backend": args.backend,
                    "model": args.model,
                    "mode": effective_mode,
                    "max_turns": args.max_turns,
                    "max_token_budget": args.max_token_budget,
                    "temperature": args.temperature,
                    "multi_turn": args.multi_turn,
                    "minimal_print": args.minimal_print,
                    "tool_inspection": args.tool_inspection,
                    "expose_all_apis": args.expose_all_apis,
                },
            }
            per_task_path = out_dir / f"{task['id']}.json"
            with trace_write_lock:
                # Mark first so late workers skip their writes.
                timed_out_task_ids.add(task["id"])
                with open(per_task_path, "w", encoding="utf-8") as f:
                    json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
            summary_row = {
                "task_id": task["id"],
                "stop_reason": "timeout",
                "tool_calls": 0,
                "ptc_execute_calls": 0,
                "ptc_cell_calls": 0,
                "top_level_tool_calls": 0,
                "llm_turns": 0,
                "has_final_answer": False,
                "wall_time_s": elapsed,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": err_msg,
            }
            results_by_index[i] = (trace, summary_row)
            recorded.append(task["id"])
        return recorded

    def _record_worker_failure(
        i: int, task: dict, exc: BaseException,
    ) -> None:
        """Write a failure trace and print the traceback for an unhandled worker exception."""
        import traceback as _tb
        tid = task["id"]
        exc_type = type(exc).__name__
        err_msg = f"Worker exception ({exc_type}): {exc}"
        print(
            f"\n  !! WORKER EXCEPTION on {tid}: {err_msg}",
            file=sys.stderr,
        )
        _tb.print_exception(exc, file=sys.stderr)
        sys.stderr.flush()

        start = task_start_times.get(tid)
        elapsed = (
            round(time.time() - start, 2) if start is not None else 0.0
        )
        trace = {
            "task_id": tid,
            "model": args.model,
            "stop_reason": "worker_exception",
            "error": err_msg,
            "turns": [],
            "wall_time_s": elapsed,
            "run_config": {
                "backend": args.backend,
                "model": args.model,
                "mode": effective_mode,
                "max_turns": args.max_turns,
                "max_token_budget": args.max_token_budget,
                "temperature": args.temperature,
                "multi_turn": args.multi_turn,
                "minimal_print": args.minimal_print,
                "tool_inspection": args.tool_inspection,
                    "expose_all_apis": args.expose_all_apis,
            },
        }
        per_task_path = out_dir / f"{tid}.json"
        with open(per_task_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
        summary_row = {
            "task_id": tid,
            "stop_reason": "worker_exception",
            "tool_calls": 0,
            "ptc_execute_calls": 0,
            "ptc_cell_calls": 0,
            "top_level_tool_calls": 0,
            "llm_turns": 0,
            "has_final_answer": False,
            "wall_time_s": elapsed,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": err_msg,
        }
        results_by_index[i] = (trace, summary_row)

    # -- Run tasks (batched or all-at-once) ---------------------------------
    indexed_tasks = list(enumerate(tasks, 1))
    results_by_index: dict[int, tuple[dict, dict]] = {}
    n_workers = max(1, args.workers)

    use_subprocess = args.task_timeout_s and args.task_timeout_s > 0

    use_batches = learning_batch > 0
    if use_batches:
        effective_batch = learning_batch

    if use_subprocess and use_batches:
        print(
            "  WARNING: --task-timeout-s is incompatible with batched learning "
            "(--learning-batch-size needs in-process state). "
            "Timeout enforcement disabled for this run; tasks will run in-thread "
            "with no per-task timeout."
        )
        use_subprocess = False

    # Only subprocess workers can be killed per task (time_exceeded).
    # In-thread runs use an overall timeout and reject late writes.
    # Batched runs rely on the agent's turn and token limits.

    if use_batches:
        batches = [
            indexed_tasks[i:i + effective_batch]
            for i in range(0, len(indexed_tasks), effective_batch)
        ]
        effective_workers = min(n_workers, effective_batch)
        for batch_idx, batch in enumerate(batches):
            header_bits = [f"{len(batch)} tasks"]
            if doc_learner is not None:
                header_bits.append(f"doc v{doc_learner.version}")
            print(f"--- Batch {batch_idx + 1}/{len(batches)} "
                  f"({', '.join(header_bits)}) ---")

            pool = ThreadPoolExecutor(max_workers=effective_workers)
            futures = {
                pool.submit(
                    _process_task, it, doc_learner,
                ): it
                for it in batch
            }
            batch_traces: list[dict] = []
            for future in as_completed(futures):
                fut_i, fut_task = futures[future]
                try:
                    i, trace, summary_row = future.result()
                    results_by_index[i] = (trace, summary_row)
                    batch_traces.append(trace)
                except (Exception, SystemExit) as exc:
                    # Write a failure placeholder so one worker exception does not abort the batch.
                    _record_worker_failure(fut_i, fut_task, exc)
            pool.shutdown(wait=True)

            # No later batch benefits from learning after the final batch.
            is_last_batch = (batch_idx == len(batches) - 1)

            if doc_learner is not None and learning_batch > 0 and not is_last_batch:
                errors = doc_learner.collect_from_traces(batch_traces)
                if errors:
                    print(f"  Collecting errors: {len(errors)} new pattern(s)")
                    if (args.learning_max_patterns is not None
                            and len(errors) > args.learning_max_patterns):
                        errors = DocLearner.select_patterns(errors, args.learning_max_patterns)
                        print(f"  Learning from the {len(errors)} most frequent "
                              f"(--learning-max-patterns)")
                    sample_task = batch[0][1]
                    _, ref_tools, _, ref_inspector = _make_executor_and_tools(
                        sample_task, args.mode,
                        minimal_print=args.minimal_print,
                        tool_inspection=args.tool_inspection,
                        expose_all_apis=args.expose_all_apis,
                    )
                    # Triage needs the hidden API method catalog in inspection mode.
                    if ref_inspector is not None:
                        ref_tools = ref_tools + ref_inspector
                    improvements = doc_learner.improve_docs(errors, ref_tools)
                    if improvements:
                        print(f"  Applied {len(improvements)} doc improvement(s) "
                              f"→ doc v{doc_learner.version}")
                    else:
                        print("  No doc improvements generated (errors were code-level, not doc gaps)")
                else:
                    print("  No new error patterns in this batch")

            print()
    elif use_subprocess:
        # Each subprocess enforces its own hard timeout and writes a trace even on timeout.
        pool = ThreadPoolExecutor(max_workers=n_workers)
        futures = {
            pool.submit(_process_task_subprocess, it): it
            for it in indexed_tasks
        }
        for future in as_completed(futures):
            i, trace, summary_row = future.result()
            results_by_index[i] = (trace, summary_row)
        pool.shutdown(wait=True)
    else:
        # In-thread runs may apply preloaded docs, but do not learn between tasks.
        pool = ThreadPoolExecutor(max_workers=n_workers)
        futures = {
            pool.submit(_process_task, it, doc_learner): it
            for it in indexed_tasks
        }
        # Allow two minutes per task across workers, with a 30-minute floor.
        overall_timeout = max(1800, len(indexed_tasks) * 120)
        try:
            for future in as_completed(futures, timeout=overall_timeout):
                fut_i, fut_task = futures[future]
                try:
                    i, trace, summary_row = future.result()
                    results_by_index[i] = (trace, summary_row)
                except (Exception, SystemExit) as exc:
                    _record_worker_failure(fut_i, fut_task, exc)
        except TimeoutError:
            timed_out_ids = _record_timeouts(futures, overall_timeout)
            print(
                f"\n  WARNING: {len(timed_out_ids)} task(s) timed out after "
                f"{overall_timeout}s; recorded as failed: "
                f"{', '.join(timed_out_ids)}"
            )
        pool.shutdown(wait=False, cancel_futures=True)

    # Surface missing results so every requested task appears in the summary.
    for i, task in indexed_tasks:
        if i not in results_by_index:
            start = task_start_times.get(task["id"])
            elapsed = (
                round(time.time() - start, 2) if start is not None else 0.0
            )
            err_msg = (
                "Worker never returned a result (task was dispatched but "
                "no future completion / timeout / exception was observed)."
            )
            lost_trace = {
                "task_id": task["id"],
                "model": args.model,
                "stop_reason": "lost",
                "error": err_msg,
                "turns": [],
                "wall_time_s": elapsed,
                "run_config": {
                    "backend": args.backend,
                    "model": args.model,
                    "mode": effective_mode,
                    "max_turns": args.max_turns,
                    "max_token_budget": args.max_token_budget,
                    "temperature": args.temperature,
                    "multi_turn": args.multi_turn,
                    "minimal_print": args.minimal_print,
                    "tool_inspection": args.tool_inspection,
                    "expose_all_apis": args.expose_all_apis,
                },
            }
            per_task_path = out_dir / f"{task['id']}.json"
            if not per_task_path.exists():
                with open(per_task_path, "w", encoding="utf-8") as f:
                    json.dump(lost_trace, f, indent=2, ensure_ascii=False, default=str)
            lost_row = {
                "task_id": task["id"],
                "stop_reason": "lost",
                "tool_calls": 0,
                "ptc_execute_calls": 0,
                "ptc_cell_calls": 0,
                "top_level_tool_calls": 0,
                "llm_turns": 0,
                "has_final_answer": False,
                "wall_time_s": elapsed,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": err_msg,
            }
            results_by_index[i] = (lost_trace, lost_row)
            print(f"  !! TASK LOST: {task['id']} had no result — recorded as 'lost'")

    all_traces: list[dict] = []
    summary_rows: list[dict] = []
    for i in sorted(results_by_index):
        trace, summary_row = results_by_index[i]
        all_traces.append(trace)
        summary_rows.append(summary_row)

    print(f"\nAll {len(all_traces)} tasks completed. Writing summary...")

    # Per-task files already preserve large runs.
    if len(all_traces) <= 30:
        all_path = out_dir / "all_traces.json"
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_traces, f, indent=2, ensure_ascii=False, default=str)

    summary_path = out_dir / "summary.json"
    total_in = sum(r["input_tokens"] for r in summary_rows)
    total_out = sum(r["output_tokens"] for r in summary_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "backend": args.backend,
            "model": args.model,
            "mode": effective_mode,
            "timestamp": ts,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "max_token_budget": args.max_token_budget,
            "tasks_total": len(tasks),
            "tasks_completed": sum(1 for r in summary_rows if r.get("stop_reason") == "completed"),
            "tasks_hit_max_turns": sum(1 for r in summary_rows if r.get("stop_reason") == "max_turns"),
            "tasks_hit_token_budget": sum(1 for r in summary_rows if r.get("stop_reason") == "token_budget"),
            "tasks_hit_length_cap": sum(1 for r in summary_rows if r.get("stop_reason") == "length_cap"),
            "tasks_hit_time_exceeded": sum(1 for r in summary_rows if r.get("stop_reason") == "time_exceeded"),
            "tasks_errored": sum(1 for r in summary_rows if r.get("error")),
            "total_tool_calls": sum(r.get("tool_calls", 0) for r in summary_rows),
            "total_ptc_execute_calls": sum(r.get("ptc_execute_calls", 0) for r in summary_rows),
            "total_ptc_cell_calls": sum(r.get("ptc_cell_calls", 0) for r in summary_rows),
            "total_top_level_tool_calls": sum(
                r.get("top_level_tool_calls", 0) for r in summary_rows
            ),
            "total_llm_turns": sum(r.get("llm_turns", 0) for r in summary_rows),
            "total_wall_time_s": round(sum(r.get("wall_time_s", 0) for r in summary_rows), 2),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            **({"doc_learner": doc_learner.snapshot()} if doc_learner else {}),
            "per_task": summary_rows,
        }, f, indent=2, ensure_ascii=False)

    # Snapshot in-memory docs without changing the source JSON files.
    learned_path = None
    if doc_learner is not None:
        augmented = doc_learner.dump_augmented_docs()
        if augmented:
            learned_path = out_dir / "learned_docs.json"
            with open(learned_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done. {len(tasks)} tasks processed.")
    print(f"  Per-task traces: {out_dir}/<task_id>.json")
    print(f"  Summary:         {summary_path}")
    if learned_path is not None:
        print(f"  Learned docs:    {learned_path}")

    for attr in ("_openai_client", "_qwen_client", "_glm_client", "_kimi_client"):
        client = getattr(Agent, attr, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
    import os
    # os._exit skips buffer flushing, including redirected stdout.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # Force exit to kill any lingering httpx threads
