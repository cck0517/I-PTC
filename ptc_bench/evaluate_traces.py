#!/usr/bin/env python3
"""Verify saved PTC-Bench traces and aggregate accuracy, token, timing, and tool metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ptc_bench.trajectory_tokens import compute_trajectory_tokens
from ptc_bench.eval_checker.multi_turn_eval.api_method_types import (
    compute_trace_api_type_distribution,
)
from ptc_bench.eval_checker.verifiers import verify
from ptc_bench.utils import load_dataset_entry, load_ground_truth_entry


def _load_task_index() -> dict[str, dict]:
    tasks = load_dataset_entry("main")
    return {t["id"]: t for t in tasks}


def _load_gt_index() -> dict[str, dict]:
    entries = load_ground_truth_entry("main")
    return {e["id"]: e["ground_truth"] for e in entries}


def _extract_trace_metrics(
    trace: dict,
    task: dict | None = None,
) -> dict[str, Any]:
    """Extract trace metrics, reconstructing trajectory token counts when task data is provided."""
    turns = trace.get("turns", [])
    llm_turns = [t for t in turns if t.get("role") == "assistant"]
    tool_calls_flat = []
    for t in turns:
        for tc in t.get("tool_calls", []):
            tool_calls_flat.append(tc)

    ptc_execute_calls = [tc for tc in tool_calls_flat if tc.get("tool_name") == "ptc_execute"]
    ptc_cell_calls = [tc for tc in tool_calls_flat if tc.get("tool_name") == "ptc_cell"]
    direct_api_calls = [
        tc for tc in tool_calls_flat
        if tc.get("tool_name") not in ("ptc_execute", "ptc_cell")
    ]

    # api_calls is truncated; prefer the uncapped count, falling back for older traces.
    ptc_sub_api_calls = 0
    for tc in ptc_execute_calls + ptc_cell_calls:
        result = tc.get("result")
        if isinstance(result, dict):
            ptc_sub_api_calls += result.get(
                "api_call_count", len(result.get("api_calls") or [])
            )

    total_api_calls = len(direct_api_calls) + ptc_sub_api_calls

    # Soft errors are API-returned error dicts; hard errors are exceptions that abort a call.
    # Count both in num_api_errors and retain the breakdown for diagnostics.
    num_api_errors_soft = 0
    num_api_errors_hard = 0
    for tc in ptc_execute_calls + ptc_cell_calls:
        result = tc.get("result")
        if isinstance(result, dict):
            num_api_errors_soft += len(result.get("api_errors") or [])
            if result.get("error"):
                num_api_errors_hard += 1
    for tc in direct_api_calls:
        result = tc.get("result")
        if isinstance(result, dict) and result.get("error"):
            num_api_errors_soft += 1
        if tc.get("error"):
            num_api_errors_hard += 1
    num_api_errors = num_api_errors_soft + num_api_errors_hard

    latencies = [t.get("latency_s", 0) for t in llm_turns]

    # Prefer recorded API types; derive them for older traces without affecting agent input.
    api_type_dist = trace.get("api_type_distribution")
    if not isinstance(api_type_dist, dict):
        api_type_dist = compute_trace_api_type_distribution(trace)

    metrics = {
        "num_llm_turns": len(llm_turns),
        "num_tool_calls": len(tool_calls_flat),
        "num_ptc_execute_calls": len(ptc_execute_calls),
        "num_ptc_cell_calls": len(ptc_cell_calls),
        "num_api_calls": total_api_calls,
        "num_api_errors": num_api_errors,
        "num_api_errors_soft": num_api_errors_soft,
        "num_api_errors_hard": num_api_errors_hard,
        "api_type_distribution": dict(api_type_dist),
        "total_input_tokens": trace.get("total_input_tokens", 0),
        "total_output_tokens": trace.get("total_output_tokens", 0),
        "total_tokens": (
            trace.get("total_input_tokens", 0)
            + trace.get("total_output_tokens", 0)
        ),
        "wall_time_s": trace.get("wall_time_s", trace.get("total_time_s", 0)),
        "mean_turn_latency_s": round(
            sum(latencies) / len(latencies), 3
        ) if latencies else 0,
        "has_error": trace.get("error") is not None,
        "has_final_answer": trace.get("final_answer") is not None,
    }

    tj = compute_trajectory_tokens(trace, task=task)
    metrics.update(tj)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agent traces against PTC-Bench verifiers.",
    )
    parser.add_argument(
        "--traces-dir", required=True,
        help="Directory containing per-task trace JSON files.",
    )
    parser.add_argument(
        "--queries-file", default=None,
        help=(
            "Path to a custom JSONL queries file. If omitted, uses the "
            "default ptc_bench/data/main.jsonl."
        ),
    )
    parser.add_argument(
        "--gt-file", default=None,
        help=(
            "Path to a custom JSONL ground-truth file. If omitted, uses the "
            "default ptc_bench/data/possible_answer/main.jsonl."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write evaluation results JSON. Defaults to <traces-dir>/eval_results.json.",
    )
    parser.add_argument(
        "--summary-csv", default=None,
        help=(
            "If set, append one row of summary stats to this CSV file. "
            "Header is written automatically when the file doesn't exist. "
            "Useful for accumulating stats across multiple eval runs "
            "(e.g. a model sweep). Columns: timestamp, traces_dir, model, "
            "mode, tasks_passed, tasks_evaluated, pass_rate, "
            "avg_tokens_per_task, avg_trajectory_tokens_per_task, "
            "num_api_errors, num_api_errors_soft, num_api_errors_hard, "
            "avg_api_errors_per_task, "
            "num_api_calls, api_dist_io, api_dist_compute, "
            "api_dist_io_mutation, api_dist_unknown."
        ),
    )
    parser.add_argument(
        "--run-label", default=None,
        help=(
            "Optional label included in --summary-csv rows (e.g. 'run1', "
            "'sweep_gpt5mini_rep2'). Helpful when the same model+mode is "
            "evaluated multiple times for variance studies."
        ),
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    if not traces_dir.is_dir():
        print(f"Error: {traces_dir} is not a directory.")
        sys.exit(1)

    if args.queries_file:
        from ptc_bench.utils import load_file
        task_index = {t["id"]: t for t in load_file(args.queries_file)}
    else:
        task_index = _load_task_index()

    if args.gt_file:
        from ptc_bench.utils import load_file as _lf
        gt_index = {e["id"]: e["ground_truth"] for e in _lf(args.gt_file)}
    else:
        gt_index = _load_gt_index()

    trace_files = sorted(traces_dir.glob("*.json"))
    skip = {"all_traces.json", "summary.json", "eval_results.json", "learned_docs.json"}
    trace_files = [f for f in trace_files if f.name not in skip and not f.name.endswith("_final_state.json")]

    if not trace_files:
        print("No trace files found.")
        sys.exit(1)

    results: list[dict[str, Any]] = []
    total_passed = 0

    col_w = {"task": 50, "pass": 6, "turns": 6, "tools": 6, "tokens": 8, "time": 7}
    header = (
        f"{'Task':<{col_w['task']}} "
        f"{'Pass':<{col_w['pass']}} "
        f"{'Turns':<{col_w['turns']}} "
        f"{'Tools':<{col_w['tools']}} "
        f"{'Tokens':<{col_w['tokens']}} "
        f"{'Time':<{col_w['time']}}"
    )
    print(header)
    print("-" * len(header))

    for tf in trace_files:
        with open(tf, encoding="utf-8") as f:
            trace = json.load(f)

        task_id = trace.get("task_id", tf.stem)
        task_def = task_index.get(task_id)
        ground_truth = gt_index.get(task_id)

        if task_def is None:
            row = {
                "task_id": task_id,
                "passed": False,
                "score": 0.0,
                "error": "Task definition not found",
            }
            results.append(row)
            print(
                f"{task_id:<{col_w['task']}} "
                f"{'--':<{col_w['pass']}} "
                f"{'--':<{col_w['turns']}} "
                f"{'--':<{col_w['tools']}} "
                f"{'--':<{col_w['tokens']}} "
                f"{'N/A':<{col_w['time']}}"
            )
            continue

        final_answer = trace.get("final_answer")
        final_state = trace.get("final_state")
        turn_answers = trace.get("turn_answers")
        initial_config = task_def["initial_config"]

        try:
            vr = verify(
                task_id=task_id,
                final_answer=final_answer,
                final_state=final_state,
                initial_config=initial_config,
                ground_truth=ground_truth or {},
                turn_answers=turn_answers,
            )
        except Exception as exc:
            # A malformed final state can crash a verifier; fail this task and finish the batch.
            from ptc_bench.eval_checker.verifiers.core import VerificationResult
            vr = VerificationResult(
                passed=False,
                score=0.0,
                checks={"verifier_error": False},
                errors=[f"verifier raised {type(exc).__name__}: {exc}"],
            )

        total_passed += int(vr.passed)

        metrics = _extract_trace_metrics(trace, task=task_def)

        row = {
            "task_id": task_id,
            "stop_reason": trace.get("stop_reason"),
            **vr.to_dict(),
            **metrics,
        }
        results.append(row)

        status = "PASS" if vr.passed else "FAIL"
        print(
            f"{task_id:<{col_w['task']}} "
            f"{status:<{col_w['pass']}} "
            f"{metrics['num_llm_turns']:<{col_w['turns']}} "
            f"{metrics['num_tool_calls']:<{col_w['tools']}} "
            f"{metrics['total_tokens']:<{col_w['tokens']}} "
            f"{metrics['wall_time_s']:<{col_w['time']}.1f}"
        )

    print("-" * len(header))
    n = len(results)

    agg_turns = sum(r.get("num_llm_turns", 0) for r in results)
    agg_tools = sum(r.get("num_tool_calls", 0) for r in results)
    agg_tokens = sum(r.get("total_tokens", 0) for r in results)
    agg_time = sum(r.get("wall_time_s", 0) for r in results)

    print(
        f"{'TOTAL':<{col_w['task']}} "
        f"{total_passed}/{n:<{col_w['pass'] - 2}} "
        f"{agg_turns:<{col_w['turns']}} "
        f"{agg_tools:<{col_w['tools']}} "
        f"{agg_tokens:<{col_w['tokens']}} "
        f"{agg_time:<{col_w['time']}.1f}"
    )
    _traj_total = sum(r.get("trajectory_total_tokens", 0) for r in results)
    _traj_in    = sum(r.get("trajectory_input_tokens", 0) for r in results)
    _traj_out   = sum(r.get("trajectory_output_tokens", 0) for r in results)
    print(
        f"  trajectory tokens (counted once): "
        f"total={_traj_total:,}  in={_traj_in:,}  out={_traj_out:,}  "
        f"(raw total={agg_tokens:,}, ratio={agg_tokens/_traj_total:.2f}x)"
        if _traj_total else ""
    )
    if n:
        avg_raw = agg_tokens / n
        avg_traj = _traj_total / n if _traj_total else 0
        print(
            f"  avg tokens per task: raw={avg_raw:,.0f}  trajectory={avg_traj:,.0f}"
        )
    _api_errs      = sum(r.get("num_api_errors", 0)      for r in results)
    _api_errs_soft = sum(r.get("num_api_errors_soft", 0) for r in results)
    _api_errs_hard = sum(r.get("num_api_errors_hard", 0) for r in results)
    if n:
        print(
            f"  tool errors: total={_api_errs:,} "
            f"(soft={_api_errs_soft}, hard={_api_errs_hard})  "
            f"per_task={_api_errs/n:.2f}"
        )

    agg_input_tokens = sum(r.get("total_input_tokens", 0) for r in results)
    agg_output_tokens = sum(r.get("total_output_tokens", 0) for r in results)
    agg_traj_input = sum(r.get("trajectory_input_tokens", 0) for r in results)
    agg_traj_output = sum(r.get("trajectory_output_tokens", 0) for r in results)
    agg_ptc_calls = sum(r.get("num_ptc_execute_calls", 0) for r in results)
    agg_ptc_cell_calls = sum(r.get("num_ptc_cell_calls", 0) for r in results)
    agg_api_calls = sum(r.get("num_api_calls", 0) for r in results)
    agg_api_errors = sum(r.get("num_api_errors", 0) for r in results)
    agg_api_errors_soft = sum(r.get("num_api_errors_soft", 0) for r in results)
    agg_api_errors_hard = sum(r.get("num_api_errors_hard", 0) for r in results)
    agg_api_type_dist: dict[str, int] = {}
    for r in results:
        for k, v in (r.get("api_type_distribution") or {}).items():
            agg_api_type_dist[k] = agg_api_type_dist.get(k, 0) + v
    tasks_with_answer = sum(1 for r in results if r.get("has_final_answer"))
    tasks_with_error = sum(1 for r in results if r.get("has_error"))

    run_config: dict[str, Any] = {}
    for tf in trace_files:
        with open(tf, encoding="utf-8") as f:
            t = json.load(f)
        if "run_config" in t:
            run_config = dict(t["run_config"])
            break

    # Normalize stale mode labels in older Claude PTC traces.
    if run_config.get("backend") == "claude_ptc":
        run_config["mode"] = "native_ptc"

    eval_output = {
        "traces_dir": str(traces_dir),
        "run_config": run_config,
        "tasks_evaluated": n,
        "tasks_passed": total_passed,
        "pass_rate": round(total_passed / n, 4) if n else 0.0,
        "aggregate_metrics": {
            "total_llm_turns": agg_turns,
            "avg_llm_turns": round(agg_turns / n, 2) if n else 0,
            "total_tool_calls": agg_tools,
            "total_ptc_execute_calls": agg_ptc_calls,
            "total_ptc_cell_calls": agg_ptc_cell_calls,
            "total_api_calls": agg_api_calls,
            "total_api_errors": agg_api_errors,
            "total_api_errors_soft": agg_api_errors_soft,
            "total_api_errors_hard": agg_api_errors_hard,
            "avg_api_errors_per_task": round(agg_api_errors / n, 2) if n else 0,
            "api_type_distribution": agg_api_type_dist,
            "avg_tool_calls": round(agg_tools / n, 2) if n else 0,
            "total_input_tokens": agg_input_tokens,
            "total_output_tokens": agg_output_tokens,
            "total_tokens": agg_input_tokens + agg_output_tokens,
            "avg_tokens_per_task": round((agg_input_tokens + agg_output_tokens) / n, 0) if n else 0,
            "trajectory_input_tokens": agg_traj_input,
            "trajectory_output_tokens": agg_traj_output,
            "trajectory_total_tokens": agg_traj_input + agg_traj_output,
            "avg_trajectory_tokens_per_task": round(
                (agg_traj_input + agg_traj_output) / n, 0
            ) if n else 0,
            "total_wall_time_s": round(agg_time, 2),
            "avg_wall_time_s": round(agg_time / n, 2) if n else 0,
            "tasks_with_final_answer": tasks_with_answer,
            "tasks_with_error": tasks_with_error,
        },
        "per_task": results,
    }

    out_path = Path(args.output) if args.output else traces_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {out_path}")

    if args.summary_csv:
        import csv
        from datetime import datetime as _dt
        agg = eval_output["aggregate_metrics"]
        rc = eval_output["run_config"]
        # Flatten registered API types plus unknown into fixed summary columns.
        api_dist = agg.get("api_type_distribution") or {}
        row = {
            "timestamp": _dt.utcnow().isoformat(timespec="seconds") + "Z",
            "run_label": args.run_label or "",
            "traces_dir": str(traces_dir),
            "model": rc.get("model", ""),
            "mode": rc.get("mode", ""),
            "tasks_passed": eval_output["tasks_passed"],
            "tasks_evaluated": eval_output["tasks_evaluated"],
            "pass_rate": round(eval_output["pass_rate"], 4),
            "avg_tokens_per_task": int(agg.get("avg_tokens_per_task", 0)),
            "avg_trajectory_tokens_per_task": int(
                agg.get("avg_trajectory_tokens_per_task", 0)
            ),
            "num_api_calls": int(agg.get("total_api_calls", 0)),
            "num_api_errors": int(agg.get("total_api_errors", 0)),
            "num_api_errors_soft": int(agg.get("total_api_errors_soft", 0)),
            "num_api_errors_hard": int(agg.get("total_api_errors_hard", 0)),
            "avg_api_errors_per_task": round(agg.get("avg_api_errors_per_task", 0.0), 2),
            "api_dist_io": int(api_dist.get("io", 0)),
            "api_dist_compute": int(api_dist.get("compute", 0)),
            "api_dist_io_mutation": int(api_dist.get("io_mutation", 0)),
            "api_dist_unknown": int(api_dist.get("unknown", 0)),
        }
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"Summary row appended to {csv_path}")


if __name__ == "__main__":
    main()
