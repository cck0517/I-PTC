#!/usr/bin/env python3
"""Single-task subprocess entrypoint for ``run_eval``.

Runs exactly one PTC-Bench task and writes the resulting trace JSON to
``--out-file``. Used by the parent runner when ``--task-timeout-s`` is set so
that a hung task can be killed at the OS level without taking down sibling
tasks. No summary is written — the parent owns aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_eval.agent import Agent
from agent_eval.run_eval import _run_single_task  # noqa: E402
from ptc_eval.utils import load_dataset_entry


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single PTC-Bench task.")
    p.add_argument("--task-id", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--backend", required=True,
                   choices=["openai", "gemini", "claude", "claude_ptc", "qwen", "glm", "kimi", "openrouter", "openai_native", "gemini_native"])
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-token-budget", type=int, default=0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--multi-turn", action="store_true", default=False)
    p.add_argument("--minimal-print", action="store_true", default=False)
    p.add_argument("--queries-file", default=None)
    p.add_argument("--prompt-overrides", default=None)
    # Execution-shape flags — must mirror the parent's defaults (especially
    # ``--expose-all-apis``, whose parent default is True).
    p.add_argument("--tool-inspection", action="store_true", default=False)
    p.add_argument("--record-runtime-metrics", action="store_true", default=False)
    p.add_argument(
        "--expose-all-apis", action=argparse.BooleanOptionalAction, default=True,
    )
    # Preload artifacts. The batched-learning flag (--learning-batch-size)
    # intentionally isn't forwarded — the parent disables subprocess mode
    # whenever it's active, so a subprocess run is guaranteed single-pass
    # and only needs preload.
    p.add_argument("--preload-learned-docs", default=None)
    p.add_argument("--learner-model", default="gpt-5-mini")
    args = p.parse_args()

    if args.prompt_overrides:
        from agent_eval.ptc_tool import set_prompt_overrides
        with open(args.prompt_overrides) as f:
            set_prompt_overrides(json.load(f))

    if args.queries_file:
        from ptc_eval.utils import load_file
        tasks = load_file(args.queries_file)
    else:
        tasks = load_dataset_entry("main")
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if task is None:
        print(f"Error: task '{args.task_id}' not found.", file=sys.stderr)
        sys.exit(2)

    # Mirror the parent's preload paths: applying preloaded artifacts to
    # this single task is independent of whether learning is happening.
    doc_learner = None
    if args.preload_learned_docs:
        from agent_eval.doc_learner import DocLearner
        doc_learner = DocLearner(model=args.learner_model)
        doc_learner.preload(args.preload_learned_docs)

    token_budget = args.max_token_budget if args.max_token_budget else None
    t0 = time.time()
    try:
        trace = _run_single_task(
            task, args.backend, args.model, args.mode,
            args.max_turns, args.temperature, token_budget,
            multi_turn=args.multi_turn,
            minimal_print=args.minimal_print,
            doc_learner=doc_learner,
            tool_inspection=args.tool_inspection,
            expose_all_apis=args.expose_all_apis,
            record_runtime_metrics=args.record_runtime_metrics,
        )
    except Exception as exc:
        effective_mode = "native_ptc" if args.backend in ("claude_ptc", "openai_native", "gemini_native") else args.mode
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

    trace["wall_time_s"] = round(time.time() - t0, 2)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

    if hasattr(Agent, "_openai_client") and Agent._openai_client is not None:
        try:
            Agent._openai_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
    # os._exit skips the interpreter's buffer flush; flush explicitly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
