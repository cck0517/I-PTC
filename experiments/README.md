# Experiments and paper reproduction

This folder connects the [`iptc/`](../iptc/README.md) implementation with
[`ptc_bench/`](../ptc_bench/README.md) to run the paper's comparison conditions and save
agent traces. Run all commands below from the repository root.

## Setup

Tested with Python 3.12. Follow the [root setup instructions](../README.md#quick-start):

```bash
pip install -r requirements.txt
```

The minimum SDK versions in `requirements.txt` are the ones the harness was developed and run
with; older SDKs lack APIs it calls, such as the OpenAI Responses API and Anthropic's
code-execution tool.

**Security.** In the PTC and I-PTC conditions, the model's Python code runs with `exec()`
inside the harness process, with full builtins. It can read files and environment variables
(including your API keys) and reach the network. The tool descriptions call this a sandbox
from the model's point of view, but it is not an isolation boundary. Run the harness only in
a disposable container or VM that holds nothing beyond the API key the run needs.

Set the API key for the backend you use:

| `--backend` | Provider | Environment variable |
|---|---|---|
| `openai`, `openai_native` | OpenAI | `OPENAI_API_KEY` |
| `claude`, `claude_ptc` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini`, `gemini_native` | Google | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) |
| `openrouter` | OpenRouter (any hosted model, e.g. `moonshotai/kimi-k2.5`) | `OPENROUTER_API_KEY` |
| `qwen` | Alibaba DashScope | `DASHSCOPE_API_KEY` |
| `kimi` | Moonshot | `MOONSHOT_API_KEY` |
| `glm` | Z.ai | `ZHIPUAI_API_KEY` |

The OpenAI-compatible providers (`openrouter`, `qwen`, `kimi`, `glm`) also accept a base-URL
override: `OPENROUTER_BASE_URL`, `DASHSCOPE_BASE_URL`, `MOONSHOT_BASE_URL`, `GLM_BASE_URL`.
Any other OpenAI-compatible server (for example a self-hosted vLLM) works with
`--backend openai` and `OPENAI_BASE_URL` pointing at the server.

## Conditions

| Condition | Flags | What the agent gets |
|---|---|---|
| **Tool-only** | `--mode tool_only` | One tool per API method; no code execution |
| **PTC** | `--mode ptc` | API method tools plus `ptc_execute`, a stateless Python tool; the prompt encourages writing code |
| **Raw I-PTC** | `--mode raw_iptc` | API method tools plus `ptc_cell`, a persistent Jupyter-style cell, without inspection or doc learning |
| **I-PTC** | `--mode iptc` | Only `ptc_cell`. Methods are discovered inside cells with `<api>??` / `<api>.<method>??` (on-demand tool inspection). After every 10 tasks, an update model reads the batch's API errors and appends corrections to the tool descriptions for later tasks (self-evolving API documentation) |
| **Native PTC** | `--backend claude_ptc` | Anthropic's code-execution server tool, with the API tools callable from inside the sandbox |

`--mode iptc` is shorthand for `--mode raw_iptc --tool-inspection --learning-batch-size 10
--learner-model <same as --model>`; any of those flags given explicitly overrides the preset
(for example `--learning-batch-size 5`). The two mechanisms can also be added to Raw I-PTC
separately: `--mode raw_iptc --tool-inspection` or `--mode raw_iptc --learning-batch-size 10
--learner-model <model>`. They can also be added to stateless PTC (`--mode ptc --tool-inspection`,
`--mode ptc --learning-batch-size 10 --learner-model <model>`); these are the stateless rows of
the paper's component ablation.

By default every task exposes all ten APIs, not only the ones it uses
(`--no-expose-all-apis` restricts the catalog to the task's `involved_classes`). The older
mode names `ptc_encouraged` (= `ptc`) and `ptc_cell_encouraged` (= `raw_iptc`) are still
accepted. `openai_native` and `gemini_native` are native-interpreter baselines for OpenAI and
Gemini.

When `--learner-model` is the same as `--model`, the doc learner uses the run's own
`--backend`, so OpenRouter and self-hosted models work as their own learners. For a learner
from a different provider, add `--learner-backend` (`openai`, `claude`, `gemini`, `qwen`,
`glm`, `kimi` or `openrouter`). Without the flag, a different learner model is routed by its
name: `claude-*` to Anthropic, `gemini-*` to Google, `qwen*`, `glm*` and `kimi*`/`moonshot*`
to their providers, and anything else to OpenAI.

## Running

Main split, one model, all four conditions:

```bash
MODEL=gpt-5.4-mini

python -m experiments.run_eval --backend openai --model $MODEL --mode tool_only --workers 10
python -m experiments.run_eval --backend openai --model $MODEL --mode ptc --workers 10
python -m experiments.run_eval --backend openai --model $MODEL --mode iptc --workers 10

python -m experiments.run_eval --backend claude_ptc --model claude-sonnet-4-6 --workers 10
```

`python -m experiments` is an equivalent runner entry point.

The main split is the default. For the other splits, pass the file, and add `--multi-turn`
for the multi-turn split:

```bash
# Error recovery (add the backend, model, and mode flags from above)
python -m experiments.run_eval ... --queries-file ptc_bench/data/error_recovery.jsonl

# Multi-turn
python -m experiments.run_eval ... --queries-file ptc_bench/data/multi_turn.jsonl --multi-turn
```

Each run writes to `experiments/traces/<model>_<mode>_<timestamp>/` unless you pass
`--output-dir`. Run directories and traces record the internal mode name: `ptc_encouraged`
for `ptc`, and `ptc_cell_encouraged` for `raw_iptc` and `iptc` (the trace's `run_config`
also records `tool_inspection`).

Score saved traces with `python -m ptc_bench.evaluate_traces --traces-dir <run_dir>/`.
See the [benchmark scoring guide](../ptc_bench/README.md#scoring-traces) for split-specific
commands and result fields.

## Other options

| Flag | Effect |
|---|---|
| `--task-id <id>` | Run a single task |
| `--workers N` | Run N tasks in parallel (default 1) |
| `--max-turns N` | Agent turn limit per task (default 30) |
| `--max-token-budget N` | Stop a task after N total tokens (default off) |
| `--task-timeout-s N` | Per-task wall-clock limit; a task that times out counts as failed (default off) |
| `--temperature T` | Sampling temperature (leave unset for reasoning models) |
| `--record-runtime-metrics` | Record code-execution time, namespace size and process memory in each trace |
| `--minimal-print` | Ask the model to keep printed output short in code cells |
| `--learner-backend <name>` | Provider for the doc-learning model (see [Conditions](#conditions)) |
| `--learning-max-patterns N` | Learn from at most N new error patterns per batch, the most frequent first (default: no cap) |
| `--preload-learned-docs <file>` | Start from tool descriptions learned in an earlier run (its `learned_docs.json`) |

`python -m experiments.run_eval --help` lists everything.

## Output

```text
<run_dir>/
├── <task_id>.json        # Full trace: system prompt, turns, tool calls, tokens, latencies, final answer, final API state
├── summary.json          # Run configuration and per-task counts (turns, tool calls, tokens, wall time, stop reason)
├── all_traces.json       # All traces in one file (runs of 30 tasks or fewer)
├── learned_docs.json     # Tool descriptions learned during the run (I-PTC)
└── eval_results.json     # Written by ptc_bench.evaluate_traces
```

## Code

| File | Responsibility |
|---|---|
| [`run_eval.py`](run_eval.py) | Runner CLI, task setup, comparison conditions, batch learning, and trace output |
| [`_single_task.py`](_single_task.py) | Subprocess worker for task execution with a wall-clock timeout |
| [`__main__.py`](__main__.py) | `python -m experiments` entry point |
