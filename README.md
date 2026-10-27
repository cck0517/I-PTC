# PTC-Bench

A benchmark for evaluating LLM agents' **programmatic tool calling (PTC)**. Agents solve
multi-step tasks over ten simulated APIs (calendars, org chart, restaurant reviews, rides,
food and grocery delivery, payments, music) either by calling the APIs as tools one at a
time or by writing code that calls them.

The benchmark compares four ways of giving an agent the same APIs: tool calls only, a
stateless code-execution tool (PTC), a persistent notebook-style cell with API inspection
and doc learning (I-PTC), and a provider's native code-execution sandbox.

## Benchmark

| Split | File | Tasks | Description |
|---|---|---|---|
| Main | `ptc_eval/data/main.jsonl` | 100 | Single-turn tasks |
| Error recovery | `ptc_eval/data/error_recovery.jsonl` | 70 | Single-turn tasks in which API calls fail at scheduled points; the agent has to notice and recover |
| Multi-turn | `ptc_eval/data/multi_turn.jsonl` | 70 | Two-turn tasks: a follow-up user turn refines the first request |

The splits are also on Hugging Face: [`shuo-yuan/PTC-Bench`](https://huggingface.co/datasets/shuo-yuan/PTC-Bench).

The paper's results were produced with v1.0. Later releases correct main-split tasks whose
golden answers or verifiers contradicted their prompts; see [`CHANGELOG.md`](CHANGELOG.md).

Each file is JSONL (one task per line) with the fields `id`, `question` (the user turns),
`involved_classes` (the APIs the task uses) and `initial_config` (the initial state of each
API); error-recovery tasks may also carry a `flakiness_config`. Every task is scored by a
verifier in `ptc_eval/eval_checker/verifiers/`, selected by task `id`.

### Simulated APIs

| API | Public methods | Description |
|---|---|---|
| `EnterpriseCalendarAPI` | 5 | Company calendar; employees and events in their own timezones |
| `PersonalCalendarAPI` | 9 | Personal calendar events and schedules |
| `OrgChartAPI` | 2 | Org chart with reporting history |
| `YelpAPI` | 22 | Restaurant search, reviews, bookmarks, distances |
| `GoogleMapsAPI` | 1 | Address geocoding |
| `UberAPI` | 7 | Ride cost and time estimates |
| `GroceryDeliveryAPI` | 4 | Store inventories, delivery fees, gift cards |
| `DoorDashAPI` | 8 | Food delivery |
| `VenmoAPI` | 9 | Payments and transactions |
| `SpotifyAPI` | 11 | Tracks, artists, playlists |

All API state is synthetic. Implementations are in
`ptc_eval/eval_checker/multi_turn_eval/func_source_code/` and the method documentation shown
to agents is in `ptc_eval/data/multi_turn_func_doc/`.

## Setup

Tested with Python 3.12.

```bash
git clone https://github.com/cck0517/I-PTC.git && cd I-PTC
pip install -r agent_eval/requirements.txt
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

python -m agent_eval.run_eval --backend openai --model $MODEL --mode tool_only --workers 10
python -m agent_eval.run_eval --backend openai --model $MODEL --mode ptc --workers 10
python -m agent_eval.run_eval --backend openai --model $MODEL --mode iptc --workers 10

python -m agent_eval.run_eval --backend claude_ptc --model claude-sonnet-4-6 --workers 10
```

The main split is the default. For the other splits, pass the file, and add `--multi-turn`
for the multi-turn split:

```bash
# Error recovery
python -m agent_eval.run_eval ... --queries-file ptc_eval/data/error_recovery.jsonl

# Multi-turn
python -m agent_eval.run_eval ... --queries-file ptc_eval/data/multi_turn.jsonl --multi-turn
```

Each run writes to `agent_eval/traces/<model>_<mode>_<timestamp>/` unless you pass
`--output-dir`. Run directories and traces record the internal mode name: `ptc_encouraged`
for `ptc`, and `ptc_cell_encouraged` for `raw_iptc` and `iptc` (the trace's `run_config`
also records `tool_inspection`).

### Evaluating

```bash
# Main (default queries and answers)
python -m agent_eval.evaluate_traces --traces-dir <run_dir>/

# Error recovery
python -m agent_eval.evaluate_traces --traces-dir <run_dir>/ \
    --queries-file ptc_eval/data/error_recovery.jsonl \
    --gt-file ptc_eval/data/possible_answer/error_recovery.jsonl

# Multi-turn (verifiers compute the answers; there is no answer file)
python -m agent_eval.evaluate_traces --traces-dir <run_dir>/ \
    --queries-file ptc_eval/data/multi_turn.jsonl
```

This writes `<run_dir>/eval_results.json`. `--summary-csv <file> --run-label <label>` also
appends a one-line summary to a CSV, which is convenient for aggregating repeated runs.

### Other options

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

`python -m agent_eval.run_eval --help` lists everything.

## Output

```
<run_dir>/
├── <task_id>.json        # full trace: system prompt, turns, tool calls, tokens, latencies, final answer, final API state
├── summary.json          # run configuration and per-task counts (turns, tool calls, tokens, wall time, stop reason)
├── all_traces.json       # all traces in one file (runs of 30 tasks or fewer)
├── learned_docs.json     # tool descriptions learned during the run (I-PTC)
└── eval_results.json     # written by evaluate_traces
```

`eval_results.json` contains `pass_rate`, `tasks_passed`, `tasks_evaluated`, aggregate metrics
(tool calls, API calls and errors, total and trajectory tokens, wall time) and a `per_task`
list with each task's `passed` flag, `score`, per-check results and the same metrics.

## Repository layout

```
agent_eval/
├── run_eval.py           # runner CLI
├── agent.py              # agent loop and backends
├── ptc_tool.py           # ptc_execute / ptc_cell tools, API tool definitions, prompts
├── doc_learner.py        # doc learning (--learning-batch-size)
├── evaluate_traces.py    # scoring CLI
└── trajectory_tokens.py  # trajectory-token accounting
ptc_eval/
├── data/                 # the three splits, answer files, API method docs
├── eval_checker/
│   ├── multi_turn_eval/  # simulated API implementations
│   └── verifiers/        # per-task verifiers
└── constants/            # API registry and paths
```

## License

Code and data are released under the Apache License 2.0.

## Citation

<!-- TODO: add the BibTeX entry for the paper -->

## Acknowledgements

The data format and directory layout follow the
[Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla).
