# PTC-Bench

PTC-Bench evaluates LLM agents' **programmatic tool calling (PTC)**. Agents solve
multi-step tasks over ten simulated APIs either by calling the APIs as tools one at a
time or by writing code that calls them. This folder contains the dataset, simulated
environments, task verifiers, and trace-scoring tools.

## Dataset

| Split | File | Tasks | Description |
|---|---|---|---|
| Main | [`data/main.jsonl`](data/main.jsonl) | 100 | Single-turn tasks |
| Error recovery | [`data/error_recovery.jsonl`](data/error_recovery.jsonl) | 70 | Single-turn tasks in which API calls fail at scheduled points; the agent has to notice and recover |
| Multi-turn | [`data/multi_turn.jsonl`](data/multi_turn.jsonl) | 70 | Two-turn tasks: a follow-up user turn refines the first request |

The splits are bundled in this repository and are also on Hugging Face:
[`shuo-yuan/PTC-Bench`](https://huggingface.co/datasets/shuo-yuan/PTC-Bench).

Each file is JSONL (one task per line) with the fields `id`, `question` (the user turns),
`involved_classes` (the APIs the task uses), and `initial_config` (the initial state of each
API); error-recovery tasks may also carry a `flakiness_config`. Every task is scored by a
verifier in [`eval_checker/verifiers/`](eval_checker/verifiers/), selected by task `id`.
Main and error-recovery answer files are in [`data/possible_answer/`](data/possible_answer/).
Multi-turn answers are computed by verifiers; there is no multi-turn answer file.

## Simulated APIs

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
[`eval_checker/multi_turn_eval/func_source_code/`](eval_checker/multi_turn_eval/func_source_code/)
and the method documentation shown to agents is in
[`data/multi_turn_func_doc/`](data/multi_turn_func_doc/).

## Scoring traces

See the [experiments guide](../experiments/README.md) to generate traces with the supplied
runner. Run the following commands from the repository root:

```bash
# Main (default queries and answers)
python -m ptc_bench.evaluate_traces --traces-dir <run_dir>/

# Error recovery
python -m ptc_bench.evaluate_traces --traces-dir <run_dir>/ \
    --queries-file ptc_bench/data/error_recovery.jsonl \
    --gt-file ptc_bench/data/possible_answer/error_recovery.jsonl

# Multi-turn (verifiers compute the answers; there is no answer file)
python -m ptc_bench.evaluate_traces --traces-dir <run_dir>/ \
    --queries-file ptc_bench/data/multi_turn.jsonl
```

This writes `<run_dir>/eval_results.json`. `--summary-csv <file> --run-label <label>` also
appends a one-line summary to a CSV, which is convenient for aggregating repeated runs.

`eval_results.json` contains `pass_rate`, `tasks_passed`, `tasks_evaluated`, aggregate metrics
(tool calls, API calls and errors, total and trajectory tokens, wall time), and a `per_task`
list with each task's `passed` flag, `score`, per-check results, and the same metrics.

## Code and data layout

```text
ptc_bench/
├── data/                 # The three splits, answer files, API method docs
├── eval_checker/
│   ├── multi_turn_eval/  # Simulated API implementations
│   └── verifiers/        # Per-task verifiers
├── constants/            # API registry and paths
├── utils.py              # Dataset and answer loaders
├── evaluate_traces.py    # Scoring CLI
└── trajectory_tokens.py  # Trajectory-token accounting
```
