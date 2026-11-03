# I-PTC framework

I-PTC combines three mechanisms for programmatic tool calling:

- **Persistent execution:** `ptc_cell` runs Python in a notebook-style namespace that
  persists across cells within a task.
- **On-demand API inspection:** agents discover API methods inside cells with `<api>??`
  and inspect method documentation with `<api>.<method>??`.
- **Self-evolving API documentation:** a learner reads API errors between task batches
  and appends corrections to tool descriptions for subsequent tasks.

This folder contains the framework implementation used in the paper's experiments.
Its execution bindings and tool descriptions use the simulated APIs in
[`ptc_bench/`](../ptc_bench/README.md).

## Code

| File | Responsibility |
|---|---|
| [`ptc_tool.py`](ptc_tool.py) | Stateless `ptc_execute`, persistent `ptc_cell`, API inspection, tool definitions, and prompts |
| [`doc_learner.py`](doc_learner.py) | Collect error patterns and learn documentation addenda between batches |
| [`agent.py`](agent.py) | Agent loop, provider backends, and shared tool/trace types |

The experiment runner creates a fresh executor for each benchmark task, attaches API
objects and tool definitions, and applies learned documentation between batches.
Stateless PTC, persistent execution alone, inspection, and documentation learning can
also be evaluated separately.

## Run I-PTC

Follow the [root setup instructions](../README.md#quick-start), set the backend's API key,
and run from the repository root:

```bash
python -m experiments.run_eval --backend openai --model gpt-5.4-mini --mode iptc
```

`--mode iptc` enables persistent cells, API inspection, and documentation learning after
every 10 tasks, with the run's model as the learner. Explicit flags override this preset.
The [experiments guide](../experiments/README.md#conditions) describes all conditions,
component ablations, learner providers, and saved documentation.

**Execution security:** generated Python runs with `exec()` in the harness process,
with full builtins, filesystem access, environment access, and network access. Run only
in a disposable container or VM holding nothing beyond the API key the run needs;
see the [root security note](../README.md#quick-start).
