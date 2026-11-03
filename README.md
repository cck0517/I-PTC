# [NeurIPS 2026] I-PTC and PTC-Bench

This repository releases two contributions from the same paper:

- **[I-PTC](iptc/README.md):** a programmatic tool-calling framework implementation with
  persistent Python cells, on-demand API inspection, and self-evolving API documentation.
- **[PTC-Bench](ptc_bench/README.md):** a benchmark for evaluating LLM agents' programmatic
  tool calling, with 240 tasks, ten simulated APIs, and per-task verifiers.

The **[experiments guide](experiments/README.md)** connects the framework and benchmark
to compare tool-only agents, stateless PTC, I-PTC, and provider-native code execution.

## Repository layout

```text
I-PTC/
├── iptc/           # Framework: execution, API inspection, doc learning, agent backends
├── ptc_bench/      # Dataset, simulated APIs, verifiers, scoring
├── experiments/    # Runners, comparison conditions, paper reproduction
├── tests/          # Offline regression tests
└── requirements.txt
```

Each component has its own README. The framework implementation and experiment runners
use the benchmark's API definitions; the folders organize one paper release in one repo.

## Quick start

Tested with Python 3.12. Run commands from the repository root.

```bash
git clone https://github.com/cck0517/I-PTC.git && cd I-PTC
pip install -r requirements.txt
```

**Security.** In the PTC and I-PTC conditions, the model's Python code runs with `exec()`
inside the harness process, with full builtins. It can read files and environment variables
(including your API keys) and reach the network. The tool descriptions call this a sandbox
from the model's point of view, but it is not an isolation boundary. Run the harness only in
a disposable container or VM that holds nothing beyond the API key the run needs.

Set `OPENAI_API_KEY`, then run I-PTC on the main split and score the saved traces:

```bash
python -m experiments.run_eval --backend openai --model gpt-5.4-mini --mode iptc \
    --workers 10 --output-dir experiments/traces/quickstart
python -m ptc_bench.evaluate_traces --traces-dir experiments/traces/quickstart
```

See the [experiments guide](experiments/README.md) for other providers, comparison
conditions, split selection, and run options. See the [benchmark guide](ptc_bench/README.md)
for the task format, simulated APIs, and scoring.

## Tests

The unit tests need no API keys or network access:

```bash
python -m unittest discover -s tests -t .
```

## License

Code and data are released under the Apache License 2.0.

## Citation

The paper has been accepted to the NeurIPS main conference; proceedings are forthcoming.

If you use I-PTC or PTC-Bench in your research, please cite:

```bibtex
@inproceedings{mao2026iptc,
  title = {{I-PTC}: Interactive Programmatic Tool Calling for Stateful Tool-Augmented Agents},
  author = {Mao, Huanzhi and Cao, Chengkun and Yuan, Shuo and Gonzalez, Joseph E.},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  note = {To appear},
  url = {https://github.com/cck0517/I-PTC}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgements

The data format and directory layout follow the
[Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla).
