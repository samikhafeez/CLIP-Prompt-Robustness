# Contributing

Keep the original notebook and its stored outputs intact when making documentation changes. For new experiments, create a separate notebook or script and record the library versions, hardware, gallery sample and prompt sets used.

Use a topic branch, stage only the intended changes, then run `python3 tools/check_repository.py` from the repository root. This is a static check and does not execute the experiments.

Keep recordings, reports with personal details, environments, caches and credentials out of commits. Describe actual validation in pull requests and distinguish reproduced results from historical outputs. See the [reproduction notes](docs/REPRODUCIBILITY.md) before extending the analysis.
