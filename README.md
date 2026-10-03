# Replication package

Code, data and analysis for the EMSE extension of the FORGE 2026 paper *Code
for Machines, Not Just Humans: Quantifying AI-Friendliness with Code Health
Metrics*.

`make report` prints every table and number in the paper and saves every
figure, all from the committed data.

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | One module per stage |
| `prompts/` | The three task prompts |
| `data/problems/` | Problem descriptions, checkers and tests (`make tests` unpacks them); difficulty ratings |
| `data/dataset/` | The corpus: source, corpus CH (used for the 50/50 split), SLOC and number of tests of each solution |
| `data/codehealth/` | CodeScene CLI v1.0.33 review of every solution |
| `data/bugfix/` | The seeded faults, shared by every model |
| `data/perplexity/` | Perplexity of every solution under each model |
| `data/{testgen,bugfix,refactor}_<model>/` | Each model's task outputs |
| `figures/` | Every figure |

## Pipeline

| Stage | Make target | Module |
|---|---|---|
| Unpack the problems' tests | `tests` | |
| Task A: tests, mutation score, coverage | `testgen` | `testgen` |
| Task B: seed faults, repair, judge | `bugfix` | `bugfix`, `oracle`, `checker` |
| Task C: refactor, judge, score CH | `refactor` | `refactor`, `oracle`, `checker` |
| Perplexity | `ppl-score` | `perplexity` |
| Tables and figures | `report` | `rq1_perplexity`, `rq2_outcomes`, `rq3_predictors` |

The pipeline starts from the corpus in `data/dataset/`.

The tasks need a model served with `make serve MODEL=<slug>`, where the slug
is `qwen3.6-35b`, `gemma-4-26b` or `nemotron-3-nano-30b-a3b`. Every model
config is in `pipeline/config.py`.
