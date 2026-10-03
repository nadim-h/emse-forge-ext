PY    ?= $(CURDIR)/.venv/bin/python
# MODEL=<slug> overrides EMSE_MODEL_PROFILE in .env; every model stage uses it.
TASK   = $(if $(MODEL),EMSE_MODEL_PROFILE=$(MODEL)) $(PY)

.PHONY: all tests \
        testgen bugfix refactor ppl-score report serve

## Every stage; the tasks need `make serve` running.
all: tests testgen bugfix refactor report

## Unpack the committed tests (xz, split under GitHub's file limit).
tests:
	set -e; for l in cpp java; do cat data/problems/$${l}_tests.jsonl.xz.* | xz -d > data/problems/$${l}_tests.jsonl; done

## Task A: generate tests, score them by mutation and coverage.
testgen:
	$(TASK) -m pipeline.testgen generate
	$(TASK) -m pipeline.testgen evaluate

## Task B: seed faults (shared by all models), repair, judge.
bugfix:
	$(PY) -m pipeline.bugfix seed
	$(TASK) -m pipeline.bugfix repair
	$(TASK) -m pipeline.bugfix verify

## Task C: refactor, judge behavior and score CH.
refactor:
	$(TASK) -m pipeline.refactor generate
	$(TASK) -m pipeline.refactor evaluate

## Perplexity under the model (GPU and its weights).
ppl-score:
	$(TASK) -m pipeline.perplexity

## Every paper table and number (printed) and figure (figures/).
report:
	$(PY) -m pipeline.rq1_perplexity
	$(PY) -m pipeline.rq2_outcomes
	$(PY) -m pipeline.rq3_predictors

## vLLM for the active model; its flags are in pipeline/config.py.
serve:
	$(TASK) -m vllm.entrypoints.openai.api_server \
	  --model $$($(TASK) -m pipeline.config model-name) --host 127.0.0.1 --port 8123 \
	  $$($(TASK) -m pipeline.config serve-args)
