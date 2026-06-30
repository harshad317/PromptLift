.PHONY: test compile run-toy-mvp run-toy-closed-loop check-benchmark-readiness run-openai-fixed-pool run-openai-causal-pilot run-openai-closed-loop run-openai-composability run-openai-cross-model-transfer write-budget-pareto-report clean-artifacts

PYTHON ?= python3
CONFIG ?= configs/toy_schemaevo.yaml
OUT ?= artifacts/openai_fixed_pool
HOTPOTQA ?=
HOVER ?=
DATASET ?= hotpotqa
TRAIN ?=
SELECTION ?=
CONFIRMATION ?=
SMOKE ?=
HELDOUT ?=
MODEL ?= gpt-4.1-mini
USE_TIKTOKEN ?=
INPUT_PRICE_PER_MILLION ?=
OUTPUT_PRICE_PER_MILLION ?=
CACHED_INPUT_PRICE_PER_MILLION ?=
PRICE_SOURCE_DATE ?= cli
MAX_TARGET_TASK_CALLS ?=
MAX_PROMPT_TOKENS ?=
MAX_COMPLETION_TOKENS ?=
MAX_TOTAL_TOKENS ?=
MAX_DOLLAR_COST ?=
WORKERS ?=
PROGRESS ?=
OPTIMIZER ?=
PROMPT_OPTIMIZER_COMMAND ?=
PROMPT_OPTIMIZER_NAME ?= external_prompt_optimizer
EVAL ?=
SOURCE_MODEL ?= gpt-4.1-mini
TARGET_MODEL ?=
BUDGET_RUNS ?=

compile:
	$(PYTHON) -m compileall schemaevo tests

test:
	$(PYTHON) -m pytest

run-toy-mvp:
	$(PYTHON) -m schemaevo.cli run-toy-mvp --config configs/toy_schemaevo.yaml --out artifacts/toy_mvp $(if $(WORKERS),--workers $(WORKERS),) $(if $(PROGRESS),--progress $(PROGRESS),)

run-toy-closed-loop:
	$(PYTHON) -m schemaevo.cli run-toy-closed-loop --config configs/toy_schemaevo.yaml --out artifacts/toy_closed_loop $(if $(PROGRESS),--progress $(PROGRESS),)

check-benchmark-readiness:
	$(PYTHON) -m schemaevo.cli check-benchmark-readiness $(if $(HOTPOTQA),--hotpotqa $(HOTPOTQA),) $(if $(HOVER),--hover $(HOVER),)

run-openai-fixed-pool:
	$(PYTHON) -m schemaevo.cli run-openai-fixed-pool \
		--config $(CONFIG) \
		--dataset $(DATASET) \
		--train $(TRAIN) \
		--selection $(SELECTION) \
		--confirmation $(CONFIRMATION) \
		$(if $(SMOKE),--smoke $(SMOKE),) \
		$(if $(HELDOUT),--heldout $(HELDOUT),) \
		--model $(MODEL) \
		$(if $(USE_TIKTOKEN),--use-tiktoken-costing,) \
		$(if $(INPUT_PRICE_PER_MILLION),--input-price-per-million $(INPUT_PRICE_PER_MILLION),) \
		$(if $(OUTPUT_PRICE_PER_MILLION),--output-price-per-million $(OUTPUT_PRICE_PER_MILLION),) \
		$(if $(CACHED_INPUT_PRICE_PER_MILLION),--cached-input-price-per-million $(CACHED_INPUT_PRICE_PER_MILLION),) \
		$(if $(PRICE_SOURCE_DATE),--price-source-date $(PRICE_SOURCE_DATE),) \
		$(if $(MAX_TARGET_TASK_CALLS),--max-target-task-calls $(MAX_TARGET_TASK_CALLS),) \
		$(if $(MAX_PROMPT_TOKENS),--max-prompt-tokens $(MAX_PROMPT_TOKENS),) \
		$(if $(MAX_COMPLETION_TOKENS),--max-completion-tokens $(MAX_COMPLETION_TOKENS),) \
		$(if $(MAX_TOTAL_TOKENS),--max-total-tokens $(MAX_TOTAL_TOKENS),) \
		$(if $(MAX_DOLLAR_COST),--max-dollar-cost $(MAX_DOLLAR_COST),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(PROGRESS),--progress $(PROGRESS),) \
		--out $(OUT)

run-openai-causal-pilot:
	$(PYTHON) -m schemaevo.cli run-openai-causal-pilot \
		--config $(CONFIG) \
		--dataset $(DATASET) \
		--train $(TRAIN) \
		--selection $(SELECTION) \
		--confirmation $(CONFIRMATION) \
		$(if $(SMOKE),--smoke $(SMOKE),) \
		$(if $(HELDOUT),--heldout $(HELDOUT),) \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(USE_TIKTOKEN),--use-tiktoken-costing,) \
		$(if $(INPUT_PRICE_PER_MILLION),--input-price-per-million $(INPUT_PRICE_PER_MILLION),) \
		$(if $(OUTPUT_PRICE_PER_MILLION),--output-price-per-million $(OUTPUT_PRICE_PER_MILLION),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(PROGRESS),--progress $(PROGRESS),) \
		--out $(OUT)

run-openai-closed-loop:
	$(PYTHON) -m schemaevo.cli run-openai-closed-loop \
		--config $(CONFIG) \
		--dataset $(DATASET) \
		--optimizer $(OPTIMIZER) \
		--confirmation $(CONFIRMATION) \
		$(if $(HELDOUT),--heldout $(HELDOUT),) \
		--model $(MODEL) \
		$(if $(USE_TIKTOKEN),--use-tiktoken-costing,) \
		$(if $(INPUT_PRICE_PER_MILLION),--input-price-per-million $(INPUT_PRICE_PER_MILLION),) \
		$(if $(OUTPUT_PRICE_PER_MILLION),--output-price-per-million $(OUTPUT_PRICE_PER_MILLION),) \
		$(if $(PROGRESS),--progress $(PROGRESS),) \
		--out $(OUT)

run-openai-composability:
	$(PYTHON) -m schemaevo.cli run-openai-composability \
		--config $(CONFIG) \
		--dataset $(DATASET) \
		--schema-optimizer $(OPTIMIZER) \
		--eval $(EVAL) \
		--model $(MODEL) \
		--prompt-optimizer-name $(PROMPT_OPTIMIZER_NAME) \
		--prompt-optimizer-command '$(PROMPT_OPTIMIZER_COMMAND)' \
		$(if $(USE_TIKTOKEN),--use-tiktoken-costing,) \
		$(if $(INPUT_PRICE_PER_MILLION),--input-price-per-million $(INPUT_PRICE_PER_MILLION),) \
		$(if $(OUTPUT_PRICE_PER_MILLION),--output-price-per-million $(OUTPUT_PRICE_PER_MILLION),) \
		$(if $(PROGRESS),--progress $(PROGRESS),) \
		--out $(OUT)

run-openai-cross-model-transfer:
	$(PYTHON) -m schemaevo.cli run-openai-cross-model-transfer \
		--config $(CONFIG) \
		--dataset $(DATASET) \
		--train $(TRAIN) \
		--selection $(SELECTION) \
		--confirmation $(CONFIRMATION) \
		$(if $(SMOKE),--smoke $(SMOKE),) \
		$(if $(HELDOUT),--heldout $(HELDOUT),) \
		--source-model $(SOURCE_MODEL) \
		--target-model $(TARGET_MODEL) \
		$(if $(USE_TIKTOKEN),--use-tiktoken-costing,) \
		$(if $(INPUT_PRICE_PER_MILLION),--input-price-per-million $(INPUT_PRICE_PER_MILLION),) \
		$(if $(OUTPUT_PRICE_PER_MILLION),--output-price-per-million $(OUTPUT_PRICE_PER_MILLION),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		$(if $(PROGRESS),--progress $(PROGRESS),) \
		--out $(OUT)

write-budget-pareto-report:
	$(PYTHON) -m schemaevo.cli write-budget-pareto-report $(BUDGET_RUNS) --out $(OUT)

clean-artifacts:
	rm -rf artifacts/toy_mvp
