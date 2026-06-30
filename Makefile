.PHONY: test compile run-toy-mvp run-toy-closed-loop check-benchmark-readiness run-openai-fixed-pool clean-artifacts

PYTHON ?= python3
HOTPOTQA ?=
HOVER ?=
DATASET ?= hotpotqa
TRAIN ?=
SELECTION ?=
CONFIRMATION ?=
SMOKE ?=
HELDOUT ?=
MODEL ?= gpt-4.1-mini

compile:
	$(PYTHON) -m compileall schemaevo tests

test:
	$(PYTHON) -m pytest

run-toy-mvp:
	$(PYTHON) -m schemaevo.cli run-toy-mvp --config configs/toy_schemaevo.yaml --out artifacts/toy_mvp

run-toy-closed-loop:
	$(PYTHON) -m schemaevo.cli run-toy-closed-loop --config configs/toy_schemaevo.yaml --out artifacts/toy_closed_loop

check-benchmark-readiness:
	$(PYTHON) -m schemaevo.cli check-benchmark-readiness $(if $(HOTPOTQA),--hotpotqa $(HOTPOTQA),) $(if $(HOVER),--hover $(HOVER),)

run-openai-fixed-pool:
	$(PYTHON) -m schemaevo.cli run-openai-fixed-pool --dataset $(DATASET) --train $(TRAIN) --selection $(SELECTION) --confirmation $(CONFIRMATION) $(if $(SMOKE),--smoke $(SMOKE),) $(if $(HELDOUT),--heldout $(HELDOUT),) --model $(MODEL) --out artifacts/openai_fixed_pool

clean-artifacts:
	rm -rf artifacts/toy_mvp
