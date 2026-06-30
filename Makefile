.PHONY: test compile run-toy-mvp run-toy-closed-loop clean-artifacts

PYTHON ?= python3

compile:
	$(PYTHON) -m compileall schemaevo tests

test:
	$(PYTHON) -m pytest

run-toy-mvp:
	$(PYTHON) -m schemaevo.cli run-toy-mvp --config configs/toy_schemaevo.yaml --out artifacts/toy_mvp

run-toy-closed-loop:
	$(PYTHON) -m schemaevo.cli run-toy-closed-loop --config configs/toy_schemaevo.yaml --out artifacts/toy_closed_loop

clean-artifacts:
	rm -rf artifacts/toy_mvp
