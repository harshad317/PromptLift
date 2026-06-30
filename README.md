# SchemaEvo

Fixed-call schema evolution for multi-module LLM prompt programs.

This repository implements the prompt-optimization method from the supplied implementation plan: evolve typed intermediate schema contracts, validators, and downstream consumption rules while preserving the deployment call graph.

## Visual Overview

SchemaEvo changes the typed intermediate contracts between existing program modules. The optimizer can improve what modules emit and consume, but it does not add LLM calls, retrieval calls, tools, demos, or test-label access.

![SchemaEvo fixed-call architecture](docs/images/schemaevo-architecture.svg)

The fixed-pool path evaluates a frozen set of schema candidates. The closed-loop path keeps the same deployment boundary, then searches with legal mutations, shared minibatches, budget gates, and Pareto tracking.

![SchemaEvo optimizer loop](docs/images/schemaevo-loop.svg)

## Scope

Implemented:

- One-page SOTA pre-registration in `PRE_REGISTRATION.md`.
- Schema candidate representation, token-budget checks, legal mutation grammar, and forbidden mutation guardrails.
- Human semantic templates, random same-capacity controls, and train-only trace-based schema proposals.
- Optional OpenAI reflective schema proposer using `gpt-4.1-mini` and Structured Outputs.
- Failure-tolerant proposal parsing: malformed OpenAI fields/schemas are dropped and recorded instead of aborting pool construction.
- Schema compiler that patches module output signatures and inserts schema contracts into frozen prompts.
- Automated call-graph invariants for module order, LLM-call count, retriever-call count, retrieval top-k, and fixed few-shot/demo IDs.
- Pydantic-backed local validation plus executable constraints (`non_empty`, `regex`, numeric ranges, max tokens/items) and known HoVer/HotpotQA object-field shape checks with deterministic coercion only; no uncounted LLM repair.
- Per-call logging, rollout cache, cost ledger structures, configurable provider prices, optional `tiktoken` accounting, hard token/call/dollar budget caps, per-example prediction/score rows, latency/token accounting, bootstrap LCB selection, paired bootstrap, approximate randomization, multiple-comparison correction, and field masking/blanking/shuffling/downstream-disabled ablations.
- Generic closed-loop SchemaEvo optimizer with population initialization, all grammar operators sampled, UCB-style parent/operator allocation, optional successive-halving promotion, paired minibatch evaluation per rung, significance-aware Pareto tracking, rejected-schema audit records, and final candidate selection.
- Local HotpotQA/HoVer JSON/JSONL loaders and scorers, DSPy and OpenAI Responses API adapters to `LMProgram`, a benchmark readiness preflight, and a composability harness that can run an external prompt optimizer first and then run SchemaEvo as the additive schema layer.
- Phase-report drivers for causal pilot go/no-go, real-data closed-loop, external GEPA/MIPRO-style composability, budget/Pareto aggregation, cross-model schema transfer, and deployment-cost invariance.
- A local deterministic toy multi-hop program that exercises the full fixed-pool SchemaEvo path without external APIs or benchmark data.

Intentionally not implemented yet:

- GEPA/GEPA+Merge reproduction.
- Published HotpotQA/HoVer benchmark evidence from downloaded data and real model credentials.
- Competing prompt optimization methods or benchmark baselines such as MIPROv2, CAPO, ADOPT, Trace/Opto, PCO, BO, Hyperband, or prompt-only GEPA.

Those are experimental/benchmarking tasks. The repository now has adapters/loaders/harnesses needed
to wire them, but it does not claim benchmark evidence until those external runs are configured and
executed.

## Quick Start

```bash
python3 -m pytest -q
python3 -m schemaevo.cli run-toy-mvp --config configs/toy_schemaevo.yaml --workers 2 --progress rich --out artifacts/toy_mvp
python3 -m schemaevo.cli run-toy-closed-loop --config configs/toy_schemaevo.yaml --out artifacts/toy_closed_loop
```

Or:

```bash
make test
make run-toy-mvp
make run-toy-closed-loop
make check-benchmark-readiness HOTPOTQA=/path/to/hotpot.json HOVER=/path/to/hover.jsonl
make run-openai-fixed-pool \
  CONFIG=configs/mvp_hotpotqa_gpt41mini.yaml \
  DATASET=hotpotqa \
  TRAIN=/path/to/hotpot_train.json \
  SELECTION=/path/to/hotpot_dev_selection.json \
  CONFIRMATION=/path/to/hotpot_dev_confirmation.json \
  MODEL=gpt-4.1-mini \
  OUT=artifacts/openai_fixed_pool_hotpotqa
```

The toy run writes a frozen schema pool, per-call logs, per-example prediction/score rows with call/token/dollar/latency counters, payloads, cost ledgers, and `artifacts/toy_mvp/results/mvp_summary.json`.
Fixed-pool candidate batches support process-level parallelism with `--workers N` or `WORKERS=N`.
Progress rendering is controlled with `--progress {auto,rich,tqdm,none}` or `PROGRESS=...`; progress
is written to stderr so stdout remains JSON. `auto` uses Rich on interactive terminals and disables
itself in non-TTY automation. Runs with hard budget caps stay serial to preserve exact budget gating.

To use the real LLM proposer, install the API extra, set `OPENAI_API_KEY`, and switch the config:

```bash
python3 -m pip install -e '.[api]'
```

```yaml
proposer:
  kind: openai
  model: gpt-4.1-mini
```

By default token counting uses a deterministic local proxy so tests and toy runs never perform
network-backed tokenizer setup. For production accounting with `tiktoken`, install the API extra and
set:

```bash
export SCHEMAEVO_USE_TIKTOKEN=1
```

Rollouts are cached under each artifact directory using a fingerprint of the compiled program,
schema, example ID, seed, and intervention ID.

For real benchmark preflight:

```bash
python3 -m schemaevo.cli check-benchmark-readiness \
  --hotpotqa /path/to/hotpot.json \
  --hover /path/to/hover.jsonl \
  --strict
```

This validates `OPENAI_API_KEY`, optional API packages, and local dataset readability. It does not
download datasets or claim benchmark results.

By default the preflight requires non-empty context fields for HotpotQA/HoVer so a question/answer-only
file cannot accidentally be treated as benchmark evidence. For a pipeline smoke run on contextless
data, pass `--allow-contextless`. `run-openai-fixed-pool` performs the same quality checks on every
supplied runtime split (`train`, optional `smoke`, `selection`, `confirmation`, and optional
`heldout`) before launching model calls, and rejects overlapping example IDs across those splits.

Once readiness passes, a local-data OpenAI fixed-pool run can be launched with:

```bash
python3 -m schemaevo.cli run-openai-fixed-pool \
  --config configs/mvp_hotpotqa_gpt41mini.yaml \
  --dataset hotpotqa \
  --train /path/to/hotpot_train.json \
  --selection /path/to/hotpot_dev_selection.json \
  --confirmation /path/to/hotpot_dev_confirmation.json \
  --heldout /path/to/hotpot_test.json \
  --use-tiktoken-costing \
  --input-price-per-million <input_price> \
  --output-price-per-million <output_price> \
  --max-target-task-calls <call_budget> \
  --model gpt-4.1-mini \
  --out artifacts/openai_fixed_pool
```

For a combined HotpotQA or HoVer JSON with top-level `train`, `smoke`, `selection`, `confirmation`,
`heldout_validation`, and/or `test` keys, pass the same file to each split flag; the loaders map
runtime splits to the matching source section. Prices are intentionally not hardcoded; set them from
the provider price sheet you want the artifact to represent. Strict OpenAI fixed-pool runs require
`tiktoken` costing plus nonzero input and output prices for the selected model; pass `--allow-unready`
only for dry development or injected-client tests. Fixed-pool runs
also accept `--max-target-task-calls`, `--max-prompt-tokens`, `--max-completion-tokens`,
`--max-total-tokens`, and `--max-dollar-cost`; the optimizer trims optional candidate/ablation work
while reserving enough budget for the paired confirmation gate. Token and dollar gates use a
preflight estimate from module prompts, example inputs, configured tokenizer/prices, and declared
`max_output_tokens` before starting additional candidate evaluations.
When an OpenAI schema proposer is configured, its proposal call usage is captured separately as
`proposal_usage` and included in fixed-pool `cost_summary`/`budget` totals.

The repository includes MVP config templates for the first real-data runs:

- `configs/mvp_hotpotqa_gpt41mini.yaml`
- `configs/mvp_hover_gpt41mini.yaml`

This uses OpenAI module runners and the fixed-pool SchemaEvo path. It still does not run or compare
GEPA/MIPRO; those remain external prompt optimizers for the composability harness.

## Phase Drivers

Phase 0 pre-registration:

```bash
cat PRE_REGISTRATION.md
```

Causal pilot with automatic mask/shuffle go/no-go and deployment invariance reports:

```bash
python3 -m schemaevo.cli run-openai-causal-pilot \
  --config configs/mvp_hotpotqa_gpt41mini.yaml \
  --dataset hotpotqa \
  --train /path/to/hotpot_train.json \
  --smoke /path/to/hotpot_smoke.json \
  --selection /path/to/hotpot_selection.json \
  --confirmation /path/to/hotpot_confirmation.json \
  --heldout /path/to/hotpot_heldout.json \
  --model gpt-4.1-mini \
  --use-tiktoken-costing \
  --input-price-per-million <input_price> \
  --output-price-per-million <output_price> \
  --price-source-date <price_sheet_date> \
  --workers 4 \
  --out artifacts/causal_pilot_hotpotqa
```

Real-data closed-loop on HotpotQA or HoVer:

```bash
python3 -m schemaevo.cli run-openai-closed-loop \
  --config configs/toy_schemaevo.yaml \
  --dataset hover \
  --optimizer /path/to/hover_selection.json \
  --confirmation /path/to/hover_confirmation.json \
  --heldout /path/to/hover_heldout.json \
  --model gpt-4.1-mini \
  --use-tiktoken-costing \
  --input-price-per-million <input_price> \
  --output-price-per-million <output_price> \
  --out artifacts/openai_closed_loop_hover
```

External GEPA/MIPRO-style composability. The external command receives
`SCHEMAEVO_INPUT_PROGRAM` and must write `SCHEMAEVO_OUTPUT_PROGRAM` with patched module prompts:

```bash
python3 -m schemaevo.cli run-openai-composability \
  --config configs/toy_schemaevo.yaml \
  --dataset hotpotqa \
  --schema-optimizer /path/to/hotpot_selection.json \
  --eval /path/to/hotpot_confirmation.json \
  --model gpt-4.1-mini \
  --prompt-optimizer-name gepa \
  --prompt-optimizer-command "python /path/to/run_gepa_bridge.py" \
  --use-tiktoken-costing \
  --input-price-per-million <input_price> \
  --output-price-per-million <output_price> \
  --out artifacts/gepa_plus_schemaevo_hotpotqa
```

Cross-model schema transfer:

```bash
python3 -m schemaevo.cli run-openai-cross-model-transfer \
  --config configs/mvp_hotpotqa_gpt41mini.yaml \
  --dataset hotpotqa \
  --train /path/to/hotpot_train.json \
  --selection /path/to/hotpot_selection.json \
  --confirmation /path/to/hotpot_confirmation.json \
  --source-model gpt-4.1-mini \
  --target-model <target_model> \
  --use-tiktoken-costing \
  --input-price-per-million <input_price> \
  --output-price-per-million <output_price> \
  --out artifacts/cross_model_transfer_hotpotqa
```

Budget/Pareto aggregation from completed run summaries:

```bash
python3 -m schemaevo.cli write-budget-pareto-report \
  --run gepa=artifacts/gepa/results/summary.json \
  --run schemaevo=artifacts/openai_fixed_pool_hotpotqa/results/mvp_summary.json \
  --out artifacts/budget_pareto
```

## Method Path

The implemented fixed-pool MVP follows the plan:

1. Accept a reproduced fixed-call base program supplied by the user or adapter.
2. Collect train-only traces.
3. Generate a frozen schema pool from trace proposals, random controls, human templates, and validator-only control.
4. Optionally run train-trace-only reflection rounds that propose more schemas from the current primary schema context without exposing validation labels to the proposer.
5. Compile every schema into the same program call graph with frozen prompt text.
6. Smoke-test schema validity and call-count invariants.
7. Score surviving candidates on a selection split.
8. Select by lower confidence bound minus cost and invalid-output penalties.
9. Confirm top candidates on a held-out confirmation split.
10. Treat the selection-ranked top candidate as the pre-committed primary candidate.
11. Compare all confirmed candidates against the fixed-schema reference with paired tests and Benjamini-Hochberg correction.
12. Optionally score the pre-committed winner once on a third held-out test split.
13. Run field masking, blanking, shuffling, and downstream-disabled ablations on the primary schema.

The closed-loop optimizer then extends this with:

1. Initialize a schema population from supplied schemas, human templates, and random controls.
2. Evaluate candidates on deterministic shared minibatches, or on a cheap shared rung followed by larger shared-batch promotion.
3. Select parents by uncertainty-aware value plus field-use signal.
4. Apply legal schema mutations only.
5. Reject duplicate or statically invalid schemas with auditable reasons.
6. Maintain a Pareto front over score, cost, tokens, invalidity, and latency.
7. Enforce configured call/token/dollar caps before starting additional candidate evaluations.
8. Return final Pareto candidates without changing the call graph or demo IDs.

## Adapter Boundary

The core abstraction is `LMProgram` with ordered `ModuleSpec` objects. A DSPy adapter can map each DSPy module to a `ModuleSpec` runner while preserving the same module sequence, target model settings, retriever calls, and top-k. The OpenAI adapter builds `ModuleSpec` runners that call the Responses API with structured JSON output schemas for each module. The compiler does not add modules or calls; it only appends a marked schema contract block to each module prompt and extends output signatures.

Local HotpotQA and HoVer loaders/scorers live in `schemaevo.datasets`. They expect already-downloaded JSON or
JSONL files and convert records into `ProgramExample`s. The composability harness in
`schemaevo.experiments` accepts an external prompt optimizer callable, checks that it preserves the
call graph/demo invariant, then runs SchemaEvo on top of the optimized prompts. It re-scores the
final SchemaEvo candidates on the same held evaluation examples used for base and prompt-only scores,
and reports additive deltas plus evaluation and SchemaEvo optimizer budget summaries.

## Primary Safety Invariants

- No schema proposal may use non-train traces.
- No candidate may add LLM calls, retrieval calls, self-consistency, external tools, test labels, gold evidence, final-metric changes, model changes, or data-split changes.
- Primary validation never performs LLM repair. Deterministic coercion is allowed and logged.
- Candidate programs must match the base program call graph and demo IDs exactly.
- Fixed-pool confirmation uses the selection-ranked primary candidate for its main claim; the
  post-hoc best confirmation candidate is reported separately.
- Confirmed candidates get paired tests plus Benjamini-Hochberg correction.
- Field-use ablations are run after selection, not used as a selection bonus in the MVP.
