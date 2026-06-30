# SchemaEvo

Fixed-call schema evolution for multi-module LLM prompt programs.

This repository implements the prompt-optimization method from the supplied implementation plan: evolve typed intermediate schema contracts, validators, and downstream consumption rules while preserving the deployment call graph.

## Scope

Implemented:

- Schema candidate representation, token-budget checks, legal mutation grammar, and forbidden mutation guardrails.
- Human semantic templates, random same-capacity controls, and train-only trace-based schema proposals.
- Optional OpenAI reflective schema proposer using `gpt-4.1-mini` and Structured Outputs.
- Schema compiler that patches module output signatures and inserts schema contracts into frozen prompts.
- Automated call-graph invariants for module order, LLM-call count, retriever-call count, and retrieval top-k.
- Pydantic-backed local validation plus executable constraints (`non_empty`, `regex`, numeric ranges, max tokens/items) with deterministic coercion only; no uncounted LLM repair.
- Per-call logging, rollout cache, cost ledger structures, per-example scoring, latency/token accounting, bootstrap LCB selection, paired bootstrap, approximate randomization, multiple-comparison correction, and field masking/blanking/shuffling/downstream-disabled ablations.
- Generic closed-loop SchemaEvo optimizer with population initialization, all grammar operators sampled, operator-bandit weighting, paired minibatch evaluation, significance-aware Pareto tracking, rejected-schema audit records, and final candidate selection.
- A local deterministic toy multi-hop program that exercises the full fixed-pool SchemaEvo path without external APIs or benchmark data.

Intentionally not implemented yet:

- GEPA/GEPA+Merge reproduction.
- HotpotQA/HoVer dataset runners.
- Competing prompt optimization methods or benchmark baselines such as MIPROv2, CAPO, ADOPT, Trace/Opto, PCO, BO, Hyperband, or prompt-only GEPA.

Those are experimental/benchmarking tasks and are outside the current instruction.

## Quick Start

```bash
python3 -m pytest -q
python3 -m schemaevo.cli run-toy-mvp --config configs/toy_schemaevo.yaml --out artifacts/toy_mvp
python3 -m schemaevo.cli run-toy-closed-loop --config configs/toy_schemaevo.yaml --out artifacts/toy_closed_loop
```

Or:

```bash
make test
make run-toy-mvp
make run-toy-closed-loop
```

The toy run writes a frozen schema pool, per-call logs, per-example scores, payloads, cost ledgers, and `artifacts/toy_mvp/results/mvp_summary.json`.

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

## Method Path

The implemented fixed-pool MVP follows the plan:

1. Accept a reproduced fixed-call base program supplied by the user or adapter.
2. Collect train-only traces.
3. Generate a frozen schema pool from trace proposals, random controls, human templates, and validator-only control.
4. Compile every schema into the same program call graph with frozen prompt text.
5. Smoke-test schema validity and call-count invariants.
6. Score surviving candidates on a selection split.
7. Select by lower confidence bound minus cost and invalid-output penalties.
8. Confirm top candidates on a held-out confirmation split.
9. Treat the selection-ranked top candidate as the pre-committed primary candidate.
10. Compare all confirmed candidates against the fixed-schema reference with paired tests and Benjamini-Hochberg correction.
11. Optionally score the pre-committed winner once on a third held-out test split.
12. Run field masking, blanking, shuffling, and downstream-disabled ablations on the primary schema.

The closed-loop optimizer then extends this with:

1. Initialize a schema population from supplied schemas, human templates, and random controls.
2. Evaluate candidates on deterministic minibatches.
3. Select parents by LCB-style value plus field-use signal.
4. Apply legal schema mutations only.
5. Reject duplicate or statically invalid schemas with auditable reasons.
6. Maintain a Pareto front over score, cost, tokens, invalidity, and latency.
7. Return final Pareto candidates without changing the call graph.

## Adapter Boundary

The core abstraction is `LMProgram` with ordered `ModuleSpec` objects. A DSPy adapter can map each DSPy module to a `ModuleSpec` runner while preserving the same module sequence, target model settings, retriever calls, and top-k. The compiler does not add modules or calls; it only appends a marked schema contract block to each module prompt and extends output signatures.

## Primary Safety Invariants

- No schema proposal may use non-train traces.
- No candidate may add LLM calls, retrieval calls, self-consistency, external tools, test labels, gold evidence, final-metric changes, model changes, or data-split changes.
- Primary validation never performs LLM repair. Deterministic coercion is allowed and logged.
- Candidate programs must match the base program call graph exactly.
- Fixed-pool confirmation uses the selection-ranked primary candidate for its main claim; the
  post-hoc best confirmation candidate is reported separately.
- Confirmed candidates get paired tests plus Benjamini-Hochberg correction.
- Field-use ablations are run after selection, not used as a selection bonus in the MVP.
