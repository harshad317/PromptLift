# SchemaEvo Pre-Registration

This file pre-registers the SchemaEvo SOTA claim before running benchmark-scale experiments.

## Claim

SchemaEvo is a fixed-call interface optimizer for multi-module LLM programs. It mutates typed
intermediate schemas, validators, and downstream consumption rules while preserving the target
model, module order, LLM-call count, retriever-call count, retriever top-k, examples, and decoding
settings.

## Headline Axes

Primary axes:

- Additive delta over prompt optimization: `GEPA -> GEPA + SchemaEvo` and `MIPRO -> MIPRO + SchemaEvo`.
- Cost at equal accuracy: dollars/tokens needed to reach the same held-out score.
- Structured-output validity rate: valid outputs without uncounted LLM repair.
- Cross-model transfer: retention of optimized schema gains from model A to model B compared with prompt transfer.

Secondary axes:

- Field-use causality: mask/shuffle/blank/downstream-disabled ablations.
- Deployment invariance: target LLM calls, retriever calls, retriever top-k, and serving latency/cost.
- Search efficiency: gain per optimizer rollout and accuracy-vs-budget curve.

## Tasks

Primary:

- HotpotQA multi-hop question answering.
- HoVer many-hop claim verification.

The experiment may add another interface-heavy task only as secondary evidence.

## Splits

Every run uses frozen, overlap-checked train, smoke, selection, confirmation, and held-out splits.
Schema proposal sees train traces only. Selection and confirmation labels are never exposed to the
schema proposer.

## Metrics

HotpotQA primary metric: exact-match answer score from `schemaevo.datasets.scorers.hotpotqa_exact_match`.

HoVer primary metric: normalized label accuracy from `schemaevo.datasets.scorers.hover_label_accuracy`.

Every method reports:

- Mean score and paired bootstrap confidence interval.
- Approximate-randomization p-value with Benjamini-Hochberg correction across confirmed candidates.
- Target task calls, retriever calls, prompt tokens, completion tokens, total tokens, dollars, p50/p95 latency.
- Invalid output rate and schema validation repair calls.

## Budget Points

Budget comparisons are run at pre-declared caps:

- Target-task calls: 25%, 50%, 100% of the reproduced prompt-optimizer rollout budget.
- Dollar/token caps: matched to the same target-task rollout points using the committed provider price table.
- Deployment budget: unchanged target calls and retriever calls per evaluated example.

If an official baseline artifact specifies a rollout count, that count is used. Otherwise the reproduced
baseline's measured target-task rollout count is the 100% budget.

## Win / Tie / Fail Rules

A win requires one of:

- Held-out additive delta over GEPA or MIPRO has paired bootstrap 95% CI excluding 0 and BH-adjusted
  approximate-randomization p-value below 0.05.
- SchemaEvo reaches the same held-out accuracy at lower measured dollar/token budget.
- Cross-model schema transfer retains at least 50% more source-model gain than prompt transfer.

A tie is any absolute held-out delta within +/-1.0 point or any paired CI that crosses 0 without a
cost or transfer advantage.

A fail is any SOTA-relevant gain that requires extra target-model calls, extra retriever calls, increased
retriever top-k, uncounted self-consistency, uncounted repair, split leakage, or hidden prompt/demo changes.

## Causal Pilot Go / No-Go

Before benchmark-scale spending, run one cheap real-model causal pilot on an interface-heavy task.

Go if at least one high-value field intervention removes meaningful signal:

- Shuffle or mask drop is at least 50% of the primary score delta, or
- Absolute shuffle or mask drop is at least 1.5 points.

No-go if scrambling field content drops score by approximately 0 on the pilot. In that case, stop or
reframe before large-scale GEPA/MIPRO experiments.

## Reporting

All reported artifacts must include the frozen schema pool, split IDs, per-example predictions, per-example
scores, per-call logs, payloads, cost ledgers, pricing source/date, tokenizer choice, and seed/config files.
