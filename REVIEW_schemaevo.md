# SchemaEvo — Code & Method Review

A review of the implemented method plus a concrete roadmap to make it competitive with / superior to MIPROv2, GEPA, and friends.

---

## 1. What the method actually is (one-paragraph framing)

SchemaEvo optimizes a dimension that MIPRO and GEPA **do not touch**: the **typed intermediate
interface between modules** of a multi-module LLM program. It freezes prompt *text* and the *call
graph* (module order, #LLM calls, #retriever calls, top-k), and searches over the schema contract —
which typed fields each module emits, how downstream modules are told to consume them, and the
validators on those fields. This orthogonality is the single most valuable thing in the repo and
should be the headline of any comparison paper. MIPRO optimizes instructions + few-shot demos; GEPA
optimizes prompt text via reflective evolution. SchemaEvo optimizes the data contract. They are
*composable*, not competing — which is both your differentiation and your path to SOTA (see §4C).

---

## 2. What you did well

**Genuinely novel optimization target.** Evolving the inter-module schema at a frozen call graph
and frozen prompt text is a real, defensible contribution. It isolates "information flow between
modules" as the lever, which prompt-text optimizers conflate with everything else.

**Experimental hygiene is well above the norm for prompt-opt repos.** Specifically:
- Train-only proposal guardrail, *enforced* (`_assert_train_only`, split-overlap detection in
  `_validate_examples_and_splits`). Most repos leak.
- Call-graph invariants enforced at *both* compile time and eval time (`assert_same_call_graph`,
  the `extract_call_graph(...) == original` assert inside the compiler). This is exactly what makes
  a "we didn't just add compute" claim credible.
- An explicit `FORBIDDEN_MUTATIONS` set (add LLM call, increase top-k, add self-consistency, change
  model/metric/split, use gold/test labels). This is the rollout-budget-parity discipline that
  GEPA/MIPRO comparisons usually get wrong.
- No uncounted LLM repair in primary eval — deterministic coercion only, and it's logged
  (`deterministic_repair_applied`). Cost ledger, token/latency accounting, per-call logs.

**Causal field-use ablations.** mask / blank / shuffle / downstream-disabled (`field_ablations.py`)
is a differentiator. The *shuffle* (rotated values) test in particular separates "the model uses the
field's content" from "the model just reacts to the field's presence." Few optimizers verify their
edits are load-bearing rather than placebo.

**Clean, reproducible architecture.** Frozen dataclasses, content-addressed schema IDs
(`with_id_from_content` / `stable_hash`) give free dedup, caching keys, and reproducibility;
deterministic seeding throughout; a pluggable `SchemaProposer` Protocol and an `LMProgram`/`ModuleSpec`
adapter boundary that can wrap DSPy modules without changing the call graph. Good separation:
candidate ↔ grammar ↔ mutation ↔ proposer ↔ compiler ↔ evaluator ↔ stats ↔ optimizer.

**Multi-objective selection.** A Pareto front over (score, cost, tokens, invalidity, latency) plus an
LCB-minus-cost selection value is the right *shape* for deployment-aware optimization.

---

## 3. Weaknesses, ranked by how much they threaten a SOTA claim

### Worst (these are what a reviewer/competitor will attack first)

1. **Candidates are compared across *different* minibatches — the comparisons aren't paired.**
   In both `schema_evo` (`seed=config.seed + step`) and the fixed-pool selection round, each
   candidate is scored on its own minibatch. Means over non-identical example sets are not
   comparable, the LCB ranking is biased, and the Pareto front is built from incomparable points.
   Fix: a **shared, fixed evaluation set** per round (or paired evaluation across identical
   examples). This single change moves the whole pipeline from "suggestive" to "defensible."

2. **Winner's curse / multiple comparisons.** You LCB-select top-k on selection, pick the `max`
   mean on confirmation, then run a paired test on *only that winner* vs baseline. Selecting the max
   of k noisy candidates and then testing it inflates the effect. Fix: reserve a **third held-out
   test split** scored only by the single pre-committed winner, or apply Benjamini–Hochberg / Bonferroni
   across the k confirmed candidates. As written, reported gains are optimistically biased.

3. **The optimizer is a thin random hill-climber, not competitive search.**
   `_select_parent` = "top-5 by value, then uniform random"; `sample_schema_mutation` emits only
   **4 of the 15** defined ops (`toggle_required`, `tighten_validator`, `drop_field`,
   `add_template_field`). split / merge / move / change_type / rename / enum ops are implemented in
   `apply_mutation` but **never sampled**. There is no surrogate model, no bandit allocation of
   rollouts, no Bayesian/TPE search. MIPRO uses TPE/Bayesian optimization over its space; this is
   currently nowhere near that.

4. **No real-LLM evidence; the cost/token numbers are synthetic.** Execution runs against a
   deterministic local runner, the "tokenizer" is `len(text)//4`, and dollar cost is a proxy. Fine
   for plumbing/CI, but you cannot claim SOTA over MIPRO/GEPA without HotpotQA/HoVer (already named
   in scope-not-done) on real models with a real tokenizer and provider pricing.

### Significant

5. **Validators don't bite.** `validators` is a free-text string dict; "tighten/relax validator"
   just overwrites the string and pydantic only enforces *type*. So a `tighten_validator` mutation
   often has zero behavioral effect (and `tighten` can re-write the rule to what it already was → a
   wasted rollout that isn't even caught unless the whole schema hashes identically). Validators are
   currently decorative.

6. **Statistics are shaky at small n.** `standard_error` is a normal approximation on bounded/
   near-Bernoulli scores with n ≤ 32; the LCB uses a hardcoded `1.96`. Use bootstrap or Wilson
   intervals. The acquisition weights (`cost 0.1`, `invalid 2.0`, `field 0.05`) are uncalibrated
   magic numbers.

7. **Pareto front bloats under noise.** With 5 objectives estimated from ≤32 examples, almost
   nothing is dominated, so the "front" keeps most candidates. And `ParetoFront.top()` re-sorts by
   `mean_score` first, so the final pick collapses back to score-ranking — the cost/latency objectives
   don't actually drive the final choice. Use **significance-aware dominance** (only dominate on a
   statistically meaningful gap).

8. **`strict_invalid_policy` double-counts validity.** Invalid → score 0 *and* invalid_output_rate
   penalizes again in the selection value. An invalid-but-near-correct output is zeroed. Consider
   decoupling task score from validity, or report both and penalize once.

9. **No rollout cache.** Schema IDs are content-addressed, yet identical `(schema_hash, example_id,
   seed)` executions are recomputed across rounds. On real LLMs this is the difference between an
   affordable and an unaffordable experiment, and it's prerequisite for honest budget-matching.

### Minor

10. The "trace-based proposal" is keyword counting over trace text (`_prioritize_fields`), not
    reflection — it can only reorder/subset a fixed template field list. It cannot invent a field
    a failure implies.
11. `_missing_template_fields` calls `make_human_minimal_schemas` without the seed in one path;
    harmless now, but a determinism smell.
12. No few-shot/demo axis at all. Defensible as "we hold demos fixed," but it's the single biggest
    known lever (MIPRO's bootstrapped demos), so it must be controlled for explicitly in any
    head-to-head.

---

## 4. How to make it SOTA and beat MIPRO / GEPA

### A. Make the comparison airtight (cheapest, highest credibility)
- **Shared fixed eval set per round** + **paired** candidate-vs-candidate and candidate-vs-baseline
  scoring (you already have paired bootstrap + approximate randomization — feed them paired data).
- **Third held-out test split** scored only by the single pre-registered winner; multiple-testing
  correction across confirmed candidates.
- **Budget-matched rollouts**: define one rollout/$ budget and hold it equal across SchemaEvo,
  MIPRO, GEPA. Your call-graph invariant + cost ledger already set this up — wire a real cost model
  and a `(schema_hash, example_id, seed)` rollout cache so budget-matching is real.

### B. Make the search actually competitive
- **Reflective LLM proposer (steal GEPA's engine, point it at the schema).** Implement a
  `SchemaProposer` that feeds failing traces + current schema + validator errors to an LLM and asks
  for the next mutation *with rationale* — the Protocol seam and the `rationale` field already exist.
  This is what lets you invent fields a failure implies, which keyword-matching cannot.
- **Bandit / successive-halving allocation (steal MIPRO's budget discipline).** Instead of one
  minibatch per candidate, allocate evaluation budget adaptively (Hyperband / successive halving,
  or a TPE surrogate over the discrete schema space). Reflection *proposes*, the bandit *allocates*,
  LCB/Pareto *selects*.
- **Use all 15 mutation ops**, and add an **operator bandit** that learns which op is paying off and
  adapts the mutation distribution (self-adjusting evolution).

### C. Position it as a composable *layer*, not a rival (this is the strongest empirical story)
- Run GEPA/MIPRO on prompts+demos, *then* SchemaEvo on the interface, and show **additive** gains:
  "GEPA + SchemaEvo > GEPA alone" budget-matched. You win by stacking on top, and the ablation cleanly
  attributes the delta to the interface.
- Add a **GEPA+Merge-style merge operator**: combine the field sets of complementary Pareto-best
  schemas from different lineages. You already track `parent_schema_id`, so lineage-aware merge is a
  small step and a known source of GEPA's edge.

### D. Make validators executable (turns a decorative field into a real lever)
- Replace free-text validators with **executable constraints**: regex, numeric ranges, cross-field
  consistency, "evidence titles must appear in the retrieved set," enum-membership. Then
  tighten/relax actually change behavior and you can plot the **validity ↔ score** tradeoff — a
  result MIPRO/GEPA structurally cannot produce, because they have no typed interface to constrain.

### E. Pick benchmarks where the hypothesis has maximum leverage
- HotpotQA + HoVer (named already), plus at least one **agentic/structured-extraction** pipeline
  where inter-module information loss is the dominant failure mode. The method should shine exactly
  where the typed interface matters most; choosing those tasks is the differentiation narrative.

### F. Statistics upgrade
- Bootstrap/Wilson CIs instead of normal-approx SE; tune the LCB `beta` (treat selection as a real
  bandit/BO acquisition); significance-aware Pareto dominance to stop front bloat.

---

## 5. The 30-second version

You built the *rigor* of a SOTA system (guardrails, invariants, ablations, paired tests, cost
accounting) around a *novel and defensible target* (the typed inter-module contract). What's missing
is (1) statistical comparability — shared/paired eval sets and a held-out winner test; (2) a real
search — reflective proposer + bandit allocation using all the operators you already implemented;
and (3) real-LLM benchmark evidence. Fix those three and frame SchemaEvo as a **composable layer that
stacks on top of GEPA/MIPRO** rather than a competitor, and the "additive, budget-matched, interface-
attributed gain" result is both SOTA-credible and uniquely yours.
