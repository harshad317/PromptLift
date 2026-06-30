from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.field_ablations import FieldAblationResult, run_field_use_ablations
from schemaevo.eval.scoring import CandidateEvalResult, Scorer, evaluate_program
from schemaevo.eval.stats import PairedComparison, benjamini_hochberg_adjust, compare_paired
from schemaevo.optimizers.selection import select_top_k_by_lcb
from schemaevo.programs.base import LMProgram, ProgramExample
from schemaevo.programs.call_graph import assert_same_call_graph
from schemaevo.programs.compile_schema_program import compile_schema_program
from schemaevo.schemas.candidate import SchemaCandidate
from schemaevo.schemas.grammar import SchemaGrammar
from schemaevo.schemas.human_templates import make_human_minimal_schemas, make_validator_only_schema
from schemaevo.schemas.proposer import SchemaProposer, TraceExample, propose_schemas_from_traces
from schemaevo.schemas.random_controls import make_random_schema_controls
from schemaevo.schemas.serialization import freeze_jsonl, write_json


@dataclass(frozen=True)
class FixedPoolConfig:
    task: str = "HoVer"
    target_model: str = "gpt-4.1-mini"
    seed: int = 0
    n_trace_schemas: int = 40
    n_random_schemas: int = 10
    schema_token_budget: int = 512
    min_smoke_validity: float = 0.97
    top_k_confirmation: int = 5
    min_confirmation_delta: float = 0.0
    bootstrap_resamples: int = 2000
    randomization_swaps: int = 2000
    freeze_prompt_text: bool = True
    allow_only_schema_contract_insert: bool = True
    strict_invalid_policy: bool = True
    multiple_comparison_correction: str = "benjamini_hochberg"

    def __post_init__(self) -> None:
        if self.n_trace_schemas < 0:
            raise ValueError("n_trace_schemas must be non-negative")
        if self.n_random_schemas < 0:
            raise ValueError("n_random_schemas must be non-negative")
        if self.schema_token_budget <= 0:
            raise ValueError("schema_token_budget must be positive")
        if not 0.0 <= self.min_smoke_validity <= 1.0:
            raise ValueError("min_smoke_validity must be in [0, 1]")
        if self.top_k_confirmation <= 0:
            raise ValueError("top_k_confirmation must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if self.randomization_swaps <= 0:
            raise ValueError("randomization_swaps must be positive")
        if self.multiple_comparison_correction not in {"benjamini_hochberg", "none"}:
            raise ValueError("multiple_comparison_correction must be 'benjamini_hochberg' or 'none'")


@dataclass(frozen=True)
class MVPDecision:
    proceed: bool
    score_delta: float
    invalid_output_rate: float
    field_masking_max_drop: float
    reasons: tuple[str, ...]


@dataclass
class FixedPoolResult:
    baseline_selection_result: CandidateEvalResult
    baseline_confirmation_result: CandidateEvalResult
    schema_pool: tuple[SchemaCandidate, ...]
    smoke_results: tuple[CandidateEvalResult, ...]
    selection_results: tuple[CandidateEvalResult, ...]
    top_selection_results: tuple[CandidateEvalResult, ...]
    confirmation_results: tuple[CandidateEvalResult, ...]
    primary_confirmation_result: CandidateEvalResult
    best_confirmation_result: CandidateEvalResult
    paired_stats: PairedComparison
    corrected_confirmation_stats: dict[str, PairedComparison]
    heldout_test_result: CandidateEvalResult | None
    heldout_test_stats: PairedComparison | None
    field_ablation_results: tuple[FieldAblationResult, ...]
    decision: MVPDecision
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "decision": asdict(self.decision),
            "baseline_confirmation_mean": self.baseline_confirmation_result.mean_score,
            "primary_schema_id": self.primary_confirmation_result.schema_id,
            "primary_confirmation_mean": self.primary_confirmation_result.mean_score,
            "best_schema_id": self.best_confirmation_result.schema_id,
            "best_confirmation_mean": self.best_confirmation_result.mean_score,
            "heldout_test_mean": self.heldout_test_result.mean_score if self.heldout_test_result else None,
            "score_delta": self.decision.score_delta,
            "invalid_output_rate": self.best_confirmation_result.invalid_output_rate,
            "field_ablations": [asdict(result) for result in self.field_ablation_results],
            "artifacts": self.artifacts,
        }


def run_fixed_pool_schema_mvp(
    *,
    base_program: LMProgram,
    train_traces: tuple[TraceExample, ...],
    smoke_examples: tuple[ProgramExample, ...],
    selection_examples: tuple[ProgramExample, ...],
    confirmation_examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    proposer: SchemaProposer | None = None,
    heldout_test_examples: tuple[ProgramExample, ...] = (),
    artifact_dir: str | Path | None = None,
) -> FixedPoolResult:
    artifact_root = Path(artifact_dir) if artifact_dir else None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
    rollout_cache = RolloutCache(artifact_root / "rollout_cache" if artifact_root else None)
    _validate_examples_and_splits(
        train_traces=train_traces,
        smoke_examples=smoke_examples,
        selection_examples=selection_examples,
        confirmation_examples=confirmation_examples,
        heldout_test_examples=heldout_test_examples,
    )

    module_names = base_program.module_names
    grammar = SchemaGrammar(
        allowed_modules=module_names,
        max_schema_tokens=config.schema_token_budget,
    )
    schema_pool = _build_schema_pool(
        traces=train_traces,
        task=config.task,
        module_names=module_names,
        config=config,
        proposer=proposer,
    )
    schema_pool = tuple(_static_filter(schema_pool, grammar))
    if not schema_pool:
        raise RuntimeError("schema pool is empty after static checks")
    schema_pool_path = ""
    if artifact_root:
        schema_pool_path = str(freeze_jsonl(schema_pool, artifact_root / "schemas" / "frozen_pool.jsonl"))

    compiled_candidates = tuple(
        _compile_candidate(base_program=base_program, schema=schema, config=config) for schema in schema_pool
    )

    baseline_selection = evaluate_program(
        program=base_program,
        examples=selection_examples,
        scorer=scorer,
        method="fixed_schema_reference",
        candidate_id="fixed_schema_reference_selection",
        seed=config.seed,
        baseline_program=base_program,
        strict_invalid_policy=config.strict_invalid_policy,
        artifact_dir=artifact_root,
        rollout_cache=rollout_cache,
        run_id="fixed_schema_reference_selection",
    )
    smoke_results = (
        tuple(
            _evaluate_many(
                candidates=compiled_candidates,
                examples=smoke_examples,
                scorer=scorer,
                config=config,
                base_program=base_program,
                artifact_root=artifact_root,
                method="schema_smoke",
                baseline_cost=baseline_selection.dollar_cost_per_example,
                rollout_cache=rollout_cache,
            )
        )
        if smoke_examples
        else ()
    )
    smoke_pass_schema_ids = {
        result.schema_id
        for result in smoke_results
        if 1.0 - result.invalid_output_rate >= config.min_smoke_validity
    }
    if smoke_results:
        compiled_candidates = tuple(
            candidate
            for candidate in compiled_candidates
            if candidate.schema_candidate and candidate.schema_candidate.schema_id in smoke_pass_schema_ids
        )
    if not compiled_candidates:
        raise RuntimeError("no schema candidates survived smoke validation")

    selection_results = tuple(
        _evaluate_many(
            candidates=compiled_candidates,
            examples=selection_examples,
            scorer=scorer,
            config=config,
            base_program=base_program,
            artifact_root=artifact_root,
            method="schema_selection",
            baseline_cost=baseline_selection.dollar_cost_per_example,
            rollout_cache=rollout_cache,
        )
    )
    top_selection_results = tuple(
        select_top_k_by_lcb(
            list(selection_results),
            k=min(config.top_k_confirmation, len(selection_results)),
            use_field_bonus=False,
        )
    )
    primary_schema_id = top_selection_results[0].schema_id if top_selection_results else ""
    top_schema_ids = {result.schema_id for result in top_selection_results}
    top_programs = tuple(
        candidate
        for candidate in compiled_candidates
        if candidate.schema_candidate and candidate.schema_candidate.schema_id in top_schema_ids
    )

    baseline_confirmation = evaluate_program(
        program=base_program,
        examples=confirmation_examples,
        scorer=scorer,
        method="fixed_schema_reference",
        candidate_id="fixed_schema_reference_confirmation",
        seed=config.seed,
        baseline_program=base_program,
        strict_invalid_policy=config.strict_invalid_policy,
        artifact_dir=artifact_root,
        rollout_cache=rollout_cache,
        run_id="fixed_schema_reference_confirmation",
    )
    confirmation_results = tuple(
        _evaluate_many(
            candidates=top_programs,
            examples=confirmation_examples,
            scorer=scorer,
            config=config,
            base_program=base_program,
            artifact_root=artifact_root,
            method="schema_confirmation",
            baseline_cost=baseline_confirmation.dollar_cost_per_example,
            rollout_cache=rollout_cache,
        )
    )
    if not confirmation_results:
        raise RuntimeError("no schema candidates survived smoke/selection")
    primary_confirmation = next(
        result for result in confirmation_results if result.schema_id == primary_schema_id
    )
    best_confirmation = max(
        confirmation_results,
        key=lambda result: (result.mean_score, -result.invalid_output_rate, result.schema_id),
    )
    best_program = next(
        candidate
        for candidate in top_programs
        if candidate.schema_candidate and candidate.schema_candidate.schema_id == primary_confirmation.schema_id
    )
    paired_stats = compare_paired(
        baseline_confirmation.per_example_scores,
        primary_confirmation.per_example_scores,
        n_resamples=config.bootstrap_resamples,
        n_swaps=config.randomization_swaps,
        seed=config.seed,
    )
    corrected_confirmation_stats = _corrected_confirmation_stats(
        baseline=baseline_confirmation,
        candidates=confirmation_results,
        config=config,
    )
    paired_stats = corrected_confirmation_stats.get(primary_confirmation.schema_id, paired_stats)
    heldout_test_result: CandidateEvalResult | None = None
    heldout_test_stats: PairedComparison | None = None
    if heldout_test_examples:
        heldout_baseline = evaluate_program(
            program=base_program,
            examples=heldout_test_examples,
            scorer=scorer,
            method="fixed_schema_reference",
            candidate_id="fixed_schema_reference_heldout_test",
            seed=config.seed,
            baseline_program=base_program,
            strict_invalid_policy=config.strict_invalid_policy,
            artifact_dir=artifact_root,
            rollout_cache=rollout_cache,
            run_id="fixed_schema_reference_heldout_test",
        )
        heldout_test_result = evaluate_program(
            program=best_program,
            examples=heldout_test_examples,
            scorer=scorer,
            method="schema_heldout_test",
            candidate_id=f"schema_heldout_test_{primary_confirmation.schema_id}",
            seed=config.seed,
            baseline_program=base_program,
            strict_invalid_policy=config.strict_invalid_policy,
            artifact_dir=artifact_root,
            rollout_cache=rollout_cache,
            schema_generation_calls=0,
            run_id=f"schema_heldout_test_{primary_confirmation.schema_id}",
        )
        heldout_test_result.baseline_dollar_cost_per_example = heldout_baseline.dollar_cost_per_example
        heldout_test_stats = compare_paired(
            heldout_baseline.per_example_scores,
            heldout_test_result.per_example_scores,
            n_resamples=config.bootstrap_resamples,
            n_swaps=config.randomization_swaps,
            seed=config.seed + 1000,
        )
    field_ablation_results = tuple(
        run_field_use_ablations(
            program=best_program,
            examples=confirmation_examples,
            scorer=scorer,
            unablated_result=primary_confirmation,
            fields=best_program.schema_candidate.evolved_field_names if best_program.schema_candidate else (),
            seed=config.seed,
            artifact_dir=artifact_root,
        )
    )
    decision = _make_decision(
        baseline=baseline_confirmation,
        best=primary_confirmation,
        paired_stats=paired_stats,
        field_ablation_results=field_ablation_results,
        config=config,
    )
    artifacts = {"schema_pool": schema_pool_path}
    if artifact_root:
        summary_path = write_json(
            {
                "decision": asdict(decision),
                "baseline_confirmation": baseline_confirmation.to_dict(),
                "best_confirmation": best_confirmation.to_dict(),
                "primary_confirmation": primary_confirmation.to_dict(),
                "paired_stats": {
                    "bootstrap": asdict(paired_stats.bootstrap),
                    "approximate_randomization_p": paired_stats.approximate_randomization_p,
                    "adjusted_p": paired_stats.adjusted_p,
                    "correction": paired_stats.correction,
                },
                "corrected_confirmation_stats": {
                    schema_id: {
                        "bootstrap": asdict(stats.bootstrap),
                        "approximate_randomization_p": stats.approximate_randomization_p,
                        "adjusted_p": stats.adjusted_p,
                        "correction": stats.correction,
                    }
                    for schema_id, stats in corrected_confirmation_stats.items()
                },
                "heldout_test_result": heldout_test_result.to_dict() if heldout_test_result else None,
                "heldout_test_stats": {
                    "bootstrap": asdict(heldout_test_stats.bootstrap),
                    "approximate_randomization_p": heldout_test_stats.approximate_randomization_p,
                }
                if heldout_test_stats
                else None,
                "field_ablation_results": [asdict(result) for result in field_ablation_results],
            },
            artifact_root / "results" / "mvp_summary.json",
        )
        artifacts["summary"] = str(summary_path)

    return FixedPoolResult(
        baseline_selection_result=baseline_selection,
        baseline_confirmation_result=baseline_confirmation,
        schema_pool=schema_pool,
        smoke_results=smoke_results,
        selection_results=selection_results,
        top_selection_results=top_selection_results,
        confirmation_results=confirmation_results,
        primary_confirmation_result=primary_confirmation,
        best_confirmation_result=best_confirmation,
        paired_stats=paired_stats,
        corrected_confirmation_stats=corrected_confirmation_stats,
        heldout_test_result=heldout_test_result,
        heldout_test_stats=heldout_test_stats,
        field_ablation_results=field_ablation_results,
        decision=decision,
        artifacts=artifacts,
    )


def _build_schema_pool(
    *,
    traces: tuple[TraceExample, ...],
    task: str,
    module_names: tuple[str, ...],
    config: FixedPoolConfig,
    proposer: SchemaProposer | None,
) -> tuple[SchemaCandidate, ...]:
    proposed = propose_schemas_from_traces(
        traces=traces,
        task=task,
        module_names=module_names,
        n=config.n_trace_schemas,
        seed=config.seed,
        schema_token_budget=config.schema_token_budget,
        proposer=proposer,
    )
    random_controls = make_random_schema_controls(
        task=task,
        module_names=module_names,
        n=config.n_random_schemas,
        seed=config.seed,
        schema_token_budget=config.schema_token_budget,
    )
    human = make_human_minimal_schemas(task=task, module_names=module_names, seed=config.seed)
    validator_only = make_validator_only_schema(
        task=task,
        module_names=module_names,
        seed=config.seed,
        schema_token_budget=config.schema_token_budget,
    )
    by_id: dict[str, SchemaCandidate] = {}
    for candidate in (*proposed, *random_controls, *human, validator_only):
        by_id.setdefault(candidate.schema_id, candidate)
    return tuple(by_id.values())


def _validate_examples_and_splits(
    *,
    train_traces: tuple[TraceExample, ...],
    smoke_examples: tuple[ProgramExample, ...],
    selection_examples: tuple[ProgramExample, ...],
    confirmation_examples: tuple[ProgramExample, ...],
    heldout_test_examples: tuple[ProgramExample, ...] = (),
) -> None:
    if not train_traces:
        raise ValueError("train_traces must be non-empty")
    if not selection_examples:
        raise ValueError("selection_examples must be non-empty")
    if not confirmation_examples:
        raise ValueError("confirmation_examples must be non-empty")
    bad_trace_splits = [trace.example_id for trace in train_traces if trace.split != "train"]
    if bad_trace_splits:
        raise ValueError(f"schema proposal traces must be train split only: {bad_trace_splits[:5]}")

    groups = {
        "train_traces": {trace.example_id for trace in train_traces},
        "smoke_examples": {example.example_id for example in smoke_examples},
        "selection_examples": {example.example_id for example in selection_examples},
        "confirmation_examples": {example.example_id for example in confirmation_examples},
        "heldout_test_examples": {example.example_id for example in heldout_test_examples},
    }
    for left_name, left_ids in groups.items():
        for right_name, right_ids in groups.items():
            if left_name >= right_name:
                continue
            overlap = left_ids & right_ids
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(
                    f"example IDs overlap between {left_name} and {right_name}: {preview}"
                )


def _static_filter(
    candidates: tuple[SchemaCandidate, ...],
    grammar: SchemaGrammar,
) -> list[SchemaCandidate]:
    passing: list[SchemaCandidate] = []
    for candidate in candidates:
        result = grammar.check_candidate(candidate)
        if result.passed:
            passing.append(candidate)
    return passing


def _compile_candidate(
    *,
    base_program: LMProgram,
    schema: SchemaCandidate,
    config: FixedPoolConfig,
) -> LMProgram:
    candidate = compile_schema_program(
        base_program=base_program,
        schema=schema,
        freeze_prompt_text=config.freeze_prompt_text,
        allow_only_schema_contract_insert=config.allow_only_schema_contract_insert,
    )
    assert_same_call_graph(candidate, base_program)
    return candidate


def _evaluate_many(
    *,
    candidates: tuple[LMProgram, ...],
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    base_program: LMProgram,
    artifact_root: Path | None,
    method: str,
    baseline_cost: float,
    rollout_cache: RolloutCache,
) -> list[CandidateEvalResult]:
    results: list[CandidateEvalResult] = []
    for index, candidate in enumerate(candidates):
        schema_id = candidate.schema_candidate.schema_id if candidate.schema_candidate else "original_schema"
        result = evaluate_program(
            program=candidate,
            examples=examples,
            scorer=scorer,
            method=method,
            candidate_id=f"{method}_{index}",
            seed=config.seed,
            baseline_program=base_program,
            strict_invalid_policy=config.strict_invalid_policy,
            artifact_dir=artifact_root,
            rollout_cache=rollout_cache,
            schema_generation_calls=1,
            run_id=f"{method}_{schema_id}_{index}",
        )
        result.baseline_dollar_cost_per_example = baseline_cost
        results.append(result)
    return results


def _corrected_confirmation_stats(
    *,
    baseline: CandidateEvalResult,
    candidates: tuple[CandidateEvalResult, ...],
    config: FixedPoolConfig,
) -> dict[str, PairedComparison]:
    raw_stats = [
        compare_paired(
            baseline.per_example_scores,
            candidate.per_example_scores,
            n_resamples=config.bootstrap_resamples,
            n_swaps=config.randomization_swaps,
            seed=config.seed + index,
        )
        for index, candidate in enumerate(candidates)
    ]
    if config.multiple_comparison_correction == "benjamini_hochberg":
        adjusted = benjamini_hochberg_adjust(
            tuple(stats.approximate_randomization_p for stats in raw_stats)
        )
        correction = "benjamini_hochberg"
    else:
        adjusted = tuple(stats.approximate_randomization_p for stats in raw_stats)
        correction = "none"
    return {
        candidate.schema_id: PairedComparison(
            bootstrap=stats.bootstrap,
            approximate_randomization_p=stats.approximate_randomization_p,
            adjusted_p=adjusted[index],
            correction=correction,
        )
        for index, (candidate, stats) in enumerate(zip(candidates, raw_stats))
    }


def _make_decision(
    *,
    baseline: CandidateEvalResult,
    best: CandidateEvalResult,
    paired_stats: PairedComparison,
    field_ablation_results: tuple[FieldAblationResult, ...],
    config: FixedPoolConfig,
) -> MVPDecision:
    score_delta = best.mean_score - baseline.mean_score
    max_drop = max((result.drop_vs_unablated for result in field_ablation_results), default=0.0)
    reasons: list[str] = []
    if score_delta < config.min_confirmation_delta:
        reasons.append(
            f"score delta {score_delta:.4f} is below configured bar {config.min_confirmation_delta:.4f}"
        )
    if best.invalid_output_rate > (1.0 - config.min_smoke_validity):
        reasons.append(
            f"invalid output rate {best.invalid_output_rate:.4f} exceeds primary validity bar"
        )
    if paired_stats.bootstrap.ci_low <= 0 and config.min_confirmation_delta > 0:
        reasons.append("paired bootstrap CI does not exclude zero")
    if not field_ablation_results:
        reasons.append("no field ablations were produced")
    proceed = not reasons
    return MVPDecision(
        proceed=proceed,
        score_delta=score_delta,
        invalid_output_rate=best.invalid_output_rate,
        field_masking_max_drop=max_drop,
        reasons=tuple(reasons),
    )
