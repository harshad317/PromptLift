from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
import pickle
from typing import Any

from schemaevo.eval.budgeting import EvaluationBudgetEstimate, estimate_evaluation_budget
from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.cost_ledger import BudgetLimits, BudgetTracker, CostMeter, make_cost_meter
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
from schemaevo.utils.progress import ProgressMode, progress_iter, progress_status


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
    reflection_rounds: int = 1
    reflection_schemas_per_round: int = 0
    use_tiktoken_costing: bool = False
    model_prices: dict[str, dict[str, float | str]] = field(default_factory=dict)
    max_target_task_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_dollar_cost: float | None = None
    workers: int = 1
    progress: ProgressMode = "auto"

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
        if self.reflection_rounds <= 0:
            raise ValueError("reflection_rounds must be positive")
        if self.reflection_schemas_per_round < 0:
            raise ValueError("reflection_schemas_per_round must be non-negative")
        for name in (
            "max_target_task_calls",
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_dollar_cost is not None and self.max_dollar_cost < 0:
            raise ValueError("max_dollar_cost must be non-negative")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.progress not in {"auto", "rich", "tqdm", "none"}:
            raise ValueError("progress must be one of: auto, rich, tqdm, none")


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
    cost_summary: dict[str, float | int]
    budget_summary: dict[str, float | int | bool | None]
    proposal_usage: dict[str, float | int]
    reflection_rounds: tuple[dict[str, Any], ...]
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
            "cost_summary": self.cost_summary,
            "budget": self.budget_summary,
            "proposal_usage": self.proposal_usage,
            "reflection_rounds": self.reflection_rounds,
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
    cost_meter = make_cost_meter(
        model_prices=config.model_prices,
        use_tiktoken=config.use_tiktoken_costing,
    )
    budget = _make_budget_tracker(config)
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
    _reset_proposal_usage(proposer)
    with progress_status("schema proposal", mode=config.progress):
        schema_pool = _build_schema_pool(
            traces=train_traces,
            task=config.task,
            module_names=module_names,
            config=config,
            proposer=proposer,
        )
    recorded_proposal_usage = _proposal_usage_from_proposer(proposer)
    _record_proposal_usage(budget, recorded_proposal_usage)
    with progress_status("static schema checks", mode=config.progress):
        schema_pool = tuple(_static_filter(schema_pool, grammar))
    if not schema_pool:
        raise RuntimeError("schema pool is empty after static checks")
    schema_pool_path = ""
    if artifact_root:
        schema_pool_path = str(freeze_jsonl(schema_pool, artifact_root / "schemas" / "frozen_pool.jsonl"))

    compiled_candidates = tuple(
        _compile_candidate(base_program=base_program, schema=schema, config=config)
        for schema in progress_iter(
            schema_pool,
            total=len(schema_pool),
            description="compile schemas",
            mode=config.progress,
        )
    )
    confirmation_pair_calls = len(confirmation_examples) * base_program.calls_per_example * 2
    selection_min_calls = len(selection_examples) * base_program.calls_per_example

    baseline_selection = _evaluate_required_with_budget(
        budget=budget,
        program=base_program,
        examples=selection_examples,
        scorer=scorer,
        config=config,
        method="fixed_schema_reference",
        candidate_id="fixed_schema_reference_selection",
        baseline_program=base_program,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        run_id="fixed_schema_reference_selection",
        stage="selection baseline",
        reserve_target_task_calls=confirmation_pair_calls + selection_min_calls,
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
                cost_meter=cost_meter,
                rollout_cache=rollout_cache,
                budget=budget,
                reserve_target_task_calls=selection_min_calls + confirmation_pair_calls,
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
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            budget=budget,
            reserve_target_task_calls=confirmation_pair_calls,
        )
    )
    reflection_rounds: tuple[dict[str, Any], ...] = ()
    if config.reflection_rounds > 1 and proposer is not None:
        (
            schema_pool,
            compiled_candidates,
            smoke_results,
            selection_results,
            reflection_rounds,
        ) = _run_reflection_rounds(
            base_program=base_program,
            schema_pool=schema_pool,
            compiled_candidates=compiled_candidates,
            smoke_results=smoke_results,
            selection_results=selection_results,
            train_traces=train_traces,
            selection_examples=selection_examples,
            smoke_examples=smoke_examples,
            scorer=scorer,
            config=config,
            proposer=proposer,
            grammar=grammar,
            module_names=module_names,
            artifact_root=artifact_root,
            baseline_cost=baseline_selection.dollar_cost_per_example,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            budget=budget,
            reserve_target_task_calls=confirmation_pair_calls,
        )
        if artifact_root:
            schema_pool_path = str(freeze_jsonl(schema_pool, artifact_root / "schemas" / "frozen_pool.jsonl"))
    proposal_usage = _proposal_usage_from_proposer(proposer)
    reflection_usage = _proposal_usage_delta(
        previous=recorded_proposal_usage,
        current=proposal_usage,
    )
    proposal_usage = {
        **proposal_usage,
        "optimizer_reflection_calls": int(reflection_usage.get("optimizer_proposal_calls", 0)),
    }
    _record_proposal_usage_delta(
        budget=budget,
        previous=recorded_proposal_usage,
        current=proposal_usage,
    )
    top_selection_results = tuple(
        select_top_k_by_lcb(
            list(selection_results),
            k=min(config.top_k_confirmation, len(selection_results)),
            use_field_bonus=False,
        )
    )
    primary_schema_id = top_selection_results[0].schema_id if top_selection_results else ""
    programs_by_schema_id = {
        candidate.schema_candidate.schema_id: candidate
        for candidate in compiled_candidates
        if candidate.schema_candidate
    }
    top_programs = tuple(
        programs_by_schema_id[result.schema_id]
        for result in top_selection_results
        if result.schema_id in programs_by_schema_id
    )
    if not top_programs:
        raise RuntimeError("no schema candidates survived budgeted selection")
    primary_program = top_programs[0]
    primary_confirmation_min_calls = len(confirmation_examples) * primary_program.calls_per_example

    baseline_confirmation = _evaluate_required_with_budget(
        budget=budget,
        program=base_program,
        examples=confirmation_examples,
        scorer=scorer,
        config=config,
        method="fixed_schema_reference",
        candidate_id="fixed_schema_reference_confirmation",
        baseline_program=base_program,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        run_id="fixed_schema_reference_confirmation",
        stage="confirmation baseline",
        reserve_target_task_calls=primary_confirmation_min_calls,
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
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            budget=budget,
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
    heldout_baseline_result: CandidateEvalResult | None = None
    heldout_test_stats: PairedComparison | None = None
    heldout_pair_calls = (
        len(heldout_test_examples) * base_program.calls_per_example
        + len(heldout_test_examples) * primary_program.calls_per_example
    )
    heldout_pair_estimate = estimate_evaluation_budget(
        program=base_program,
        examples=heldout_test_examples,
        cost_meter=cost_meter,
    ).plus(
        estimate_evaluation_budget(
            program=primary_program,
            examples=heldout_test_examples,
            cost_meter=cost_meter,
        )
    )
    if heldout_test_examples and _can_start_budget(
        budget,
        min_target_task_calls=heldout_pair_calls,
        min_prompt_tokens=heldout_pair_estimate.prompt_tokens,
        min_completion_tokens=heldout_pair_estimate.completion_tokens,
        min_dollar_cost=heldout_pair_estimate.dollar_cost,
    ):
        heldout_baseline_result = _evaluate_required_with_budget(
            budget=budget,
            program=base_program,
            examples=heldout_test_examples,
            scorer=scorer,
            config=config,
            method="fixed_schema_reference",
            candidate_id="fixed_schema_reference_heldout_test",
            baseline_program=base_program,
            artifact_dir=artifact_root,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            run_id="fixed_schema_reference_heldout_test",
            stage="heldout baseline",
            reserve_target_task_calls=len(heldout_test_examples) * primary_program.calls_per_example,
        )
        heldout_test_result = _evaluate_required_with_budget(
            budget=budget,
            program=primary_program,
            examples=heldout_test_examples,
            scorer=scorer,
            config=config,
            method="schema_heldout_test",
            candidate_id=f"schema_heldout_test_{primary_confirmation.schema_id}",
            baseline_program=base_program,
            artifact_dir=artifact_root,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            schema_generation_calls=0,
            run_id=f"schema_heldout_test_{primary_confirmation.schema_id}",
            stage="heldout primary",
        )
        heldout_test_result.baseline_dollar_cost_per_example = heldout_baseline_result.dollar_cost_per_example
        heldout_test_stats = compare_paired(
            heldout_baseline_result.per_example_scores,
            heldout_test_result.per_example_scores,
            n_resamples=config.bootstrap_resamples,
            n_swaps=config.randomization_swaps,
            seed=config.seed + 1000,
        )
    with progress_status("field ablations", mode=config.progress):
        field_ablation_results = tuple(
            run_field_use_ablations(
                program=primary_program,
                examples=confirmation_examples,
                scorer=scorer,
                unablated_result=primary_confirmation,
                fields=primary_program.schema_candidate.evolved_field_names if primary_program.schema_candidate else (),
                seed=config.seed,
                artifact_dir=artifact_root,
                cost_meter=cost_meter,
                budget=budget,
                progress=config.progress,
            )
        )
    decision = _make_decision(
        baseline=baseline_confirmation,
        best=primary_confirmation,
        paired_stats=paired_stats,
        field_ablation_results=field_ablation_results,
        config=config,
    )
    cost_summary = _cost_summary(
        eval_results=(
            baseline_selection,
            *smoke_results,
            *selection_results,
            baseline_confirmation,
            *confirmation_results,
            *((heldout_baseline_result,) if heldout_baseline_result else ()),
            *((heldout_test_result,) if heldout_test_result else ()),
        ),
        field_ablation_results=field_ablation_results,
        proposal_usage=proposal_usage,
    )
    budget_summary = budget.summary()
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
                "cost_summary": cost_summary,
                "budget": budget_summary,
                "proposal_usage": proposal_usage,
                "reflection_rounds": list(reflection_rounds),
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
        cost_summary=cost_summary,
        budget_summary=budget_summary,
        proposal_usage=proposal_usage,
        reflection_rounds=reflection_rounds,
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


@dataclass(frozen=True)
class _CandidateEvalJob:
    candidate: LMProgram
    examples: tuple[ProgramExample, ...]
    scorer: Scorer
    config: FixedPoolConfig
    base_program: LMProgram
    artifact_root: Path | None
    method: str
    baseline_cost: float
    index: int


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
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
    budget: BudgetTracker,
    reserve_target_task_calls: int = 0,
) -> list[CandidateEvalResult]:
    if budget.limits.enabled:
        return _evaluate_many_budgeted_serial(
            candidates=candidates,
            examples=examples,
            scorer=scorer,
            config=config,
            base_program=base_program,
            artifact_root=artifact_root,
            method=method,
            baseline_cost=baseline_cost,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            budget=budget,
            reserve_target_task_calls=reserve_target_task_calls,
        )
    runnable = _candidate_jobs(
        candidates=candidates,
        examples=examples,
        scorer=scorer,
        config=config,
        base_program=base_program,
        artifact_root=artifact_root,
        method=method,
        baseline_cost=baseline_cost,
    )
    if not runnable:
        return []
    if config.workers > 1:
        return _evaluate_many_parallel(
            jobs=runnable,
            budget=budget,
            progress=config.progress,
            workers=config.workers,
            description=method,
        )
    return _evaluate_many_serial(
        jobs=runnable,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        budget=budget,
        progress=config.progress,
        description=method,
    )


def _candidate_jobs(
    *,
    candidates: tuple[LMProgram, ...],
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    base_program: LMProgram,
    artifact_root: Path | None,
    method: str,
    baseline_cost: float,
) -> list[_CandidateEvalJob]:
    jobs: list[_CandidateEvalJob] = []
    for index, candidate in enumerate(candidates):
        jobs.append(
            _CandidateEvalJob(
                candidate=candidate,
                examples=examples,
                scorer=scorer,
                config=config,
                base_program=base_program,
                artifact_root=artifact_root,
                method=method,
                baseline_cost=baseline_cost,
                index=index,
            )
        )
    return jobs


def _evaluate_many_budgeted_serial(
    *,
    candidates: tuple[LMProgram, ...],
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    base_program: LMProgram,
    artifact_root: Path | None,
    method: str,
    baseline_cost: float,
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
    budget: BudgetTracker,
    reserve_target_task_calls: int = 0,
) -> list[CandidateEvalResult]:
    results: list[CandidateEvalResult] = []
    indexed_candidates = list(enumerate(candidates))
    for index, candidate in progress_iter(
        indexed_candidates,
        total=len(indexed_candidates),
        description=method,
        mode=config.progress,
    ):
        min_calls = len(examples) * candidate.calls_per_example
        estimate = estimate_evaluation_budget(
            program=candidate,
            examples=examples,
            cost_meter=cost_meter,
        )
        if not _can_start_budget(
            budget,
            min_target_task_calls=min_calls,
            min_prompt_tokens=estimate.prompt_tokens,
            min_completion_tokens=estimate.completion_tokens,
            min_dollar_cost=estimate.dollar_cost,
            reserve_target_task_calls=reserve_target_task_calls,
        ):
            break
        job = _CandidateEvalJob(
            candidate=candidate,
            examples=examples,
            scorer=scorer,
            config=config,
            base_program=base_program,
            artifact_root=artifact_root,
            method=method,
            baseline_cost=baseline_cost,
            index=index,
        )
        result = _evaluate_candidate_job(
            job,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
        )
        budget.record_result(result)
        results.append(result)
    return results


def _evaluate_many_serial(
    *,
    jobs: list[_CandidateEvalJob],
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
    budget: BudgetTracker,
    progress: ProgressMode,
    description: str,
) -> list[CandidateEvalResult]:
    results: list[CandidateEvalResult] = []
    for job in progress_iter(
        jobs,
        total=len(jobs),
        description=description,
        mode=progress,
    ):
        result = _evaluate_candidate_job(
            job,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
        )
        budget.record_result(result)
        results.append(result)
    return results


def _evaluate_many_parallel(
    *,
    jobs: list[_CandidateEvalJob],
    budget: BudgetTracker,
    progress: ProgressMode,
    workers: int,
    description: str,
) -> list[CandidateEvalResult]:
    max_workers = min(workers, len(jobs))
    if not _jobs_are_process_picklable(jobs):
        return _evaluate_many_thread_pool(
            jobs=jobs,
            budget=budget,
            progress=progress,
            max_workers=max_workers,
            description=description,
        )
    return _evaluate_many_process_pool(
        jobs=jobs,
        budget=budget,
        progress=progress,
        max_workers=max_workers,
        description=description,
    )


def _evaluate_many_process_pool(
    *,
    jobs: list[_CandidateEvalJob],
    budget: BudgetTracker,
    progress: ProgressMode,
    max_workers: int,
    description: str,
) -> list[CandidateEvalResult]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(_evaluate_candidate_job_in_worker, jobs)
        results = list(
            progress_iter(
                mapped,
                total=len(jobs),
                description=f"{description} [{max_workers} processes]",
                mode=progress,
            )
        )
    for result in results:
        budget.record_result(result)
    return results


def _evaluate_many_thread_pool(
    *,
    jobs: list[_CandidateEvalJob],
    budget: BudgetTracker,
    progress: ProgressMode,
    max_workers: int,
    description: str,
) -> list[CandidateEvalResult]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        mapped = executor.map(_evaluate_candidate_job_in_worker, jobs)
        results = list(
            progress_iter(
                mapped,
                total=len(jobs),
                description=f"{description} [{max_workers} threads]",
                mode=progress,
            )
        )
    for result in results:
        budget.record_result(result)
    return results


def _jobs_are_process_picklable(jobs: list[_CandidateEvalJob]) -> bool:
    try:
        pickle.dumps(jobs)
    except Exception:
        return False
    return True


def _evaluate_candidate_job_in_worker(job: _CandidateEvalJob) -> CandidateEvalResult:
    cost_meter = make_cost_meter(
        model_prices=job.config.model_prices,
        use_tiktoken=job.config.use_tiktoken_costing,
    )
    rollout_cache = RolloutCache(job.artifact_root / "rollout_cache" if job.artifact_root else None)
    return _evaluate_candidate_job(job, cost_meter=cost_meter, rollout_cache=rollout_cache)


def _evaluate_candidate_job(
    job: _CandidateEvalJob,
    *,
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
) -> CandidateEvalResult:
    candidate = job.candidate
    schema_id = candidate.schema_candidate.schema_id if candidate.schema_candidate else "original_schema"
    result = evaluate_program(
        program=candidate,
        examples=job.examples,
        scorer=job.scorer,
        method=job.method,
        candidate_id=f"{job.method}_{job.index}",
        seed=job.config.seed,
        baseline_program=job.base_program,
        strict_invalid_policy=job.config.strict_invalid_policy,
        artifact_dir=job.artifact_root,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        schema_generation_calls=1,
        run_id=f"{job.method}_{schema_id}_{job.index}",
    )
    result.baseline_dollar_cost_per_example = job.baseline_cost
    return result


def _evaluate_required_with_budget(
    *,
    budget: BudgetTracker,
    program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    method: str,
    candidate_id: str,
    baseline_program: LMProgram,
    artifact_dir: Path | None,
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
    run_id: str,
    stage: str,
    reserve_target_task_calls: int = 0,
    schema_generation_calls: int = 0,
) -> CandidateEvalResult:
    min_calls = len(examples) * program.calls_per_example
    estimate = estimate_evaluation_budget(
        program=program,
        examples=examples,
        cost_meter=cost_meter,
    )
    if not _can_start_budget(
        budget,
        min_target_task_calls=min_calls,
        min_prompt_tokens=estimate.prompt_tokens,
        min_completion_tokens=estimate.completion_tokens,
        min_dollar_cost=estimate.dollar_cost,
        reserve_target_task_calls=reserve_target_task_calls,
    ):
        raise RuntimeError(f"budget exhausted before {stage}: {budget.summary()}")
    with progress_status(stage, mode=config.progress):
        result = evaluate_program(
            program=program,
            examples=examples,
            scorer=scorer,
            method=method,
            candidate_id=candidate_id,
            seed=config.seed,
            baseline_program=baseline_program,
            strict_invalid_policy=config.strict_invalid_policy,
            artifact_dir=artifact_dir,
            cost_meter=cost_meter,
            rollout_cache=rollout_cache,
            schema_generation_calls=schema_generation_calls,
            run_id=run_id,
        )
    budget.record_result(result)
    return result


def _can_start_budget(
    budget: BudgetTracker,
    *,
    min_target_task_calls: int = 0,
    min_prompt_tokens: int = 0,
    min_completion_tokens: int = 0,
    min_dollar_cost: float = 0.0,
    reserve_target_task_calls: int = 0,
    reserve_estimate: EvaluationBudgetEstimate | None = None,
) -> bool:
    reserve = reserve_estimate or EvaluationBudgetEstimate()
    return (not budget.exhausted) and budget.can_start(
        min_target_task_calls=min_target_task_calls
        + reserve_target_task_calls
        + reserve.target_task_calls,
        min_prompt_tokens=min_prompt_tokens + reserve.prompt_tokens,
        min_completion_tokens=min_completion_tokens + reserve.completion_tokens,
        min_dollar_cost=min_dollar_cost + reserve.dollar_cost,
    )


def _make_budget_tracker(config: FixedPoolConfig) -> BudgetTracker:
    return BudgetTracker(
        BudgetLimits(
            max_target_task_calls=config.max_target_task_calls,
            max_prompt_tokens=config.max_prompt_tokens,
            max_completion_tokens=config.max_completion_tokens,
            max_total_tokens=config.max_total_tokens,
            max_dollar_cost=config.max_dollar_cost,
        )
    )


def _cost_summary(
    *,
    eval_results: tuple[CandidateEvalResult, ...],
    field_ablation_results: tuple[FieldAblationResult, ...],
    proposal_usage: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    usage = proposal_usage or _empty_proposal_usage()
    latency_results = (*eval_results, *field_ablation_results)
    return {
        "evaluations": len(eval_results) + len(field_ablation_results),
        "target_task_calls": sum(result.target_task_calls for result in eval_results)
        + sum(result.target_task_calls for result in field_ablation_results),
        "optimizer_proposal_calls": sum(result.optimizer_proposal_calls for result in eval_results)
        + int(usage.get("optimizer_proposal_calls", 0)),
        "optimizer_reflection_calls": sum(result.optimizer_reflection_calls for result in eval_results)
        + int(usage.get("optimizer_reflection_calls", 0)),
        "schema_generation_calls": sum(result.schema_generation_calls for result in eval_results),
        "schema_validation_repair_calls": sum(
            result.schema_validation_repair_calls for result in eval_results
        ),
        "retriever_calls": sum(result.retriever_calls for result in eval_results),
        "prompt_tokens": sum(result.prompt_tokens for result in eval_results)
        + sum(result.prompt_tokens for result in field_ablation_results)
        + int(usage.get("prompt_tokens", 0)),
        "completion_tokens": sum(result.completion_tokens for result in eval_results)
        + sum(result.completion_tokens for result in field_ablation_results)
        + int(usage.get("completion_tokens", 0)),
        "total_tokens": sum(result.prompt_tokens + result.completion_tokens for result in eval_results)
        + sum(
            result.prompt_tokens + result.completion_tokens
            for result in field_ablation_results
        )
        + int(usage.get("total_tokens", 0)),
        "dollar_cost": sum(result.dollar_cost for result in eval_results)
        + sum(result.dollar_cost for result in field_ablation_results)
        + float(usage.get("dollar_cost", 0.0)),
        "wall_clock_seconds": sum(result.wall_clock_seconds for result in latency_results),
        "max_p50_latency_ms": max(
            (result.p50_latency_ms for result in latency_results),
            default=0.0,
        ),
        "max_p95_latency_ms": max(
            (result.p95_latency_ms for result in latency_results),
            default=0.0,
        ),
    }


def _proposal_usage_from_proposer(proposer: SchemaProposer | None) -> dict[str, float | int]:
    usage = getattr(proposer, "total_usage", None)
    if not isinstance(usage, dict):
        usage = getattr(proposer, "last_usage", None)
    if not isinstance(usage, dict):
        return _empty_proposal_usage()
    normalized = _empty_proposal_usage()
    for key in normalized:
        if key in usage:
            normalized[key] = usage[key]
    return normalized


def _reset_proposal_usage(proposer: SchemaProposer | None) -> None:
    if proposer is None:
        return
    if hasattr(proposer, "last_usage"):
        setattr(proposer, "last_usage", _empty_proposal_usage())
    if hasattr(proposer, "total_usage"):
        setattr(proposer, "total_usage", _empty_proposal_usage())


def _record_proposal_usage_delta(
    *,
    budget: BudgetTracker,
    previous: dict[str, float | int],
    current: dict[str, float | int],
) -> None:
    delta = _proposal_usage_delta(previous=previous, current=current)
    _record_proposal_usage(budget, delta)


def _proposal_usage_delta(
    *,
    previous: dict[str, float | int],
    current: dict[str, float | int],
) -> dict[str, float | int]:
    delta = _empty_proposal_usage()
    for key in delta:
        if key == "dollar_cost":
            delta[key] = max(0.0, float(current.get(key, 0.0)) - float(previous.get(key, 0.0)))
        else:
            delta[key] = max(0, int(current.get(key, 0)) - int(previous.get(key, 0)))
    return delta


def _record_proposal_usage(budget: BudgetTracker, usage: dict[str, float | int]) -> None:
    result = type("ProposalUsageResult", (), {})()
    result.target_task_calls = 0
    result.prompt_tokens = int(usage.get("prompt_tokens", 0))
    result.completion_tokens = int(usage.get("completion_tokens", 0))
    result.dollar_cost = float(usage.get("dollar_cost", 0.0))
    budget.record_result(result)


def _empty_proposal_usage() -> dict[str, float | int]:
    return {
        "optimizer_proposal_calls": 0,
        "optimizer_reflection_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "dollar_cost": 0.0,
    }


def _run_reflection_rounds(
    *,
    base_program: LMProgram,
    schema_pool: tuple[SchemaCandidate, ...],
    compiled_candidates: tuple[LMProgram, ...],
    smoke_results: tuple[CandidateEvalResult, ...],
    selection_results: tuple[CandidateEvalResult, ...],
    train_traces: tuple[TraceExample, ...],
    selection_examples: tuple[ProgramExample, ...],
    smoke_examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: FixedPoolConfig,
    proposer: SchemaProposer,
    grammar: SchemaGrammar,
    module_names: tuple[str, ...],
    artifact_root: Path | None,
    baseline_cost: float,
    cost_meter: CostMeter,
    rollout_cache: RolloutCache,
    budget: BudgetTracker,
    reserve_target_task_calls: int = 0,
) -> tuple[
    tuple[SchemaCandidate, ...],
    tuple[LMProgram, ...],
    tuple[CandidateEvalResult, ...],
    tuple[CandidateEvalResult, ...],
    tuple[dict[str, Any], ...],
]:
    metadata: list[dict[str, Any]] = []
    if any(trace.split != "train" for trace in train_traces):
        return (
            schema_pool,
            compiled_candidates,
            smoke_results,
            selection_results,
            (
                {
                    "round": 1,
                    "status": "skipped",
                    "reason": "reflection requires train split traces",
                },
            ),
        )

    existing_schema_ids = {schema.schema_id for schema in schema_pool}
    current_pool = list(schema_pool)
    current_programs = list(compiled_candidates)
    current_smoke = list(smoke_results)
    current_selection = list(selection_results)
    schemas_per_round = config.reflection_schemas_per_round or max(
        1,
        config.n_trace_schemas // max(1, config.reflection_rounds),
    )

    for round_index in range(1, config.reflection_rounds):
        top = select_top_k_by_lcb(
            list(current_selection),
            k=1,
            use_field_bonus=False,
        )
        if not top:
            metadata.append({"round": round_index, "status": "skipped", "reason": "no selection results"})
            break
        primary = top[0]
        failure_traces = _reflection_traces_from_train_traces(
            traces=train_traces,
            primary_schema_id=primary.schema_id,
            max_traces=max(1, schemas_per_round * 2),
        )
        if not failure_traces:
            metadata.append(
                {
                    "round": round_index,
                    "status": "skipped",
                    "reason": "no train traces available for reflection",
                    "primary_schema_id": primary.schema_id,
                }
            )
            break
        with progress_status(f"reflection {round_index} schema proposal", mode=config.progress):
            proposed = propose_schemas_from_traces(
                traces=failure_traces,
                task=config.task,
                module_names=module_names,
                n=schemas_per_round,
                seed=config.seed + 10_000 + round_index,
                schema_token_budget=config.schema_token_budget,
                proposer=proposer,
            )
        with progress_status(f"reflection {round_index} static schema checks", mode=config.progress):
            filtered = [
                schema
                for schema in _static_filter(tuple(proposed), grammar)
                if schema.schema_id not in existing_schema_ids
            ]
        if not filtered:
            metadata.append(
                {
                    "round": round_index,
                    "status": "skipped",
                    "reason": "no new grammar-valid schemas",
                    "primary_schema_id": primary.schema_id,
                    "failure_traces": len(failure_traces),
                }
            )
            continue
        new_programs = tuple(
            _compile_candidate(base_program=base_program, schema=schema, config=config)
            for schema in progress_iter(
                filtered,
                total=len(filtered),
                description=f"reflection {round_index} compile schemas",
                mode=config.progress,
            )
        )
        new_smoke = (
            tuple(
                _evaluate_many(
                    candidates=new_programs,
                    examples=smoke_examples,
                    scorer=scorer,
                    config=config,
                    base_program=base_program,
                    artifact_root=artifact_root,
                    method=f"schema_reflection_{round_index}_smoke",
                    baseline_cost=baseline_cost,
                    cost_meter=cost_meter,
                    rollout_cache=rollout_cache,
                    budget=budget,
                    reserve_target_task_calls=reserve_target_task_calls,
                )
            )
            if smoke_examples
            else ()
        )
        smoke_pass = {
            result.schema_id
            for result in new_smoke
            if 1.0 - result.invalid_output_rate >= config.min_smoke_validity
        }
        if new_smoke:
            new_programs = tuple(
                program
                for program in new_programs
                if program.schema_candidate and program.schema_candidate.schema_id in smoke_pass
            )
        if not new_programs:
            current_smoke.extend(new_smoke)
            metadata.append(
                {
                    "round": round_index,
                    "status": "skipped",
                    "reason": "no reflected schemas survived smoke",
                    "primary_schema_id": primary.schema_id,
                    "proposed": len(filtered),
                }
            )
            continue
        new_schema_ids = {
            program.schema_candidate.schema_id
            for program in new_programs
            if program.schema_candidate
        }
        current_pool.extend(schema for schema in filtered if schema.schema_id in new_schema_ids)
        current_programs.extend(new_programs)
        current_smoke.extend(new_smoke)
        new_selection = tuple(
            _evaluate_many(
                candidates=new_programs,
                examples=selection_examples,
                scorer=scorer,
                config=config,
                base_program=base_program,
                artifact_root=artifact_root,
                method=f"schema_reflection_{round_index}_selection",
                baseline_cost=baseline_cost,
                cost_meter=cost_meter,
                rollout_cache=rollout_cache,
                budget=budget,
                reserve_target_task_calls=reserve_target_task_calls,
            )
        )
        current_selection.extend(new_selection)
        existing_schema_ids.update(new_schema_ids)
        metadata.append(
            {
                "round": round_index,
                "status": "evaluated",
                "primary_schema_id": primary.schema_id,
                "failure_traces": len(failure_traces),
                "proposed": len(filtered),
                "survived_smoke": len(new_programs),
                "selection_results": len(new_selection),
            }
        )

    return (
        tuple(current_pool),
        tuple(current_programs),
        tuple(current_smoke),
        tuple(current_selection),
        tuple(metadata),
    )


def _reflection_traces_from_train_traces(
    *,
    traces: tuple[TraceExample, ...],
    primary_schema_id: str,
    max_traces: int,
) -> tuple[TraceExample, ...]:
    prioritized = [
        trace
        for trace in traces
        if trace.errors or (trace.score is not None and trace.score < 1.0)
    ] or list(traces)
    selected: list[TraceExample] = []
    for trace in prioritized[:max_traces]:
        metadata = dict(trace.metadata or {})
        metadata.update(
            {
                "source": "schemaevo_reflection_train_trace",
                "primary_schema_id": primary_schema_id,
            }
        )
        selected.append(
            TraceExample(
                example_id=trace.example_id,
                split="train",
                module_name=trace.module_name,
                input_summary=trace.input_summary,
                output_summary=trace.output_summary,
                score=trace.score,
                errors=trace.errors or ("train_reflection_trace",),
                metadata=metadata,
            )
        )
    return tuple(selected)


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
