from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
from typing import Any

from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.cost_ledger import BudgetLimits, BudgetTracker, CostMeter, make_cost_meter
from schemaevo.eval.scoring import CandidateEvalResult, Scorer, evaluate_program
from schemaevo.optimizers.minibatching import StratifyKey, sample_minibatch
from schemaevo.optimizers.pareto import CandidateRecord, ParetoFront
from schemaevo.optimizers.selection import schema_selection_value
from schemaevo.programs.base import LMProgram, ProgramExample
from schemaevo.programs.call_graph import assert_same_call_graph
from schemaevo.programs.compile_schema_program import compile_schema_program
from schemaevo.schemas.candidate import SchemaCandidate, SchemaField
from schemaevo.schemas.grammar import SchemaGrammar
from schemaevo.schemas.human_templates import make_human_minimal_schemas
from schemaevo.schemas.mutations import Mutation, apply_mutation
from schemaevo.schemas.random_controls import make_random_schema_controls
from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class SchemaEvoConfig:
    task: str = "HoVer"
    seed: int = 0
    max_program_rollouts: int = 25
    max_mutation_attempts: int = 250
    minibatch_size: int = 32
    shared_eval_batch: bool = True
    schema_token_budget: int = 512
    initial_random_schemas: int = 4
    k_final: int = 5
    freeze_prompt_text: bool = True
    allow_prompt_mutation: bool = False
    strict_invalid_policy: bool = True
    min_static_checks: bool = True
    enable_schema_merge: bool = True
    allocation_strategy: str = "single_batch"
    successive_halving_eta: int = 2
    successive_halving_min_batch_size: int = 0
    successive_halving_promote_fraction: float = 0.5
    parent_selection_strategy: str = "ucb"
    parent_ucb_exploration: float = 0.25
    operator_ucb_exploration: float = 0.4
    use_tiktoken_costing: bool = False
    model_prices: dict[str, dict[str, float | str]] = field(default_factory=dict)
    max_target_task_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_dollar_cost: float | None = None

    def __post_init__(self) -> None:
        if self.max_program_rollouts <= 0:
            raise ValueError("max_program_rollouts must be positive")
        if self.max_mutation_attempts <= 0:
            raise ValueError("max_mutation_attempts must be positive")
        if self.max_mutation_attempts < self.max_program_rollouts:
            raise ValueError("max_mutation_attempts must be at least max_program_rollouts")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.schema_token_budget <= 0:
            raise ValueError("schema_token_budget must be positive")
        if self.initial_random_schemas < 0:
            raise ValueError("initial_random_schemas must be non-negative")
        if self.k_final <= 0:
            raise ValueError("k_final must be positive")
        if self.freeze_prompt_text and self.allow_prompt_mutation:
            raise ValueError("freeze_prompt_text and allow_prompt_mutation cannot both be true")
        if self.allocation_strategy not in {"single_batch", "successive_halving"}:
            raise ValueError("allocation_strategy must be 'single_batch' or 'successive_halving'")
        if self.successive_halving_eta <= 1:
            raise ValueError("successive_halving_eta must be greater than 1")
        if self.successive_halving_min_batch_size < 0:
            raise ValueError("successive_halving_min_batch_size must be non-negative")
        if not 0.0 < self.successive_halving_promote_fraction <= 1.0:
            raise ValueError("successive_halving_promote_fraction must be in (0, 1]")
        if self.parent_selection_strategy not in {"uniform_top_k", "ucb", "thompson"}:
            raise ValueError("parent_selection_strategy must be 'uniform_top_k', 'ucb', or 'thompson'")
        if self.parent_ucb_exploration < 0:
            raise ValueError("parent_ucb_exploration must be non-negative")
        if self.operator_ucb_exploration < 0:
            raise ValueError("operator_ucb_exploration must be non-negative")


@dataclass(frozen=True)
class SchemaEvoRunResult:
    baseline_result: CandidateEvalResult
    promotion_baseline_result: CandidateEvalResult | None
    evaluated_records: tuple[CandidateRecord, ...]
    pareto_records: tuple[CandidateRecord, ...]
    final_records: tuple[CandidateRecord, ...]
    rejected_schemas: tuple[dict[str, Any], ...]
    operator_weights: dict[str, float]
    operator_counts: dict[str, int]
    budget_summary: dict[str, float | int | bool | None]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_mean": self.baseline_result.mean_score,
            "promotion_baseline_mean": (
                self.promotion_baseline_result.mean_score if self.promotion_baseline_result else None
            ),
            "evaluated": len(self.evaluated_records),
            "pareto_size": len(self.pareto_records),
            "final_schema_ids": [record.schema.schema_id for record in self.final_records],
            "final_scores": [record.result.mean_score for record in self.final_records],
            "final_stages": [record.stage for record in self.final_records],
            "rejected_schemas": len(self.rejected_schemas),
            "operator_weights": self.operator_weights,
            "operator_counts": self.operator_counts,
            "budget": self.budget_summary,
            "artifacts": self.artifacts,
        }


def schema_evo_optimize(
    *,
    base_program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: SchemaEvoConfig,
    initial_schemas: tuple[SchemaCandidate, ...] = (),
    stratify_key: StratifyKey | None = None,
    artifact_dir: str | Path | None = None,
) -> SchemaEvoRunResult:
    if not examples:
        raise ValueError("examples must be non-empty")
    artifact_root = Path(artifact_dir) if artifact_dir else None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
    rollout_cache = RolloutCache(artifact_root / "rollout_cache" if artifact_root else None)
    optimizer_examples = sample_minibatch(
        examples=examples,
        batch_size=min(config.minibatch_size, len(examples)),
        seed=config.seed,
        stratify_key=stratify_key,
    )
    stage_examples = _initial_allocation_examples(optimizer_examples, config=config)
    cost_meter = _make_cost_meter(config)
    budget = BudgetTracker(
        BudgetLimits(
            max_target_task_calls=config.max_target_task_calls,
            max_prompt_tokens=config.max_prompt_tokens,
            max_completion_tokens=config.max_completion_tokens,
            max_total_tokens=config.max_total_tokens,
            max_dollar_cost=config.max_dollar_cost,
        )
    )

    grammar = SchemaGrammar(
        allowed_modules=base_program.module_names,
        max_schema_tokens=config.schema_token_budget,
    )
    rng = random.Random(config.seed)
    baseline_result = evaluate_program(
        program=base_program,
        examples=stage_examples,
        scorer=scorer,
        method="schemaevo_baseline",
        candidate_id="schemaevo_baseline",
        seed=config.seed,
        baseline_program=base_program,
        strict_invalid_policy=config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        run_id="schemaevo_baseline",
    )
    budget.record_result(baseline_result)

    population_schemas = _initialize_population(
        task=config.task,
        module_names=base_program.module_names,
        seed=config.seed,
        initial_schemas=initial_schemas,
        initial_random_schemas=config.initial_random_schemas,
        schema_token_budget=config.schema_token_budget,
    )

    pareto = ParetoFront()
    evaluated: list[CandidateRecord] = []
    rejected: list[dict[str, Any]] = []
    schema_by_id: dict[str, SchemaCandidate] = {}
    operator_bandit: dict[str, float] = {}
    operator_stats: dict[str, dict[str, float]] = {}
    rollouts_used = 0
    mutation_attempts = 0
    search_rollout_limit = _search_rollout_limit(
        total_rollouts=config.max_program_rollouts,
        optimizer_examples=optimizer_examples,
        stage_examples=stage_examples,
        config=config,
    )

    for schema in population_schemas:
        if rollouts_used >= search_rollout_limit or budget.exhausted:
            break
        record = _compile_and_eval(
            base_program=base_program,
            schema=schema,
            examples=stage_examples,
            scorer=scorer,
            config=config,
            grammar=grammar,
            stratify_key=stratify_key,
            artifact_root=artifact_root,
            rollout_cache=rollout_cache,
            step=rollouts_used,
            baseline_cost=baseline_result.dollar_cost_per_example,
            mutation=None,
            cost_meter=cost_meter,
            budget=budget,
            stage="initial",
        )
        if isinstance(record, dict):
            rejected.append(record)
            continue
        evaluated.append(record)
        pareto.update(record)
        schema_by_id[record.schema.schema_id] = record.schema
        rollouts_used += 1

    mutation_rollout_limit = (
        max(1, search_rollout_limit - 1)
        if config.enable_schema_merge
        else search_rollout_limit
    )
    while (
        rollouts_used < mutation_rollout_limit
        and mutation_attempts < config.max_mutation_attempts
        and evaluated
        and not budget.exhausted
    ):
        mutation_attempts += 1
        parent = _select_parent(
            evaluated,
            rng,
            strategy=config.parent_selection_strategy,
            exploration=config.parent_ucb_exploration,
        )
        operator_bandit = _operator_ucb_weights(
            operator_stats,
            exploration=config.operator_ucb_exploration,
        )
        mutation = sample_schema_mutation(
            parent.schema,
            task=config.task,
            module_names=base_program.module_names,
            seed=rng.randrange(1_000_000_000),
            operator_weights=operator_bandit,
        )
        try:
            child_schema = apply_mutation(parent.schema, mutation)
        except Exception as exc:
            rejected.append(
                {
                    "parent_schema_id": parent.schema.schema_id,
                    "mutation": mutation.op.value,
                    "reason": f"mutation_failed: {exc}",
                }
            )
            continue
        if child_schema.schema_id in schema_by_id:
            rejected.append(
                {
                    "parent_schema_id": parent.schema.schema_id,
                    "schema_id": child_schema.schema_id,
                    "mutation": mutation.op.value,
                    "reason": "duplicate_schema",
                }
            )
            continue
        record = _compile_and_eval(
            base_program=base_program,
            schema=child_schema,
            examples=stage_examples if config.shared_eval_batch else examples,
            scorer=scorer,
            config=config,
            grammar=grammar,
            stratify_key=stratify_key,
            artifact_root=artifact_root,
            rollout_cache=rollout_cache,
            step=rollouts_used,
            baseline_cost=baseline_result.dollar_cost_per_example,
            mutation=mutation,
            cost_meter=cost_meter,
            budget=budget,
            stage="initial",
        )
        if isinstance(record, dict):
            rejected.append(record)
            continue
        evaluated.append(record)
        pareto.update(record)
        schema_by_id[record.schema.schema_id] = record.schema
        _update_operator_bandit(
            operator_bandit,
            mutation.op.value,
            reward=record.result.mean_score - parent.result.mean_score,
        )
        _update_operator_stats(
            operator_stats,
            mutation.op.value,
            reward=record.result.mean_score - parent.result.mean_score,
        )
        rollouts_used += 1

    if config.enable_schema_merge and rollouts_used < search_rollout_limit and len(pareto.records) >= 2:
        merged_schema = merge_schema_candidates(
            tuple(record.schema for record in pareto.top(2)),
            task=config.task,
            seed=config.seed,
            schema_token_budget=config.schema_token_budget,
        )
        if merged_schema.schema_id not in schema_by_id:
            record = _compile_and_eval(
                base_program=base_program,
                schema=merged_schema,
                examples=stage_examples,
                scorer=scorer,
                config=config,
                grammar=grammar,
                stratify_key=stratify_key,
                artifact_root=artifact_root,
                rollout_cache=rollout_cache,
                step=rollouts_used,
                baseline_cost=baseline_result.dollar_cost_per_example,
                mutation=None,
                cost_meter=cost_meter,
                budget=budget,
                stage="initial_merge",
            )
            if isinstance(record, dict):
                rejected.append({"schema_id": merged_schema.schema_id, **record})
            else:
                evaluated.append(record)
                pareto.update(record)
                schema_by_id[record.schema.schema_id] = record.schema
                rollouts_used += 1

    promotion_baseline_result: CandidateEvalResult | None = None
    final_pareto = pareto
    if _should_promote(
        optimizer_examples=optimizer_examples,
        stage_examples=stage_examples,
        config=config,
    ) and evaluated and rollouts_used < config.max_program_rollouts and not budget.exhausted:
        promotions = _select_promotions(
            evaluated,
            remaining_rollouts=config.max_program_rollouts - rollouts_used,
            config=config,
        )
        if promotions:
            promotion_baseline_result = evaluate_program(
                program=base_program,
                examples=optimizer_examples,
                scorer=scorer,
                method="schemaevo_promotion_baseline",
                candidate_id="schemaevo_promotion_baseline",
                seed=config.seed + 10_000,
                baseline_program=base_program,
                strict_invalid_policy=config.strict_invalid_policy,
                artifact_dir=artifact_root,
                cost_meter=cost_meter,
                rollout_cache=rollout_cache,
                run_id="schemaevo_promotion_baseline",
            )
            budget.record_result(promotion_baseline_result)
            promoted_pareto = ParetoFront()
            for source_record in promotions:
                if rollouts_used >= config.max_program_rollouts or budget.exhausted:
                    break
                promoted = _compile_and_eval(
                    base_program=base_program,
                    schema=source_record.schema,
                    examples=optimizer_examples,
                    scorer=scorer,
                    config=config,
                    grammar=grammar,
                    stratify_key=stratify_key,
                    artifact_root=artifact_root,
                    rollout_cache=rollout_cache,
                    step=rollouts_used,
                    baseline_cost=promotion_baseline_result.dollar_cost_per_example,
                    mutation=source_record.mutation,
                    cost_meter=cost_meter,
                    budget=budget,
                    stage="promotion",
                )
                if isinstance(promoted, dict):
                    rejected.append(promoted)
                    continue
                evaluated.append(promoted)
                promoted_pareto.update(promoted)
                rollouts_used += 1
            if promoted_pareto.records:
                final_pareto = promoted_pareto

    operator_bandit = _operator_ucb_weights(
        operator_stats,
        exploration=config.operator_ucb_exploration,
    )
    operator_counts = {
        operator: int(stats.get("count", 0))
        for operator, stats in operator_stats.items()
    }
    budget_summary = budget.summary()
    final_records = final_pareto.top(config.k_final)
    artifacts: dict[str, str] = {}
    if artifact_root:
        summary_path = write_json(
            {
                "config": asdict(config),
                "summary": {
                    "baseline_mean": baseline_result.mean_score,
                    "promotion_baseline_mean": (
                        promotion_baseline_result.mean_score if promotion_baseline_result else None
                    ),
                    "evaluated": len(evaluated),
                    "mutation_attempts": mutation_attempts,
                    "pareto_size": len(final_pareto.records),
                    "final_schema_ids": [record.schema.schema_id for record in final_records],
                    "final_stages": [record.stage for record in final_records],
                    "rejected_schemas": rejected,
                    "operator_weights": operator_bandit,
                    "operator_counts": operator_counts,
                    "budget": budget_summary,
                },
            },
            artifact_root / "results" / "schemaevo_summary.json",
        )
        artifacts["summary"] = str(summary_path)

    return SchemaEvoRunResult(
        baseline_result=baseline_result,
        promotion_baseline_result=promotion_baseline_result,
        evaluated_records=tuple(evaluated),
        pareto_records=final_pareto.records,
        final_records=final_records,
        rejected_schemas=tuple(rejected),
        operator_weights=dict(operator_bandit),
        operator_counts=operator_counts,
        budget_summary=budget_summary,
        artifacts=artifacts,
    )


def sample_schema_mutation(
    parent: SchemaCandidate,
    *,
    task: str,
    module_names: tuple[str, ...],
    seed: int,
    operator_weights: dict[str, float] | None = None,
) -> Mutation:
    rng = random.Random(seed)
    fields = parent.all_fields
    producer = module_names[0]
    consumer = module_names[-1]
    missing_template_fields = _missing_template_fields(parent, task=task, module_names=module_names)
    enum_fields = [field for field in fields if field.type == "enum"]
    choices = ["toggle_required", "tighten_validator", "relax_validator"]
    if len(fields) > 1:
        choices.extend(["drop_field", "merge_fields"])
    if fields:
        choices.extend(
            [
                "rename_field",
                "change_type",
                "split_field",
                "move_field_to_earlier_module",
                "move_field_to_later_module",
                "add_downstream_consumption_rule",
                "drop_downstream_consumption_rule",
            ]
        )
    if missing_template_fields:
        choices.append("add_template_field")
    if enum_fields:
        choices.extend(["add_enum_value", "drop_enum_value"])
    action = _weighted_choice(rng, choices, operator_weights or {})

    if action == "add_template_field":
        field = rng.choice(missing_template_fields)
        return Mutation.from_parts(
            "add_field",
            module_name=field.producer_module,
            payload=field.to_dict(),
            rationale="add missing task-semantic template field",
        )
    if action == "drop_field":
        field = rng.choice(fields)
        return Mutation.from_parts(
            "drop_field",
            module_name=field.producer_module,
            field_name=field.name,
            rationale="remove low-utility or over-budget field",
        )
    if action == "rename_field":
        field = rng.choice(fields)
        return Mutation.from_parts(
            "rename_field",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"new_name": f"{field.name}_v{rng.randint(2, 99)}"},
            rationale="test a clearer field name while preserving semantics",
        )
    if action == "change_type":
        field = rng.choice(fields)
        new_type = rng.choice(
            [item for item in ("string", "boolean", "number", "integer", "array[string]", "object") if item != field.type]
        )
        return Mutation.from_parts(
            "change_type",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"new_type": new_type},
            rationale="test whether a tighter or simpler type improves validity/use",
        )
    if action == "add_enum_value":
        field = rng.choice(enum_fields)
        return Mutation.from_parts(
            "add_enum_value",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"value": f"other_{rng.randint(0, 99)}"},
            rationale="expand enum coverage for observed trace variation",
        )
    if action == "drop_enum_value":
        field = rng.choice(enum_fields)
        values = field.enum_values or ()
        value = rng.choice(values)
        return Mutation.from_parts(
            "drop_enum_value",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"value": value},
            rationale="tighten enum to reduce invalid or ambiguous states",
        )
    if action == "split_field":
        field = rng.choice(fields)
        return Mutation.from_parts(
            "split_field",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"new_names": (f"{field.name}_signal", f"{field.name}_evidence")},
            rationale="split a compound field into separately consumable signals",
        )
    if action == "merge_fields":
        selected = rng.sample(list(fields), 2)
        module_name = selected[0].producer_module
        same_module = [field for field in fields if field.producer_module == module_name]
        if len(same_module) >= 2:
            selected = rng.sample(same_module, 2)
        return Mutation.from_parts(
            "merge_fields",
            module_name=selected[0].producer_module,
            payload={
                "field_names": [field.name for field in selected],
                "new_name": f"{selected[0].name}_{selected[1].name}",
                "new_type": "object",
            },
            rationale="merge related fields to reduce downstream fragmentation",
        )
    if action == "move_field_to_earlier_module":
        field = rng.choice(fields)
        target = module_names[max(0, module_names.index(field.producer_module) - 1)]
        return Mutation.from_parts(
            "move_field_to_earlier_module",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"target_module": target},
            rationale="test whether producing the field earlier improves information flow",
        )
    if action == "move_field_to_later_module":
        field = rng.choice(fields)
        target = module_names[min(len(module_names) - 1, module_names.index(field.producer_module) + 1)]
        return Mutation.from_parts(
            "move_field_to_later_module",
            module_name=field.producer_module,
            field_name=field.name,
            payload={"target_module": target},
            rationale="test whether delaying production improves validity",
        )
    if action == "add_downstream_consumption_rule":
        field = rng.choice(fields)
        return Mutation.from_parts(
            "add_downstream_consumption_rule",
            module_name=consumer,
            field_name=field.name,
            payload={
                "instruction": f"Use `{field.name}` explicitly before finalizing the answer.",
                "required_behavior": "Consume the field without adding calls.",
                "fallback_if_missing": "Use original prompt behavior and mark uncertainty.",
            },
            rationale="make downstream consumption explicit",
        )
    if action == "drop_downstream_consumption_rule":
        if parent.consumption_rules:
            rule = rng.choice(parent.consumption_rules)
            return Mutation.from_parts(
                "drop_downstream_consumption_rule",
                module_name=rule.consumer_module,
                field_name=rule.field_name,
                rationale="test whether the downstream rule was unnecessary or harmful",
            )
        action = "tighten_validator"
    if action == "tighten_validator":
        field = rng.choice(fields) if fields else None
        if field is None:
            return Mutation.from_parts(
                "add_downstream_consumption_rule",
                module_name=consumer,
                field_name="missing_evidence_reason",
                payload={
                    "instruction": "Use the missing evidence reason if present.",
                    "required_behavior": "Do not add calls; consume only existing schema fields.",
                    "fallback_if_missing": "Use original prompt behavior.",
                },
                rationale="add default consumption rule",
            )
        return Mutation.from_parts(
            "tighten_validator",
            field_name=field.name,
            payload={"validator": _tight_validator_for_field(field)},
            rationale="tighten local validator",
        )
    if action == "relax_validator":
        field = rng.choice(fields)
        return Mutation.from_parts(
            "relax_validator",
            field_name=field.name,
            payload={"validator": ""},
            rationale="relax validator to trade validity strictness for task score",
        )
    field = rng.choice(fields)
    return Mutation.from_parts(
        "change_required_optional",
        module_name=field.producer_module or producer,
        field_name=field.name,
        rationale="toggle required/optional status",
    )


def _weighted_choice(rng: random.Random, choices: list[str], weights: dict[str, float]) -> str:
    if not choices:
        raise ValueError("choices must be non-empty")
    raw_weights = [max(0.05, weights.get(choice, 1.0)) for choice in choices]
    total = sum(raw_weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for choice, weight in zip(choices, raw_weights):
        cumulative += weight
        if cumulative >= threshold:
            return choice
    return choices[-1]


def _update_operator_bandit(
    weights: dict[str, float],
    operator: str,
    *,
    reward: float,
    learning_rate: float = 0.25,
) -> None:
    current = weights.get(operator, 1.0)
    target = 1.0 + max(-0.8, min(2.0, reward))
    weights[operator] = (1.0 - learning_rate) * current + learning_rate * target


def _update_operator_stats(
    stats: dict[str, dict[str, float]],
    operator: str,
    *,
    reward: float,
) -> None:
    entry = stats.setdefault(operator, {"count": 0.0, "mean_reward": 0.0})
    count = entry["count"] + 1.0
    mean = entry["mean_reward"] + (reward - entry["mean_reward"]) / count
    entry["count"] = count
    entry["mean_reward"] = mean


def _operator_ucb_weights(
    stats: dict[str, dict[str, float]],
    *,
    exploration: float,
) -> dict[str, float]:
    if not stats:
        return {}
    total = sum(entry.get("count", 0.0) for entry in stats.values())
    weights: dict[str, float] = {}
    for operator, entry in stats.items():
        count = max(1.0, entry.get("count", 0.0))
        mean_reward = entry.get("mean_reward", 0.0)
        bonus = exploration * (max(0.0, total) + 1.0) ** 0.5 / count
        weights[operator] = max(0.05, 1.0 + mean_reward + bonus)
    return weights


def _initial_allocation_examples(
    examples: tuple[ProgramExample, ...],
    *,
    config: SchemaEvoConfig,
) -> tuple[ProgramExample, ...]:
    if config.allocation_strategy != "successive_halving" or len(examples) <= 1:
        return examples
    size = config.successive_halving_min_batch_size
    if size <= 0:
        size = max(1, len(examples) // config.successive_halving_eta)
    return examples[: min(len(examples), size)]


def _should_promote(
    *,
    optimizer_examples: tuple[ProgramExample, ...],
    stage_examples: tuple[ProgramExample, ...],
    config: SchemaEvoConfig,
) -> bool:
    return (
        config.allocation_strategy == "successive_halving"
        and len(stage_examples) < len(optimizer_examples)
    )


def _search_rollout_limit(
    *,
    total_rollouts: int,
    optimizer_examples: tuple[ProgramExample, ...],
    stage_examples: tuple[ProgramExample, ...],
    config: SchemaEvoConfig,
) -> int:
    if not _should_promote(
        optimizer_examples=optimizer_examples,
        stage_examples=stage_examples,
        config=config,
    ):
        return total_rollouts
    reserve = max(1, int(total_rollouts * config.successive_halving_promote_fraction))
    return max(1, total_rollouts - reserve)


def _select_promotions(
    records: list[CandidateRecord],
    *,
    remaining_rollouts: int,
    config: SchemaEvoConfig,
) -> tuple[CandidateRecord, ...]:
    if remaining_rollouts <= 0:
        return ()
    promote_count = max(1, int(len(records) * config.successive_halving_promote_fraction))
    promote_count = min(remaining_rollouts, promote_count, len(records))
    return tuple(
        sorted(
            records,
            key=lambda record: (
                schema_selection_value(record.result, use_field_bonus=True),
                record.result.mean_score,
                -record.result.invalid_output_rate,
                record.schema.schema_id,
            ),
            reverse=True,
        )[:promote_count]
    )


def _make_cost_meter(config: SchemaEvoConfig) -> CostMeter:
    return make_cost_meter(
        model_prices=config.model_prices,
        use_tiktoken=config.use_tiktoken_costing,
    )


def _initialize_population(
    *,
    task: str,
    module_names: tuple[str, ...],
    seed: int,
    initial_schemas: tuple[SchemaCandidate, ...],
    initial_random_schemas: int,
    schema_token_budget: int,
) -> tuple[SchemaCandidate, ...]:
    by_id: dict[str, SchemaCandidate] = {}
    for schema in initial_schemas:
        by_id.setdefault(schema.schema_id, schema)
    for schema in make_human_minimal_schemas(task=task, module_names=module_names, seed=seed):
        by_id.setdefault(schema.schema_id, schema)
    for schema in make_random_schema_controls(
        task=task,
        module_names=module_names,
        n=initial_random_schemas,
        seed=seed,
        schema_token_budget=schema_token_budget,
    ):
        by_id.setdefault(schema.schema_id, schema)
    return tuple(by_id.values())


def merge_schema_candidates(
    schemas: tuple[SchemaCandidate, ...],
    *,
    task: str,
    seed: int,
    schema_token_budget: int,
) -> SchemaCandidate:
    if len(schemas) < 2:
        raise ValueError("at least two schemas are required to merge")
    module_fields: dict[str, list[SchemaField]] = {}
    seen_fields: set[tuple[str, str]] = set()
    rules = []
    validators: dict[str, str] = {}
    parent_ids = []
    for schema in schemas:
        parent_ids.append(schema.schema_id)
        for module_name, fields in schema.module_fields.items():
            for field in fields:
                key = (module_name, field.name)
                if key in seen_fields:
                    continue
                seen_fields.add(key)
                module_fields.setdefault(module_name, []).append(field)
                validators.setdefault(field.name, schema.validators.get(field.name, field.validation_rule or ""))
        for rule in schema.consumption_rules:
            if all(
                not (
                    existing.consumer_module == rule.consumer_module
                    and existing.field_name == rule.field_name
                )
                for existing in rules
            ):
                rules.append(rule)
    merged = SchemaCandidate(
        schema_id="merged_schema",
        parent_schema_id="+".join(parent_ids),
        task=task,
        module_fields={module: tuple(fields) for module, fields in module_fields.items()},
        consumption_rules=tuple(rules),
        validators=validators,
        schema_token_budget=schema_token_budget,
        mutation_history=("merge_complementary_pareto_schemas",),
        proposer_seed=seed,
        control_type="schemaevo",
        metadata={"merged_parent_schema_ids": parent_ids},
    )
    merged = _prune_schema_to_budget(merged)
    return merged.with_id_from_content(prefix="merge")


def _prune_schema_to_budget(schema: SchemaCandidate) -> SchemaCandidate:
    if schema.token_cost <= schema.schema_token_budget:
        return schema
    module_fields = {module: list(fields) for module, fields in schema.module_fields.items()}
    while True:
        removable = [
            (module, index, field)
            for module, fields in module_fields.items()
            for index, field in enumerate(fields)
            if not field.required
        ]
        if not removable:
            break
        module, index, field = removable[-1]
        del module_fields[module][index]
        candidate = schema.replace(
            module_fields={name: tuple(fields) for name, fields in module_fields.items()},
            consumption_rules=tuple(
                rule for rule in schema.consumption_rules if rule.field_name != field.name
            ),
            validators={key: value for key, value in schema.validators.items() if key != field.name},
        )
        if candidate.token_cost <= candidate.schema_token_budget:
            return candidate
        schema = candidate
    return schema


def _compile_and_eval(
    *,
    base_program: LMProgram,
    schema: SchemaCandidate,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    config: SchemaEvoConfig,
    grammar: SchemaGrammar,
    stratify_key: StratifyKey | None,
    artifact_root: Path | None,
    rollout_cache: RolloutCache,
    step: int,
    baseline_cost: float,
    mutation: Mutation | None,
    cost_meter: CostMeter,
    budget: BudgetTracker,
    stage: str,
) -> CandidateRecord | dict[str, Any]:
    check = grammar.check_candidate(schema)
    if config.min_static_checks and not check.passed:
        return {
            "schema_id": schema.schema_id,
            "mutation": mutation.op.value if mutation else None,
            "reason": "static_check_failed",
            "errors": list(check.errors),
        }
    program = compile_schema_program(
        base_program=base_program,
        schema=schema,
        freeze_prompt_text=config.freeze_prompt_text,
        allow_only_schema_contract_insert=config.freeze_prompt_text and not config.allow_prompt_mutation,
    )
    assert_same_call_graph(program, base_program)
    batch = (
        examples
        if config.shared_eval_batch
        else sample_minibatch(
            examples=examples,
            batch_size=min(config.minibatch_size, len(examples)),
            seed=config.seed + step,
            stratify_key=stratify_key,
        )
    )
    min_calls = len(batch) * program.calls_per_example
    if not budget.can_start(min_target_task_calls=min_calls):
        return {
            "schema_id": schema.schema_id,
            "mutation": mutation.op.value if mutation else None,
            "reason": "budget_exhausted",
            "stage": stage,
            "budget": budget.summary(),
        }
    result = evaluate_program(
        program=program,
        examples=batch,
        scorer=scorer,
        method="schemaevo_closed_loop",
        candidate_id=f"schemaevo_step_{step}",
        seed=config.seed,
        baseline_program=base_program,
        strict_invalid_policy=config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        rollout_cache=rollout_cache,
        schema_generation_calls=1 if mutation else 0,
        run_id=f"schemaevo_step_{step}_{stage}_{schema.schema_id}",
    )
    result.baseline_dollar_cost_per_example = baseline_cost
    budget.record_result(result)
    return CandidateRecord(schema=schema, program=program, result=result, mutation=mutation, stage=stage)


def _select_parent(
    records: list[CandidateRecord],
    rng: random.Random,
    *,
    strategy: str,
    exploration: float,
) -> CandidateRecord:
    ranked = sorted(
        records,
        key=lambda record: (
            _parent_selection_score(record, strategy=strategy, exploration=exploration, rng=rng),
            record.schema.schema_id,
        ),
        reverse=True,
    )
    if strategy == "uniform_top_k":
        top = ranked[: max(1, min(5, len(ranked)))]
        return rng.choice(top)
    return ranked[0]


def _parent_selection_score(
    record: CandidateRecord,
    *,
    strategy: str,
    exploration: float,
    rng: random.Random,
) -> float:
    base = schema_selection_value(record.result, use_field_bonus=True)
    uncertainty = max(0.0, record.result.score_ci_high - record.result.score_ci_low)
    if strategy == "ucb":
        return base + exploration * uncertainty
    if strategy == "thompson":
        return rng.gauss(record.result.mean_score, max(uncertainty / 4.0, 1e-6))
    return base


def _missing_template_fields(
    parent: SchemaCandidate,
    *,
    task: str,
    module_names: tuple[str, ...],
) -> tuple[SchemaField, ...]:
    existing = set(parent.evolved_field_names)
    fields: list[SchemaField] = []
    for template in make_human_minimal_schemas(task=task, module_names=module_names, seed=parent.proposer_seed):
        fields.extend(field for field in template.all_fields if field.name not in existing)
    by_name: dict[str, SchemaField] = {}
    for field in fields:
        by_name.setdefault(field.name, field)
    return tuple(by_name.values())


def _tight_validator_for_field(field: SchemaField) -> str:
    if field.type == "string":
        return f"non_empty;max_tokens={field.max_tokens or 64}"
    if field.type in {"number", "integer"}:
        return "min=0;max=1" if "confidence" in field.name else "min=0"
    if field.type.startswith("array"):
        return f"max_items={field.max_items or 8}"
    if field.type == "enum" and field.enum_values:
        return "one_of=" + "|".join(field.enum_values)
    return field.validation_rule or ""
