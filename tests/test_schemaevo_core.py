from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

import pytest
import yaml

from schemaevo.adapters.dspy import DSpyModuleConfig, dspy_program_to_lm_program
from schemaevo.adapters.openai import OpenAIModuleConfig, openai_modules_to_lm_program
from schemaevo.benchmarks.openai_closed_loop import (
    _best as _closed_loop_best,
    _primary as _closed_loop_primary,
)
from schemaevo.benchmarks.readiness import check_benchmark_readiness, check_fixed_pool_split_readiness
from schemaevo.benchmarks.openai_fixed_pool import (
    OpenAIFixedPoolBenchmarkConfig,
    build_openai_benchmark_program,
    _scorer_for_dataset,
    run_openai_fixed_pool_benchmark,
)
from schemaevo.cli import _apply_budget_overrides, main as cli_main
from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.datasets.musique import load_musique_examples
from schemaevo.datasets.scorers import hotpotqa_exact_match, hover_label_accuracy, musique_exact_match
from schemaevo.eval.budgeting import estimate_evaluation_budget
from schemaevo.eval.cost_ledger import CostMeter, ModelPrice
from schemaevo.examples.toy_multihop import (
    build_toy_program,
    make_toy_examples,
    make_toy_traces,
    toy_scorer,
)
from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.field_ablations import FieldAblationResult
from schemaevo.eval.scoring import CandidateEvalResult, evaluate_program
from schemaevo.eval.stats import BootstrapDiff, PairedComparison
from schemaevo.experiments.budget_pareto import build_budget_pareto_report
from schemaevo.experiments.causal_pilot import build_causal_pilot_report
from schemaevo.experiments.deployment_invariance import build_fixed_pool_deployment_report
from schemaevo.experiments.external_prompt_optimizer import ExternalPromptOptimizer
from schemaevo.experiments.composability import run_prompt_optimizer_then_schemaevo
from schemaevo.experiments.transfer import run_openai_cross_model_schema_transfer
from schemaevo.optimizers.fixed_pool_schema import (
    ControlGuardrail,
    FixedPoolConfig,
    FixedPoolResult,
    MVPDecision,
    _make_control_guardrail,
    run_fixed_pool_schema_mvp,
)
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, merge_schema_candidates, schema_evo_optimize
from schemaevo.programs.base import ProgramExample
from schemaevo.programs.call_graph import assert_same_call_graph, extract_call_graph
from schemaevo.programs.compile_schema_program import CONTRACT_START, compile_schema_program
from schemaevo.schemas.candidate import ConsumptionRule, SchemaCandidate, SchemaField
from schemaevo.schemas.grammar import MutationOp, SchemaGrammar, assert_legal_mutation
from schemaevo.schemas.human_templates import (
    make_hotpotqa_schema_candidate,
    make_hover_schema_candidate,
    make_human_minimal_schemas,
    make_validator_only_schema,
)
from schemaevo.schemas.mutations import Mutation, apply_mutation
from schemaevo.schemas.proposer import (
    HeuristicTraceSchemaProposer,
    OpenAISchemaProposer,
    TraceExample,
    propose_schemas_from_traces,
)
from schemaevo.schemas.serialization import freeze_jsonl, load_jsonl
from schemaevo.schemas.validators import SchemaValidator


def test_schema_candidate_round_trip_and_static_checks(tmp_path):
    candidate = make_human_minimal_schemas(
        task="toy_multihop",
        module_names=("planner", "answerer"),
        seed=3,
    )[0]
    path = freeze_jsonl([candidate], tmp_path / "pool.jsonl")

    loaded = load_jsonl(path)

    assert loaded[0].schema_id == candidate.schema_id
    assert loaded[0].evolved_field_names == candidate.evolved_field_names
    grammar = SchemaGrammar(allowed_modules=("planner", "answerer"), max_schema_tokens=1024)
    grammar.check_candidate(loaded[0]).raise_if_failed()


def test_compile_schema_program_preserves_call_graph_and_freezes_prompt_prefix():
    base = build_toy_program()
    schema = make_hotpotqa_schema_candidate(module_names=base.module_names)

    compiled = compile_schema_program(
        base_program=base,
        schema=schema,
        freeze_prompt_text=True,
        allow_only_schema_contract_insert=True,
    )

    assert_same_call_graph(compiled, base)
    assert extract_call_graph(compiled) == extract_call_graph(base)
    assert compiled.modules[0].prompt.startswith(base.modules[0].prompt)
    assert CONTRACT_START in compiled.modules[0].prompt
    assert "bridge_entity" in compiled.modules[0].signature.output_fields


def test_schema_candidate_rejects_duplicate_field_names_across_modules():
    planner_field = SchemaField(
        name="shared_signal",
        type="string",
        description="Planner signal.",
        required=False,
        producer_module="planner",
        consumer_modules=("answerer",),
    )
    answerer_field = SchemaField(
        name="shared_signal",
        type="string",
        description="Answerer signal.",
        required=False,
        producer_module="answerer",
        consumer_modules=("planner",),
    )

    with pytest.raises(ValueError, match="duplicate schema field name"):
        SchemaCandidate(
            schema_id="duplicate_global_field",
            parent_schema_id=None,
            task="test",
            module_fields={"planner": (planner_field,), "answerer": (answerer_field,)},
            consumption_rules=(),
            validators={},
            schema_token_budget=128,
            mutation_history=(),
            proposer_seed=0,
        )


def test_schema_candidate_rejects_unknown_validator_fields():
    field = SchemaField(
        name="known_signal",
        type="string",
        description="Known signal.",
        required=False,
        producer_module="planner",
        consumer_modules=("answerer",),
    )

    with pytest.raises(ValueError, match="validators reference unknown fields"):
        SchemaCandidate(
            schema_id="stale_validator",
            parent_schema_id=None,
            task="test",
            module_fields={"planner": (field,)},
            consumption_rules=(),
            validators={"stale_signal": "non_empty"},
            schema_token_budget=128,
            mutation_history=(),
            proposer_seed=0,
        )


def test_call_graph_rejects_changed_demo_ids():
    base = build_toy_program()
    candidate = base.clone()
    base.modules[0].metadata["demo_ids"] = ("demo_a",)
    candidate.modules[0].metadata["demo_ids"] = ("demo_b",)

    with pytest.raises(AssertionError, match="call graph changed"):
        assert_same_call_graph(candidate, base)


def test_runtime_schema_fields_exclude_original_outputs():
    base = build_toy_program()
    schema = make_human_minimal_schemas(
        task="toy_multihop",
        module_names=base.module_names,
        seed=0,
    )[0]
    compiled = compile_schema_program(
        base_program=base,
        schema=schema,
        freeze_prompt_text=True,
        allow_only_schema_contract_insert=True,
    )
    original_answerer_runner = compiled.modules[1].runner

    def checking_answerer(state, module, example, context):
        assert "plan_summary" not in state["schema_fields"]
        assert "bridge_entity" in state["schema_fields"]
        return original_answerer_runner(state, module, example, context)

    compiled.modules[1].runner = checking_answerer

    prediction = compiled.run(
        make_toy_examples("validation_runtime", 1)[0],
        run_id="runtime_check",
        method="schema_runtime",
        candidate_id="schema_runtime_0",
        seed=0,
    )

    assert "plan_summary" in prediction.module_outputs["planner"]
    assert "bridge_entity" in prediction.module_outputs["planner"]


def test_cost_meter_prices_use_recorded_token_counts():
    meter = CostMeter(
        prices={
            "toy-model": ModelPrice(
                input_per_million=10.0,
                output_per_million=20.0,
                source_date="test",
            )
        },
        token_counter=lambda model, text: max(1, len(text.split())),
    )

    result = evaluate_program(
        program=build_toy_program(),
        examples=make_toy_examples("cost_validation", 2),
        scorer=toy_scorer,
        method="cost_test",
        candidate_id="cost_test",
        seed=1,
        cost_meter=meter,
    )

    expected = (result.prompt_tokens * 10.0 + result.completion_tokens * 20.0) / 1_000_000
    assert result.dollar_cost == pytest.approx(expected)
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


def test_score_artifacts_include_predictions_and_per_example_accounting(tmp_path):
    result = evaluate_program(
        program=build_toy_program(),
        examples=make_toy_examples("artifact_audit", 1),
        scorer=toy_scorer,
        method="artifact_audit",
        candidate_id="artifact_audit",
        seed=1,
        artifact_dir=tmp_path,
    )

    rows = [
        json.loads(line)
        for line in Path(result.per_example_scores_path).read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["example_id"] == "artifact_audit_0"
    assert rows[0]["final_output"] == result.predictions[0].final_output
    assert rows[0]["target_task_calls"] == 2
    assert rows[0]["retriever_calls"] == 1
    assert rows[0]["prompt_tokens"] > 0
    assert rows[0]["completion_tokens"] > 0
    assert rows[0]["dollar_cost"] >= 0.0
    assert "latency_ms" in rows[0]
    assert "validation_errors" in rows[0]


def test_evaluation_budget_estimate_uses_token_counter_and_max_outputs():
    meter = CostMeter(
        prices={
            "toy-model": ModelPrice(
                input_per_million=1.0,
                output_per_million=2.0,
                source_date="test",
            )
        },
        token_counter=lambda model, text: max(1, len(text.split())),
    )

    estimate = estimate_evaluation_budget(
        program=build_toy_program(),
        examples=make_toy_examples("budget_estimate", 2),
        cost_meter=meter,
    )

    assert estimate.target_task_calls == 4
    assert estimate.prompt_tokens > 0
    assert estimate.completion_tokens == 2 * (256 + 128)
    expected_cost = (estimate.prompt_tokens * 1.0 + estimate.completion_tokens * 2.0) / 1_000_000
    assert estimate.dollar_cost == pytest.approx(expected_cost)


def test_validator_uses_deterministic_coercion_without_llm_repair():
    field = SchemaField(
        name="bridge_entity",
        type="string",
        description="Bridge entity.",
        required=True,
        producer_module="planner",
        consumer_modules=("answerer",),
    )
    schema = SchemaCandidate(
        schema_id="schema_test",
        parent_schema_id=None,
        task="HotpotQA",
        module_fields={"planner": (field,)},
        consumption_rules=(
            ConsumptionRule(
                consumer_module="answerer",
                field_name="bridge_entity",
                instruction="Use bridge entity.",
                required_behavior="Use it.",
                fallback_if_missing="Fallback.",
            ),
        ),
        validators={"bridge_entity": "must be a string"},
        schema_token_budget=128,
        mutation_history=(),
        proposer_seed=0,
    )
    validator = SchemaValidator(schema)

    result = validator.validate_module_output(
        "planner",
        {"bridge_entity": 123},
        policy="deterministic_coercion",
    )

    assert result.valid
    assert result.deterministic_repair_applied
    assert result.parsed["bridge_entity"] == "123"


def test_executable_validator_constraints_reject_bad_values():
    field = SchemaField(
        name="confidence",
        type="number",
        description="Confidence.",
        required=True,
        producer_module="planner",
        consumer_modules=("answerer",),
        validation_rule="min=0;max=1",
    )
    schema = SchemaCandidate(
        schema_id="schema_constraints",
        parent_schema_id=None,
        task="HotpotQA",
        module_fields={"planner": (field,)},
        consumption_rules=(
            ConsumptionRule(
                consumer_module="answerer",
                field_name="confidence",
                instruction="Use confidence.",
                required_behavior="Use it.",
                fallback_if_missing="Fallback.",
            ),
        ),
        validators={"confidence": "min=0;max=1"},
        schema_token_budget=128,
        mutation_history=(),
        proposer_seed=0,
    )

    result = SchemaValidator(schema).validate_module_output(
        "planner",
        {"confidence": 2.0},
        policy="deterministic_coercion",
    )

    assert not result.valid
    assert any("max" in error for error in result.errors)


def test_validator_rejects_malformed_known_object_fields():
    schema = make_hover_schema_candidate(module_names=("planner", "verifier"))

    result = SchemaValidator(schema).validate_module_output(
        "planner",
        {
            "claim_atoms": [
                {
                    "text": "A subclaim.",
                    "entities": "not-a-list",
                    "relation": "states",
                }
            ],
            "hop_plan": [],
            "evidence_table": [],
            "evidence_conflict": {
                "has_conflict": "false",
                "conflict_description": "",
            },
            "final_verdict_preconditions": [],
        },
        policy="deterministic_coercion",
    )

    assert not result.valid
    assert any("claim_atoms[0].entities" in error for error in result.errors)
    assert any("claim_atoms[0].needs_evidence_from" in error for error in result.errors)
    assert any("evidence_conflict.has_conflict" in error for error in result.errors)


def test_trace_proposer_rejects_non_train_traces():
    with pytest.raises(ValueError, match="train traces only"):
        propose_schemas_from_traces(
            traces=(
                TraceExample(
                    example_id="bad",
                    split="validation",
                    module_name="planner",
                    input_summary="bridge",
                    output_summary="query",
                ),
            ),
            task="toy_multihop",
            module_names=("planner", "answerer"),
            n=1,
            seed=0,
        )


def test_openai_schema_proposer_parses_structured_response():
    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-4.1-mini"

            class Response:
                output_text = json.dumps(
                    {
                        "schemas": [
                            {
                                "rationale": "Bridge entity is missing downstream.",
                                "fields": [
                                    {
                                        "name": "bridge_entity",
                                        "type": "string",
                                        "description": "Bridge entity.",
                                        "required": True,
                                        "producer_module": "planner",
                                        "consumer_modules": ["answerer"],
                                        "enum_values": None,
                                        "max_items": None,
                                        "max_tokens": 16,
                                        "validator": "non_empty;max_tokens=16",
                                        "causal_hypothesis": "Carries the missing hop.",
                                    }
                                ],
                            }
                        ]
                    }
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    proposer = OpenAISchemaProposer(
        client=FakeClient(),
        cost_meter=CostMeter(
            prices={
                "gpt-4.1-mini": ModelPrice(
                    input_per_million=1.0,
                    output_per_million=2.0,
                    source_date="test",
                )
            },
            token_counter=lambda model, text: max(1, len(text.split())),
        ),
    )
    candidates = proposer.propose(
        traces=make_toy_traces(),
        task="toy_multihop",
        module_names=("planner", "answerer"),
        n=1,
        seed=5,
        schema_token_budget=256,
    )

    assert candidates[0].metadata["proposer_model"] == "gpt-4.1-mini"
    assert candidates[0].evolved_field_names == ("bridge_entity",)
    assert proposer.last_usage["optimizer_proposal_calls"] == 1
    assert proposer.last_usage["prompt_tokens"] > 0
    assert proposer.last_usage["completion_tokens"] > 0
    assert proposer.last_usage["dollar_cost"] > 0
    assert proposer.total_usage == proposer.last_usage


def test_openai_schema_proposer_drops_malformed_fields_without_crashing():
    class FakeResponses:
        def create(self, **kwargs):
            class Response:
                output_text = json.dumps(
                    {
                        "schemas": [
                            {
                                "rationale": "Bad field name.",
                                "fields": [
                                    {
                                        "name": "Bad Field",
                                        "type": "string",
                                        "description": "Invalid name.",
                                        "required": True,
                                        "producer_module": "planner",
                                        "consumer_modules": ["answerer"],
                                        "enum_values": None,
                                        "max_items": None,
                                        "max_tokens": 16,
                                        "validator": "non_empty",
                                        "causal_hypothesis": "Invalid field should be skipped.",
                                    }
                                ],
                            },
                            {
                                "rationale": "Valid reflected field.",
                                "fields": [
                                    {
                                        "name": "bridge_entity",
                                        "type": "string",
                                        "description": "Bridge entity.",
                                        "required": True,
                                        "producer_module": "planner",
                                        "consumer_modules": ["answerer"],
                                        "enum_values": None,
                                        "max_items": None,
                                        "max_tokens": 16,
                                        "validator": "non_empty",
                                        "causal_hypothesis": "Carries the missing hop.",
                                    }
                                ],
                            },
                        ]
                    }
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    proposer = OpenAISchemaProposer(client=FakeClient())
    candidates = proposer.propose(
        traces=make_toy_traces(),
        task="toy_multihop",
        module_names=("planner", "answerer"),
        n=2,
        seed=5,
        schema_token_budget=256,
    )

    assert len(candidates) == 1
    assert candidates[0].evolved_field_names == ("bridge_entity",)
    assert proposer.last_errors


def test_openai_schema_proposer_returns_empty_on_bad_response():
    class FakeResponses:
        def create(self, **kwargs):
            class Response:
                output_text = "not json"

            return Response()

    class FakeClient:
        responses = FakeResponses()

    proposer = OpenAISchemaProposer(client=FakeClient())

    assert proposer.propose(
        traces=make_toy_traces(),
        task="toy_multihop",
        module_names=("planner", "answerer"),
        n=1,
        seed=0,
        schema_token_budget=128,
    ) == []
    assert proposer.last_errors


def test_mutations_apply_legal_ops_and_reject_forbidden_ops():
    candidate = make_hotpotqa_schema_candidate(module_names=("planner", "answerer"), seed=0)
    mutated = apply_mutation(
        candidate,
        Mutation.from_parts(
            MutationOp.RENAME_FIELD,
            module_name="planner",
            field_name="bridge_entity",
            payload={"new_name": "bridge_entity_candidate"},
        ),
    )

    assert "bridge_entity_candidate" in mutated.evolved_field_names
    assert "bridge_entity" not in mutated.evolved_field_names
    with pytest.raises(ValueError, match="forbidden mutation"):
        assert_legal_mutation("ADD_LLM_CALL")


def test_fixed_pool_schema_mvp_runs_end_to_end(tmp_path):
    config = FixedPoolConfig(
        task="toy_multihop",
        target_model="toy-model",
        seed=11,
        n_trace_schemas=8,
        n_random_schemas=2,
        top_k_confirmation=2,
        min_confirmation_delta=0.1,
        bootstrap_resamples=100,
        randomization_swaps=100,
    )

    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=make_toy_examples("validation_smoke", 3),
        selection_examples=make_toy_examples("validation_selection", 8),
        confirmation_examples=make_toy_examples("validation_confirmation", 10),
        heldout_test_examples=make_toy_examples("final_test", 4),
        scorer=toy_scorer,
        config=config,
        artifact_dir=tmp_path,
    )

    assert result.decision.proceed
    assert result.best_confirmation_result.mean_score == 1.0
    assert result.baseline_confirmation_result.mean_score == 0.0
    assert result.decision.field_masking_max_drop >= 1.0
    best_schema = next(
        schema for schema in result.schema_pool if schema.schema_id == result.best_confirmation_result.schema_id
    )
    assert {"bridge_entity", "next_query_intent"}.issubset(set(best_schema.evolved_field_names))
    ablations = {item.ablation for item in result.field_ablation_results}
    assert {"mask", "blank", "shuffle", "downstream_disabled"}.issubset(ablations)
    assert result.primary_confirmation_result.schema_id == result.top_selection_results[0].schema_id
    assert result.heldout_test_result is not None
    assert result.paired_stats.adjusted_p is not None
    with open(tmp_path / "results" / "mvp_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["decision"]["proceed"] is True
    assert "control_guardrail" in summary


def test_fixed_pool_rejects_split_leakage(tmp_path):
    config = FixedPoolConfig(
        task="toy_multihop",
        target_model="toy-model",
        n_trace_schemas=1,
        n_random_schemas=0,
        bootstrap_resamples=10,
        randomization_swaps=10,
    )
    leaked = make_toy_examples("validation_selection", 2)

    with pytest.raises(ValueError, match="overlap"):
        run_fixed_pool_schema_mvp(
            base_program=build_toy_program(),
            train_traces=make_toy_traces(),
            smoke_examples=(),
            selection_examples=leaked,
            confirmation_examples=leaked,
            scorer=toy_scorer,
            config=config,
            artifact_dir=tmp_path,
        )


def test_fixed_pool_budget_reserves_confirmation_and_trims_candidates(tmp_path):
    config = FixedPoolConfig(
        task="toy_multihop",
        target_model="toy-model",
        seed=41,
        n_trace_schemas=8,
        n_random_schemas=0,
        top_k_confirmation=1,
        bootstrap_resamples=10,
        randomization_swaps=10,
        max_target_task_calls=10,
    )

    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_toy_examples("validation_selection", 1),
        confirmation_examples=make_toy_examples("validation_confirmation", 1),
        scorer=toy_scorer,
        config=config,
        artifact_dir=tmp_path,
    )

    assert result.budget_summary["target_task_calls"] == 10
    assert result.budget_summary["exhausted"] is True
    assert len(result.selection_results) == 2
    assert len(result.confirmation_results) == 1
    assert result.primary_confirmation_result.schema_id == result.top_selection_results[0].schema_id
    assert result.field_ablation_results == ()


def test_fixed_pool_control_guardrail_flags_controls_in_top_k():
    semantic = make_hotpotqa_schema_candidate(module_names=("planner", "answerer"), seed=0)
    control_field = SchemaField(
        name="opaque_key",
        type="string",
        description="Opaque control field.",
        required=False,
        producer_module="planner",
        consumer_modules=("answerer",),
    )
    control = SchemaCandidate(
        schema_id="control_schema",
        parent_schema_id=None,
        task="HotpotQA",
        module_fields={"planner": (control_field,)},
        consumption_rules=(
            ConsumptionRule(
                consumer_module="answerer",
                field_name="opaque_key",
                instruction="Use only if relevant.",
                required_behavior="Do not infer new evidence.",
                fallback_if_missing="Use original behavior.",
            ),
        ),
        validators={},
        schema_token_budget=128,
        mutation_history=("random_schema_control",),
        proposer_seed=0,
        control_type="random",
    )
    primary = _candidate_eval_result(schema_id=semantic.schema_id, mean_score=0.6)
    control_result = _candidate_eval_result(schema_id=control.schema_id, mean_score=0.7)

    guardrail = _make_control_guardrail(
        schema_pool=(semantic, control),
        top_selection_results=(primary, control_result),
        confirmation_results=(primary, control_result),
        primary_confirmation=primary,
    )

    assert guardrail.control_in_top_k_warning
    assert guardrail.selection_top_k_control_schema_ids == (control.schema_id,)
    assert guardrail.confirmation_top_k_control_schema_ids == (control.schema_id,)
    assert guardrail.best_control_schema_id == control.schema_id
    assert guardrail.best_control_vs_primary_delta == pytest.approx(0.1)
    assert not guardrail.primary_is_control
    assert guardrail.primary_control_type is None


def test_fixed_pool_control_guardrail_flags_control_primary():
    control = make_validator_only_schema(
        task="HotpotQA",
        module_names=("planner", "answerer"),
        seed=0,
    )
    primary = _candidate_eval_result(schema_id=control.schema_id, mean_score=0.6)

    guardrail = _make_control_guardrail(
        schema_pool=(control,),
        top_selection_results=(primary,),
        confirmation_results=(primary,),
        primary_confirmation=primary,
    )

    assert guardrail.control_in_top_k_warning
    assert guardrail.primary_is_control
    assert guardrail.primary_control_type == "validator_only"
    assert guardrail.best_control_schema_id == control.schema_id
    assert guardrail.best_control_vs_primary_delta == pytest.approx(0.0)
    assert "Primary selected schema is a control" in str(guardrail.warning)


def test_fixed_pool_records_schema_proposal_usage(tmp_path):
    class AccountingProposer:
        def __init__(self):
            self.total_usage = {
                "optimizer_proposal_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "dollar_cost": 0.0,
            }

        def propose(self, **kwargs):
            self.total_usage = {
                "optimizer_proposal_calls": 1,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "dollar_cost": 0.25,
            }
            return HeuristicTraceSchemaProposer().propose(**kwargs)

    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_toy_examples("validation_selection", 1),
        confirmation_examples=make_toy_examples("validation_confirmation", 1),
        scorer=toy_scorer,
        config=FixedPoolConfig(
            task="toy_multihop",
            target_model="toy-model",
            seed=43,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
        ),
        proposer=AccountingProposer(),
        artifact_dir=tmp_path,
    )

    assert result.proposal_usage["optimizer_proposal_calls"] == 1
    assert result.cost_summary["optimizer_proposal_calls"] == 1
    assert result.cost_summary["prompt_tokens"] >= 11
    assert result.cost_summary["completion_tokens"] >= 7
    assert result.cost_summary["dollar_cost"] >= 0.25
    assert "wall_clock_seconds" in result.cost_summary
    assert "max_p50_latency_ms" in result.cost_summary
    assert "max_p95_latency_ms" in result.cost_summary
    assert result.budget_summary["prompt_tokens"] >= result.cost_summary["prompt_tokens"]


def test_fixed_pool_reflection_uses_train_traces_without_validation_leakage(tmp_path):
    class TwoRoundProposer:
        def __init__(self):
            self.calls = []
            self.total_usage = {
                "optimizer_proposal_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "dollar_cost": 0.0,
            }

        def propose(self, **kwargs):
            self.calls.append(kwargs)
            self.total_usage = {
                "optimizer_proposal_calls": len(self.calls),
                "prompt_tokens": 10 * len(self.calls),
                "completion_tokens": 5 * len(self.calls),
                "total_tokens": 15 * len(self.calls),
                "dollar_cost": 0.01 * len(self.calls),
            }
            module_names = kwargs["module_names"]
            field_name = "bridge_entity" if len(self.calls) == 1 else "next_query_intent"
            field = SchemaField(
                name=field_name,
                type="string",
                description=f"{field_name} from train-only reflection.",
                required=True,
                producer_module=module_names[0],
                consumer_modules=(module_names[-1],),
                max_tokens=32,
                validation_rule="non_empty;max_tokens=32",
            )
            candidate = SchemaCandidate(
                schema_id=f"two_round_{len(self.calls)}",
                parent_schema_id=None,
                task=kwargs["task"],
                module_fields={module_names[0]: (field,)},
                consumption_rules=(
                    ConsumptionRule(
                        consumer_module=module_names[-1],
                        field_name=field.name,
                        instruction=f"Use `{field.name}` from the planner.",
                        required_behavior="Consume without adding calls.",
                        fallback_if_missing="Use original prompt behavior.",
                    ),
                ),
                validators={field.name: field.validation_rule or ""},
                schema_token_budget=kwargs["schema_token_budget"],
                mutation_history=(f"two_round_call={len(self.calls)}",),
                proposer_seed=kwargs["seed"],
            )
            return [candidate.with_id_from_content(prefix=f"two_round_{len(self.calls)}")]

    proposer = TwoRoundProposer()
    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_toy_examples("validation_selection", 4),
        confirmation_examples=make_toy_examples("validation_confirmation", 4),
        scorer=toy_scorer,
        config=FixedPoolConfig(
            task="toy_multihop",
            target_model="toy-model",
            seed=47,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            reflection_rounds=2,
            reflection_schemas_per_round=1,
        ),
        proposer=proposer,
        artifact_dir=tmp_path,
    )

    assert len(proposer.calls) == 2
    assert all(trace.split == "train" for trace in proposer.calls[1]["traces"])
    assert all(
        trace.metadata["source"] == "schemaevo_reflection_train_trace"
        for trace in proposer.calls[1]["traces"]
    )
    assert all("primary_schema_id" in trace.metadata for trace in proposer.calls[1]["traces"])
    assert result.reflection_rounds[0]["status"] == "evaluated"
    assert result.cost_summary["optimizer_proposal_calls"] == 2
    assert result.cost_summary["optimizer_reflection_calls"] == 1
    assert result.proposal_usage["optimizer_reflection_calls"] == 1
    assert any(
        "next_query_intent" in schema.evolved_field_names
        for schema in result.schema_pool
        if schema.mutation_history == ("two_round_call=2",)
    )


def test_fixed_pool_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="top_k_confirmation"):
        FixedPoolConfig(top_k_confirmation=0)
    with pytest.raises(ValueError, match="workers"):
        FixedPoolConfig(workers=0)
    with pytest.raises(ValueError, match="progress"):
        FixedPoolConfig(progress="loud")  # type: ignore[arg-type]


def test_fixed_pool_multiprocessing_candidate_eval_runs(tmp_path):
    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_toy_examples("validation_selection", 2),
        confirmation_examples=make_toy_examples("validation_confirmation", 2),
        scorer=toy_scorer,
        config=FixedPoolConfig(
            task="toy_multihop",
            target_model="toy-model",
            seed=53,
            n_trace_schemas=2,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            workers=2,
            progress="none",
        ),
        artifact_dir=tmp_path,
    )

    assert result.primary_confirmation_result.mean_score == 1.0
    assert len(result.selection_results) >= 2
    assert (tmp_path / "results" / "mvp_summary.json").exists()


def test_fixed_pool_parallel_openai_jobs_fall_back_to_threads_for_unpicklable_client(tmp_path):
    calls = []

    class FakeResponses:
        def __init__(self) -> None:
            self.lock = threading.RLock()

        def create(self, **kwargs):
            with self.lock:
                calls.append(kwargs["text"]["format"]["name"])
            schema = kwargs["text"]["format"]["schema"]
            payload = {
                field_name: _fake_openai_value(field_name, schema["properties"][field_name])
                for field_name in schema["required"]
            }

            class Response:
                output_text = json.dumps(payload)

            return Response()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    program = openai_modules_to_lm_program(
        task="HotpotQA",
        modules=(
            OpenAIModuleConfig(
                name="planner",
                input_fields=("question", "context"),
                output_fields=("plan",),
                prompt="Plan the answer.",
                model="gpt-4.1-mini",
                max_output_tokens=64,
            ),
            OpenAIModuleConfig(
                name="answerer",
                input_fields=("question", "context", "plan"),
                output_fields=("answer",),
                prompt="Answer.",
                model="gpt-4.1-mini",
                max_output_tokens=64,
            ),
        ),
        final_output_module="answerer",
        client=FakeClient(),
    )

    def make_openai_examples(split: str) -> tuple[ProgramExample, ...]:
        return tuple(
            ProgramExample(
                example_id=f"openai_parallel_{split}_{index}",
                split=split,
                inputs={"question": "What is the answer?", "context": "The answer is ok."},
                expected={"answer": "ok"},
            )
            for index in range(2)
        )

    def score(example, prediction):
        return float(prediction.final_output.get("answer") == example.expected["answer"])

    result = run_fixed_pool_schema_mvp(
        base_program=program,
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_openai_examples("selection"),
        confirmation_examples=make_openai_examples("confirmation"),
        scorer=score,
        config=FixedPoolConfig(
            task="HotpotQA",
            target_model="gpt-4.1-mini",
            seed=61,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            workers=2,
            progress="none",
        ),
        artifact_dir=tmp_path,
    )

    assert result.primary_confirmation_result.mean_score == 1.0
    assert len(calls) > 0


def test_cross_model_transfer_runs_with_fake_openai_client(tmp_path):
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            user_payload = json.loads(kwargs["input"][1]["content"])
            calls.append((kwargs["model"], user_payload["module_name"]))
            schema = kwargs["text"]["format"]["schema"]
            payload = {}
            for field_name in schema["required"]:
                if field_name == "answer":
                    payload[field_name] = "ok" if user_payload.get("schema_fields") else "wrong"
                else:
                    payload[field_name] = _fake_openai_value(field_name, schema["properties"][field_name])

            class Response:
                output_text = json.dumps(payload)

            return Response()

    class FakeClient:
        responses = FakeResponses()

    def write_hotpot_split(path: Path, split: str) -> None:
        path.write_text(
            json.dumps(
                [
                    {
                        "_id": f"{split}_{index}",
                        "question": "What is the answer?",
                        "answer": "ok",
                        "context": [["Title", ["The answer is ok."]]],
                    }
                    for index in range(2)
                ]
            ),
            encoding="utf-8",
        )

    train_path = tmp_path / "train.json"
    selection_path = tmp_path / "selection.json"
    confirmation_path = tmp_path / "confirmation.json"
    heldout_path = tmp_path / "heldout.json"
    write_hotpot_split(train_path, "train")
    write_hotpot_split(selection_path, "selection")
    write_hotpot_split(confirmation_path, "confirmation")
    write_hotpot_split(heldout_path, "heldout")

    report = run_openai_cross_model_schema_transfer(
        benchmark_config=OpenAIFixedPoolBenchmarkConfig(
            dataset="hotpotqa",
            train_path=train_path,
            selection_path=selection_path,
            confirmation_path=confirmation_path,
            heldout_path=heldout_path,
            train_limit=2,
            selection_limit=2,
            confirmation_limit=2,
            heldout_limit=2,
            model="source-model",
        ),
        source_model="source-model",
        target_model="target-model",
        fixed_pool_config=FixedPoolConfig(
            task="HotpotQA",
            target_model="source-model",
            seed=71,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            reflection_rounds=1,
            workers=1,
            progress="none",
        ),
        artifact_dir=tmp_path / "transfer",
        proposer=HeuristicTraceSchemaProposer(),
        client=FakeClient(),
    )

    assert report.transferred_schema_id
    assert report.source_delta == 1.0
    assert report.target_delta == 1.0
    assert report.schema_transfer_retention == 1.0
    assert (tmp_path / "transfer" / "cross_model_transfer_report.json").exists()
    assert ("source-model", "answerer") in calls
    assert ("target-model", "answerer") in calls


def _fake_openai_value(field_name: str, schema: dict[str, object]) -> object:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    schema_type = schema.get("type", "string")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "string")
    if field_name == "answer":
        return "ok"
    if schema_type == "array":
        return []
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(properties, dict) and isinstance(required, list):
            return {
                str(name): _fake_openai_value(str(name), properties[str(name)])
                for name in required
            }
        return {"value": ""}
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    return "ok" if field_name == "plan" else ""


def test_closed_loop_config_rejects_negative_budget_caps():
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        SchemaEvoConfig(max_prompt_tokens=-1)
    with pytest.raises(ValueError, match="max_dollar_cost"):
        SchemaEvoConfig(max_dollar_cost=-0.01)
    with pytest.raises(ValueError, match="progress"):
        SchemaEvoConfig(progress="loud")  # type: ignore[arg-type]


def test_mvp_gpt41mini_configs_parse_as_fixed_pool_configs():
    for config_path in (
        Path("configs/mvp_hotpotqa_gpt41mini.yaml"),
        Path("configs/mvp_hover_gpt41mini.yaml"),
    ):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = FixedPoolConfig(**raw["fixed_pool"])

        assert config.target_model == "gpt-4.1-mini"
        assert config.freeze_prompt_text
        assert config.allow_only_schema_contract_insert
        assert config.reflection_rounds == 2
        assert config.workers == 1
        assert config.progress == "auto"
        assert raw["proposer"]["kind"] == "openai"
        assert raw["train_examples"] == 150
        assert raw["confirmation_examples"] == 500


def test_cli_budget_overrides_update_fixed_pool_config():
    config = _apply_budget_overrides(
        FixedPoolConfig(),
        max_target_task_calls=10,
        max_prompt_tokens=None,
        max_completion_tokens=30,
        max_total_tokens=None,
        max_dollar_cost=0.25,
    )

    assert config.max_target_task_calls == 10
    assert config.max_prompt_tokens is None
    assert config.max_completion_tokens == 30
    assert config.max_dollar_cost == 0.25


def test_closed_loop_schemaevo_optimizer_runs_with_fixed_call_graph(tmp_path):
    config = SchemaEvoConfig(
        task="toy_multihop",
        seed=17,
        max_program_rollouts=5,
        minibatch_size=5,
        initial_random_schemas=1,
        k_final=2,
    )

    result = schema_evo_optimize(
        base_program=build_toy_program(),
        examples=make_toy_examples("optimizer_validation", 12),
        scorer=toy_scorer,
        config=config,
        artifact_dir=tmp_path,
    )

    assert result.evaluated_records
    assert result.final_records
    baseline_ids = tuple(pred.example_id for pred in result.baseline_result.predictions)
    for record in result.evaluated_records:
        assert_same_call_graph(record.program, build_toy_program())
        assert tuple(pred.example_id for pred in record.result.predictions) == baseline_ids
    assert (tmp_path / "results" / "schemaevo_summary.json").exists()


def test_closed_loop_successive_halving_promotes_to_larger_shared_batch(tmp_path):
    config = SchemaEvoConfig(
        task="toy_multihop",
        seed=19,
        max_program_rollouts=7,
        max_mutation_attempts=20,
        minibatch_size=8,
        initial_random_schemas=1,
        k_final=2,
        allocation_strategy="successive_halving",
        successive_halving_min_batch_size=2,
        successive_halving_promote_fraction=0.5,
    )

    result = schema_evo_optimize(
        base_program=build_toy_program(),
        examples=make_toy_examples("optimizer_validation", 12),
        scorer=toy_scorer,
        config=config,
        artifact_dir=tmp_path,
    )

    assert result.promotion_baseline_result is not None
    assert any(record.stage == "promotion" for record in result.evaluated_records)
    assert all(record.stage == "promotion" for record in result.final_records)
    assert {len(record.result.predictions) for record in result.final_records} == {8}


def test_closed_loop_budget_stops_candidate_evaluations(tmp_path):
    config = SchemaEvoConfig(
        task="toy_multihop",
        seed=23,
        max_program_rollouts=20,
        max_mutation_attempts=20,
        minibatch_size=2,
        initial_random_schemas=5,
        k_final=2,
        max_target_task_calls=8,
    )

    result = schema_evo_optimize(
        base_program=build_toy_program(),
        examples=make_toy_examples("optimizer_validation", 12),
        scorer=toy_scorer,
        config=config,
        artifact_dir=tmp_path,
    )

    assert result.budget_summary["target_task_calls"] <= 8
    assert len(result.evaluated_records) <= 1


def test_closed_loop_primary_result_is_optimizer_locked_not_confirmation_max():
    first = _candidate_eval_result(schema_id="schema_first", mean_score=0.2)
    second = _candidate_eval_result(schema_id="schema_second", mean_score=0.9)

    assert _closed_loop_primary((first, second)) is first
    assert _closed_loop_best((first, second)) is second


def test_closed_loop_token_budget_prevents_starting_over_budget_candidates():
    config = SchemaEvoConfig(
        task="toy_multihop",
        seed=31,
        max_program_rollouts=5,
        max_mutation_attempts=5,
        minibatch_size=2,
        initial_random_schemas=2,
        k_final=1,
        max_completion_tokens=500,
    )

    result = schema_evo_optimize(
        base_program=build_toy_program(),
        examples=make_toy_examples("optimizer_validation", 6),
        scorer=toy_scorer,
        config=config,
    )

    assert result.budget_summary["completion_tokens"] <= 500
    assert not result.evaluated_records
    assert any(item["reason"] == "budget_exhausted" for item in result.rejected_schemas)


def test_schema_merge_combines_complementary_fields():
    schemas = tuple(
        make_human_minimal_schemas(
            task="toy_multihop",
            module_names=("planner", "answerer"),
            seed=0,
        )
    )

    merged = merge_schema_candidates(
        schemas,
        task="toy_multihop",
        seed=0,
        schema_token_budget=512,
    )

    merged_fields = set(merged.evolved_field_names)
    assert set(schemas[0].evolved_field_names).issubset(merged_fields)
    assert merged.token_cost <= merged.schema_token_budget
    assert schemas[0].schema_id in merged.parent_schema_id


def _candidate_eval_result(*, schema_id: str, mean_score: float) -> CandidateEvalResult:
    return CandidateEvalResult(
        run_id=f"run_{schema_id}",
        method="test",
        candidate_id=schema_id,
        schema_id=schema_id,
        task="test",
        split="test",
        n_examples=1,
        mean_score=mean_score,
        standard_error=0.0,
        score_ci_low=mean_score,
        score_ci_high=mean_score,
        per_example_scores_path="",
        target_task_calls=0,
        optimizer_proposal_calls=0,
        optimizer_reflection_calls=0,
        schema_generation_calls=0,
        schema_validation_repair_calls=0,
        retriever_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        dollar_cost=0.0,
        wall_clock_seconds=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
        invalid_output_rate=0.0,
        schema_validation_failure_count=0,
        invalid_score_policy="zero",
        per_example_scores=(mean_score,),
    )


def _fixed_pool_result_for_causal_pilot_null_signal() -> FixedPoolResult:
    primary_schema = make_hotpotqa_schema_candidate(module_names=("planner", "answerer"), seed=0)
    control_field = SchemaField(
        name="opaque_key",
        type="string",
        description="Opaque control field.",
        required=False,
        producer_module="planner",
        consumer_modules=("answerer",),
    )
    control_schema = SchemaCandidate(
        schema_id="random_control",
        parent_schema_id=None,
        task="HotpotQA",
        module_fields={"planner": (control_field,)},
        consumption_rules=(
            ConsumptionRule(
                consumer_module="answerer",
                field_name="opaque_key",
                instruction="Use only if relevant.",
                required_behavior="Do not infer evidence.",
                fallback_if_missing="Use original behavior.",
            ),
        ),
        validators={},
        schema_token_budget=128,
        mutation_history=("random_schema_control",),
        proposer_seed=0,
        control_type="random",
    )
    baseline = _candidate_eval_result(schema_id="original_schema", mean_score=0.5)
    primary = _candidate_eval_result(schema_id=primary_schema.schema_id, mean_score=0.4)
    control = _candidate_eval_result(schema_id=control_schema.schema_id, mean_score=0.475)
    return FixedPoolResult(
        baseline_selection_result=baseline,
        baseline_confirmation_result=baseline,
        schema_pool=(primary_schema, control_schema),
        smoke_results=(),
        selection_results=(primary, control),
        top_selection_results=(primary, control),
        confirmation_results=(primary, control),
        primary_confirmation_result=primary,
        best_confirmation_result=control,
        paired_stats=PairedComparison(
            bootstrap=BootstrapDiff(
                mean_diff=-0.1,
                ci_low=-0.2,
                ci_high=0.0,
                n_resamples=10,
            ),
            approximate_randomization_p=1.0,
        ),
        corrected_confirmation_stats={},
        heldout_test_result=None,
        heldout_test_stats=None,
        field_ablation_results=(
            FieldAblationResult(
                ablation="mask",
                field_name="next_query_intent",
                mean_score=0.35,
                drop_vs_unablated=0.05,
                invalid_output_rate=0.0,
                per_example_scores=(0.35,),
            ),
            FieldAblationResult(
                ablation="shuffle",
                field_name="next_query_intent",
                mean_score=0.4,
                drop_vs_unablated=0.0,
                invalid_output_rate=0.0,
                per_example_scores=(0.4,),
            ),
        ),
        decision=MVPDecision(
            proceed=False,
            score_delta=-0.1,
            invalid_output_rate=0.0,
            field_masking_max_drop=0.05,
            reasons=("score delta below bar",),
        ),
        control_guardrail=ControlGuardrail(
            control_in_top_k_warning=True,
            primary_is_control=False,
            primary_control_type=None,
            selection_top_k_control_schema_ids=(control_schema.schema_id,),
            confirmation_top_k_control_schema_ids=(control_schema.schema_id,),
            best_control_schema_id=control_schema.schema_id,
            best_control_confirmation_mean=0.475,
            best_control_vs_primary_delta=0.075,
            warning="control matched primary",
        ),
        cost_summary={},
        budget_summary={},
        proposal_usage={},
        reflection_rounds=(),
        artifacts={},
    )


def _fixed_pool_result_for_causal_pilot_control_primary() -> FixedPoolResult:
    control_schema = make_validator_only_schema(
        task="MuSiQue",
        module_names=("planner", "answerer"),
        seed=0,
    )
    baseline = _candidate_eval_result(schema_id="original_schema", mean_score=0.425)
    primary = _candidate_eval_result(schema_id=control_schema.schema_id, mean_score=0.425)
    return FixedPoolResult(
        baseline_selection_result=baseline,
        baseline_confirmation_result=baseline,
        schema_pool=(control_schema,),
        smoke_results=(),
        selection_results=(primary,),
        top_selection_results=(primary,),
        confirmation_results=(primary,),
        primary_confirmation_result=primary,
        best_confirmation_result=primary,
        paired_stats=PairedComparison(
            bootstrap=BootstrapDiff(
                mean_diff=0.0,
                ci_low=-0.1,
                ci_high=0.1,
                n_resamples=10,
            ),
            approximate_randomization_p=1.0,
        ),
        corrected_confirmation_stats={},
        heldout_test_result=None,
        heldout_test_stats=None,
        field_ablation_results=(),
        decision=MVPDecision(
            proceed=False,
            score_delta=0.0,
            invalid_output_rate=0.0,
            field_masking_max_drop=0.0,
            reasons=("score delta below bar", "no field ablations were produced"),
        ),
        control_guardrail=ControlGuardrail(
            control_in_top_k_warning=True,
            primary_is_control=True,
            primary_control_type="validator_only",
            selection_top_k_control_schema_ids=(control_schema.schema_id,),
            confirmation_top_k_control_schema_ids=(control_schema.schema_id,),
            best_control_schema_id=control_schema.schema_id,
            best_control_confirmation_mean=0.425,
            best_control_vs_primary_delta=0.0,
            warning="Primary selected schema is a control; treat schema-effect claims as null.",
        ),
        cost_summary={},
        budget_summary={},
        proposal_usage={},
        reflection_rounds=(),
        artifacts={},
    )


def test_rollout_cache_reuses_identical_program_example_seed(tmp_path):
    program = build_toy_program()
    calls = {"planner": 0}
    original_runner = program.modules[0].runner

    def counting_runner(state, module, example, context):
        calls["planner"] += 1
        return original_runner(state, module, example, context)

    program.modules[0].runner = counting_runner
    cache = RolloutCache(tmp_path / "cache")
    example = make_toy_examples("cache_validation", 1)

    evaluate_program(
        program=program,
        examples=example,
        scorer=toy_scorer,
        method="cache_test",
        candidate_id="cache_test",
        seed=1,
        rollout_cache=cache,
    )
    evaluate_program(
        program=program,
        examples=example,
        scorer=toy_scorer,
        method="cache_test_again",
        candidate_id="cache_test_again",
        seed=1,
        rollout_cache=cache,
    )

    assert calls["planner"] == 1


def test_rollout_cache_key_includes_example_payload_and_program_metadata(tmp_path):
    cache = RolloutCache(tmp_path / "cache")
    program = build_toy_program()
    example = make_toy_examples("cache_identity", 1)[0]
    same_id_changed_target = ProgramExample(
        example_id=example.example_id,
        split=example.split,
        inputs=dict(example.inputs),
        expected={"answer": "different"},
        metadata=dict(example.metadata),
    )

    original_key = cache.key(program=program, example=example, seed=1)
    changed_example_key = cache.key(program=program, example=same_id_changed_target, seed=1)
    program.modules[0].metadata["temperature"] = 0
    changed_program_key = cache.key(program=program, example=example, seed=1)

    assert changed_example_key != original_key
    assert changed_program_key != original_key


def test_dspy_adapter_wraps_callable_and_preserves_demo_ids():
    def answer_module(question):
        return {"answer": "ok", "confidence": 1.0}

    program = dspy_program_to_lm_program(
        task="AdapterTask",
        modules=(
            DSpyModuleConfig(
                name="answerer",
                module=answer_module,
                input_fields=("question",),
                output_fields=("answer", "confidence"),
                prompt="Answer.",
                model="toy-model",
                demo_ids=("demo_1",),
            ),
        ),
        final_output_module="answerer",
    )

    prediction = program.run(
        make_toy_examples("adapter_validation", 1)[0],
        run_id="adapter",
        method="adapter",
        candidate_id="adapter",
        seed=0,
    )

    assert prediction.final_output["answer"] == "ok"
    assert program.modules[0].metadata["demo_ids"] == ("demo_1",)


def test_openai_adapter_runs_module_with_structured_response():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Response:
                output_text = json.dumps({"answer": "compiler", "confidence": 1.0})

            return Response()

    class FakeClient:
        responses = FakeResponses()

    program = openai_modules_to_lm_program(
        task="HotpotQA",
        modules=(
            OpenAIModuleConfig(
                name="answerer",
                input_fields=("question", "context"),
                output_fields=("answer", "confidence"),
                output_field_types={"answer": "string", "confidence": "number"},
                prompt="Answer as JSON.",
                model="gpt-4.1-mini",
                max_output_tokens=64,
            ),
        ),
        final_output_module="answerer",
        client=FakeClient(),
    )

    example = make_toy_examples("openai_adapter", 2)[1]
    prediction = program.run(
        example,
        run_id="openai_adapter",
        method="openai_adapter",
        candidate_id="openai_adapter",
        seed=0,
    )

    assert prediction.final_output["answer"] == "compiler"
    assert calls[0]["model"] == "gpt-4.1-mini"
    output_schema = calls[0]["text"]["format"]["schema"]
    assert output_schema["properties"]["confidence"]["type"] == "number"
    assert output_schema["required"] == ["answer", "confidence"]


def test_openai_adapter_emits_strict_object_schemas():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Response:
                output_text = json.dumps(
                    {"items": [{"value": "control"}], "metadata": {"value": "control"}}
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    program = openai_modules_to_lm_program(
        task="StrictSchema",
        modules=(
            OpenAIModuleConfig(
                name="collector",
                input_fields=("question",),
                output_fields=("items", "metadata"),
                output_field_types={"items": "array[object]", "metadata": "object"},
                prompt="Collect structured evidence.",
                model="gpt-4.1-mini",
            ),
        ),
        final_output_module="collector",
        client=FakeClient(),
    )

    program.run(
        make_toy_examples("strict_schema", 1)[0],
        run_id="strict_schema",
        method="strict_schema",
        candidate_id="strict_schema",
        seed=0,
    )

    properties = calls[0]["text"]["format"]["schema"]["properties"]
    assert properties["items"]["items"]["additionalProperties"] is False
    assert properties["items"]["items"]["properties"] == {"value": {"type": "string"}}
    assert properties["items"]["items"]["required"] == ["value"]
    assert properties["metadata"]["additionalProperties"] is False
    assert properties["metadata"]["properties"] == {"value": {"type": "string"}}
    assert properties["metadata"]["required"] == ["value"]


def test_openai_adapter_emits_known_nested_object_schemas():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Response:
                output_text = json.dumps(
                    {
                        "claim_atoms": [
                            {
                                "text": "claim",
                                "entities": ["entity"],
                                "relation": "related to",
                                "needs_evidence_from": "source",
                            }
                        ],
                        "evidence_conflict": {
                            "has_conflict": False,
                            "conflict_description": "",
                        },
                    }
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    program = openai_modules_to_lm_program(
        task="HoVer",
        modules=(
            OpenAIModuleConfig(
                name="planner",
                input_fields=("claim", "context"),
                output_fields=("claim_atoms", "evidence_conflict"),
                output_field_types={
                    "claim_atoms": "array[object]",
                    "evidence_conflict": "object",
                },
                prompt="Plan evidence checks.",
                model="gpt-4.1-mini",
            ),
        ),
        final_output_module="planner",
        client=FakeClient(),
    )

    example = ProgramExample(
        example_id="hover_nested",
        split="validation",
        inputs={"claim": "A claim.", "context": "Evidence."},
    )
    prediction = program.run(
        example,
        run_id="hover_nested",
        method="openai_adapter",
        candidate_id="hover_nested",
        seed=0,
    )

    assert prediction.final_output["claim_atoms"][0]["text"] == "claim"
    properties = calls[0]["text"]["format"]["schema"]["properties"]
    atom_item = properties["claim_atoms"]["items"]
    assert atom_item["additionalProperties"] is False
    assert atom_item["required"] == [
        "text",
        "entities",
        "relation",
        "needs_evidence_from",
    ]
    assert atom_item["properties"]["entities"]["items"]["type"] == "string"
    conflict = properties["evidence_conflict"]
    assert conflict["additionalProperties"] is False
    assert conflict["required"] == ["has_conflict", "conflict_description"]
    assert conflict["properties"]["has_conflict"]["type"] == "boolean"


def test_hotpotqa_hover_and_musique_loaders_read_local_json_files(tmp_path):
    hotpot_path = tmp_path / "hotpot.json"
    hotpot_path.write_text(
        json.dumps(
            [
                {
                    "_id": "h1",
                    "question": "Q?",
                    "answer": "A",
                    "context": [["Title", ["Sentence."]]],
                    "supporting_facts": [["Title", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    hover_path = tmp_path / "hover.jsonl"
    hover_path.write_text(
        json.dumps({"uid": "v1", "claim": "Claim.", "label": "SUPPORTED"}) + "\n",
        encoding="utf-8",
    )
    musique_path = tmp_path / "musique.json"
    musique_path.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "question": "Who wrote the book?",
                    "answer": "Ada Lovelace",
                    "answer_aliases": ["Countess of Lovelace"],
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": "Book",
                            "paragraph_text": "The book was written by Ada Lovelace.",
                            "is_supporting": True,
                        }
                    ],
                    "question_decomposition": [{"question": "Who wrote it?", "answer": "Ada Lovelace"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    hotpot = load_hotpotqa_examples(hotpot_path, split="train")
    hover = load_hover_examples(hover_path, split="dev")
    musique = load_musique_examples(musique_path, split="validation_selection")

    assert hotpot[0].inputs["question"] == "Q?"
    assert hotpot[0].expected["answer"] == "A"
    assert hotpot[0].metadata["dataset"] == "hotpotqa"
    assert hover[0].inputs["claim"] == "Claim."
    assert hover[0].expected["label"] == "SUPPORTED"
    assert hover[0].metadata["dataset"] == "hover"
    assert musique[0].inputs["question"] == "Who wrote the book?"
    assert musique[0].inputs["context"][0]["text"] == "The book was written by Ada Lovelace."
    assert musique[0].inputs["context"][0]["is_supporting"] is True
    assert musique[0].expected["answer_aliases"] == ["Countess of Lovelace"]
    assert musique[0].metadata["dataset"] == "musique"


def test_hotpotqa_loader_reads_combined_split_dict(tmp_path):
    hotpot_path = tmp_path / "hotpot_splits.json"
    hotpot_path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "id": "train_1",
                        "question": "Train?",
                        "gold": "train answer",
                        "context": [{"title": "Train", "text": "train answer is supported."}],
                    }
                ],
                "selection": [
                    {
                        "id": "selection_1",
                        "question": "Selection?",
                        "gold": "selection answer",
                        "question_type": "bridge",
                        "context": [{"title": "Selection", "text": "selection answer is supported."}],
                    }
                ],
                "confirmation": [
                    {
                        "id": "confirmation_1",
                        "question": "Confirm?",
                        "gold": "confirm answer",
                        "context": [{"title": "Confirm", "text": "confirm answer is supported."}],
                    }
                ],
                "heldout_validation": [
                    {
                        "id": "heldout_1",
                        "question": "Heldout?",
                        "gold": "heldout answer",
                        "context": [{"title": "Heldout", "text": "heldout answer is supported."}],
                    }
                ],
                "test": [
                    {
                        "id": "test_1",
                        "question": "Test?",
                        "gold": "test answer",
                        "context": [{"title": "Test", "text": "test answer is supported."}],
                    }
                ],
                "corpus": [{"id": "doc_1", "title": "Doc", "text": "unused corpus document."}],
            }
        ),
        encoding="utf-8",
    )

    selection = load_hotpotqa_examples(hotpot_path, split="validation_selection")
    final_test = load_hotpotqa_examples(hotpot_path, split="final_test")
    heldout = load_hotpotqa_examples(
        hotpot_path,
        split="final_test",
        source_split="heldout_validation",
    )
    readiness = load_hotpotqa_examples(hotpot_path, split="readiness")

    assert selection[0].example_id == "selection_1"
    assert selection[0].expected["answer"] == "selection answer"
    assert selection[0].metadata["type"] == "bridge"
    assert selection[0].metadata["source_split"] == "selection"
    assert selection[0].inputs["context"][0]["text"] == "selection answer is supported."
    assert final_test[0].example_id == "test_1"
    assert heldout[0].example_id == "heldout_1"
    assert readiness[0].example_id == "selection_1"


def test_hover_loader_reads_combined_split_dict_and_answer_labels(tmp_path):
    hover_path = tmp_path / "hover_splits.json"
    hover_path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "id": "train_v1",
                        "claim": "Train claim.",
                        "answer": "yes",
                        "context": [{"title": "Train", "text": "Evidence supports it."}],
                    }
                ],
                "selection": [
                    {
                        "id": "selection_v1",
                        "claim": "Selection claim.",
                        "gold": "NOT_SUPPORTED",
                        "evidence": [["Doc", ["Evidence contradicts it."]]],
                        "num_hops": 2,
                    }
                ],
                "confirmation": [
                    {
                        "id": "confirmation_v1",
                        "claim": "Confirmation claim.",
                        "label": "SUPPORTED",
                        "documents": [{"title": "Doc", "text": "Evidence supports it."}],
                    }
                ],
                "test": [
                    {
                        "id": "test_v1",
                        "claim": "Test claim.",
                        "label": "SUPPORTED",
                        "context": "Evidence supports it.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    train = load_hover_examples(hover_path, split="train")
    selection = load_hover_examples(hover_path, split="validation_selection")
    final_test = load_hover_examples(hover_path, split="final_test")

    assert train[0].expected["label"] == "yes"
    assert selection[0].example_id == "selection_v1"
    assert selection[0].expected["label"] == "NOT_SUPPORTED"
    assert selection[0].metadata["source_split"] == "selection"
    assert selection[0].metadata["num_hops"] == 2
    assert selection[0].inputs["context"][0]["text"] == "Evidence contradicts it."
    assert final_test[0].inputs["context"] == ["Evidence supports it."]


def test_benchmark_scorers_normalize_hotpotqa_hover_and_musique_outputs(tmp_path):
    hotpot_path = tmp_path / "hotpot.json"
    hotpot_path.write_text(
        json.dumps([{"_id": "h1", "question": "Q?", "answer": "The Compiler", "context": []}]),
        encoding="utf-8",
    )
    hover_path = tmp_path / "hover.json"
    hover_path.write_text(
        json.dumps([{"uid": "v1", "claim": "Claim.", "label": "SUPPORTED"}]),
        encoding="utf-8",
    )
    musique_path = tmp_path / "musique.json"
    musique_path.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "question": "Q?",
                    "answer": "Ada Lovelace",
                    "answer_aliases": ["Countess of Lovelace"],
                    "paragraphs": [{"paragraph_text": "Ada Lovelace is supported."}],
                }
            ]
        ),
        encoding="utf-8",
    )
    hotpot_example = load_hotpotqa_examples(hotpot_path, split="dev")[0]
    hover_example = load_hover_examples(hover_path, split="dev")[0]
    musique_example = load_musique_examples(musique_path, split="dev")[0]

    class Prediction:
        def __init__(self, final_output):
            self.final_output = final_output

    assert hotpotqa_exact_match(hotpot_example, Prediction({"answer": "compiler"})) == 1.0
    assert hover_label_accuracy(hover_example, Prediction({"label": "supported"})) == 1.0
    assert musique_exact_match(musique_example, Prediction({"answer": "the countess of lovelace"})) == 1.0


def test_openai_hotpotqa_benchmark_uses_preregistered_exact_match(tmp_path):
    hotpot_path = tmp_path / "hotpot.json"
    hotpot_path.write_text(
        json.dumps([{"_id": "h1", "question": "Q?", "answer": "compiler", "context": []}]),
        encoding="utf-8",
    )
    example = load_hotpotqa_examples(hotpot_path, split="dev")[0]

    class Prediction:
        def __init__(self, final_output):
            self.final_output = final_output

    scorer = _scorer_for_dataset("hotpotqa")

    assert scorer(example, Prediction({"answer": "the compiler"})) == 1.0
    assert scorer(example, Prediction({"answer": "the compiler was correct"})) == 0.0


def test_openai_musique_benchmark_uses_musique_identity_and_alias_scorer(tmp_path):
    musique_path = tmp_path / "musique.json"
    musique_path.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "question": "Q?",
                    "answer": "Ada Lovelace",
                    "answer_aliases": ["Countess of Lovelace"],
                    "paragraphs": [{"paragraph_text": "Ada Lovelace is supported."}],
                }
            ]
        ),
        encoding="utf-8",
    )
    example = load_musique_examples(musique_path, split="dev")[0]
    program = build_openai_benchmark_program(
        OpenAIFixedPoolBenchmarkConfig(
            dataset="musique",
            train_path=musique_path,
            selection_path=musique_path,
            confirmation_path=musique_path,
            model="gpt-4.1-mini",
        )
    )

    class Prediction:
        def __init__(self, final_output):
            self.final_output = final_output

    scorer = _scorer_for_dataset("musique")

    assert program.task == "MuSiQue"
    assert "MuSiQue question" in program.modules[-1].prompt
    assert scorer(example, Prediction({"answer": "countess of lovelace"})) == 1.0


def test_musique_human_templates_use_musique_identity():
    schemas = make_human_minimal_schemas(
        task="musique",
        module_names=("planner", "answerer"),
        seed=0,
    )

    assert schemas[0].task == "MuSiQue"
    assert schemas[1].task == "MuSiQue"
    assert schemas[0].schema_id.startswith("musique_human_minimal_")
    assert schemas[1].schema_id.startswith("musique_human_")


def test_benchmark_readiness_reports_local_data_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hotpot_path = tmp_path / "hotpot.json"
    hotpot_path.write_text(
        json.dumps(
            [
                {
                    "_id": "h1",
                    "question": "Q?",
                    "answer": "A",
                    "context": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    readiness = check_benchmark_readiness(hotpotqa_path=hotpot_path)

    assert not readiness.ready
    assert not readiness.datasets["hotpotqa"]["ok"]
    assert readiness.datasets["hotpotqa"]["quality"]["context_coverage"] == 0.0
    assert "OPENAI_API_KEY is not set" in readiness.reasons
    permissive = check_benchmark_readiness(hotpotqa_path=hotpot_path, require_context=False)
    assert permissive.datasets["hotpotqa"]["ok"]

    hotpot_path.write_text(
        json.dumps([{"_id": "h1", "question": "Q?", "answer": "A", "context": [["Title", ["Sentence."]]]}]),
        encoding="utf-8",
    )
    contextual = check_benchmark_readiness(hotpotqa_path=hotpot_path, require_context=True)
    assert contextual.datasets["hotpotqa"]["quality"]["context_coverage"] == 1.0

    hotpot_path.write_text(
        json.dumps(
            {
                "selection": [
                    {
                        "id": "selection_1",
                        "question": "Q?",
                        "gold": "A",
                        "context": [{"title": "Title", "text": "Sentence."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    split_dict = check_benchmark_readiness(hotpotqa_path=hotpot_path, require_context=True)
    assert split_dict.datasets["hotpotqa"]["ok"]
    assert split_dict.datasets["hotpotqa"]["quality"]["target_coverage"] == 1.0

    hover_path = tmp_path / "hover.json"
    hover_path.write_text(
        json.dumps(
            {
                "selection": [
                    {
                        "id": "hover_selection_1",
                        "claim": "Claim.",
                        "answer": "yes",
                        "context": [{"title": "Doc", "text": "Evidence supports it."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    hover_ready = check_benchmark_readiness(hover_path=hover_path, require_context=True)
    assert hover_ready.datasets["hover"]["ok"]
    assert hover_ready.datasets["hover"]["quality"]["target_coverage"] == 1.0


def test_cli_check_benchmark_readiness_strict_fails_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hotpot_path = tmp_path / "hotpot.json"
    hotpot_path.write_text(
        json.dumps([{"_id": "h1", "question": "Q?", "answer": "A", "context": []}]),
        encoding="utf-8",
    )

    exit_code = cli_main(
        [
            "check-benchmark-readiness",
            "--hotpotqa",
            str(hotpot_path),
            "--strict",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["datasets"]["hotpotqa"]["ok"] is False
    assert output["ready"] is False


def test_openai_fixed_pool_benchmark_runs_with_injected_client(tmp_path):
    def write_hotpot(path, example_id):
        path.write_text(
            json.dumps(
                [
                    {
                        "_id": example_id,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    train_path = tmp_path / "train.json"
    selection_path = tmp_path / "selection.json"
    confirmation_path = tmp_path / "confirmation.json"
    write_hotpot(train_path, "train_1")
    write_hotpot(selection_path, "selection_1")
    write_hotpot(confirmation_path, "confirmation_1")

    class FakeResponses:
        def create(self, **kwargs):
            schema = kwargs["text"]["format"]["schema"]

            class Response:
                output_text = json.dumps(
                    {
                        key: _fake_schema_value(key, value)
                        for key, value in schema["properties"].items()
                    }
                )

            return Response()

    class FakeClient:
        responses = FakeResponses()

    result = run_openai_fixed_pool_benchmark(
        benchmark_config=OpenAIFixedPoolBenchmarkConfig(
            dataset="hotpotqa",
            train_path=train_path,
            selection_path=selection_path,
            confirmation_path=confirmation_path,
            model="gpt-4.1-mini",
        ),
        fixed_pool_config=FixedPoolConfig(
            task="hotpotqa",
            target_model="gpt-4.1-mini",
            seed=3,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            model_prices={
                "gpt-4.1-mini": {
                    "input_per_million": 1.0,
                    "output_per_million": 1.0,
                    "source_date": "test",
                }
            },
        ),
        artifact_dir=tmp_path / "artifacts",
        client=FakeClient(),
    )

    assert result.baseline_confirmation_result.mean_score == 1.0
    assert result.primary_confirmation_result.mean_score == 1.0
    assert result.cost_summary["prompt_tokens"] > 0
    assert result.cost_summary["dollar_cost"] > 0
    assert (tmp_path / "artifacts" / "results" / "mvp_summary.json").exists()


def test_fixed_pool_split_readiness_checks_all_runtime_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def write_hotpot(path, example_id, *, context):
        path.write_text(
            json.dumps(
                [
                    {
                        "_id": example_id,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": context,
                    }
                ]
            ),
            encoding="utf-8",
        )

    train_path = tmp_path / "train.json"
    selection_path = tmp_path / "selection.json"
    confirmation_path = tmp_path / "confirmation.json"
    write_hotpot(train_path, "train_1", context=[["Title", ["alpha is supported."]]])
    write_hotpot(selection_path, "selection_1", context=[["Title", ["alpha is supported."]]])
    write_hotpot(confirmation_path, "confirmation_1", context=[])

    readiness = check_fixed_pool_split_readiness(
        dataset="hotpotqa",
        train_path=train_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        require_context=True,
    )

    assert readiness.datasets["hotpotqa.train"]["ok"]
    assert readiness.datasets["hotpotqa.selection"]["ok"]
    assert not readiness.datasets["hotpotqa.confirmation"]["ok"]
    assert any("confirmation data unavailable" in reason for reason in readiness.reasons)


def test_fixed_pool_split_readiness_rejects_cross_split_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def write_hotpot(path, example_id):
        path.write_text(
            json.dumps(
                [
                    {
                        "_id": example_id,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    train_path = tmp_path / "train.json"
    selection_path = tmp_path / "selection.json"
    confirmation_path = tmp_path / "confirmation.json"
    write_hotpot(train_path, "shared_id")
    write_hotpot(selection_path, "shared_id")
    write_hotpot(confirmation_path, "confirmation_id")

    readiness = check_fixed_pool_split_readiness(
        dataset="hotpotqa",
        train_path=train_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        require_context=True,
    )

    assert readiness.datasets["hotpotqa.train"]["ok"]
    assert readiness.datasets["hotpotqa.selection"]["ok"]
    assert not readiness.ready
    assert any("overlap between train and selection" in reason for reason in readiness.reasons)


def test_fixed_pool_split_readiness_rejects_dataset_path_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "schemaevo.benchmarks.readiness.importlib.util.find_spec",
        lambda _name: object(),
    )
    musique_dir = tmp_path / "data" / "musique"
    musique_dir.mkdir(parents=True)

    for name in ("train", "selection", "confirmation"):
        (musique_dir / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    readiness = check_fixed_pool_split_readiness(
        dataset="hotpotqa",
        train_path=musique_dir / "train.json",
        selection_path=musique_dir / "selection.json",
        confirmation_path=musique_dir / "confirmation.json",
        require_context=True,
    )

    assert not readiness.ready
    assert any("use --dataset musique" in reason for reason in readiness.reasons)


def test_cli_run_openai_fixed_pool_checks_all_split_readiness(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name, context in (
        ("train", [["Title", ["alpha is supported."]]]),
        ("selection", [["Title", ["alpha is supported."]]]),
        ("confirmation", []),
    ):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": context,
                    }
                ]
            ),
            encoding="utf-8",
        )

    exit_code = cli_main(
        [
            "run-openai-fixed-pool",
            "--dataset",
            "hotpotqa",
            "--train",
            str(tmp_path / "train.json"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["datasets"]["hotpotqa.selection"]["ok"] is True
    assert output["datasets"]["hotpotqa.confirmation"]["ok"] is False
    assert any("confirmation data unavailable" in reason for reason in output["reasons"])


def test_cli_run_openai_fixed_pool_fails_readiness_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in ("train", "selection", "confirmation"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    exit_code = cli_main(
        [
            "run-openai-fixed-pool",
            "--dataset",
            "hotpotqa",
            "--train",
            str(tmp_path / "train.json"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["ready"] is False
    assert "OPENAI_API_KEY is not set" in output["reasons"]


def test_cli_run_openai_fixed_pool_reports_readiness_and_accounting_blockers_together(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_find_spec(name):
        return None if name == "openai" else object()

    monkeypatch.setattr("schemaevo.benchmarks.readiness.importlib.util.find_spec", fake_find_spec)
    for name in ("train", "selection", "confirmation"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    exit_code = cli_main(
        [
            "run-openai-fixed-pool",
            "--config",
            "configs/mvp_hotpotqa_gpt41mini.yaml",
            "--dataset",
            "hotpotqa",
            "--train",
            str(tmp_path / "train.json"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["ready"] is False
    assert output["openai_package"] is False
    assert output["accounting"]["ok"] is False
    assert "openai package is not installed" in output["reasons"]
    assert any("missing pricing table" in reason for reason in output["reasons"])


def test_cli_run_openai_fixed_pool_requires_model_prices(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name in ("train", "selection", "confirmation"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    exit_code = cli_main(
        [
            "run-openai-fixed-pool",
            "--config",
            "configs/mvp_hotpotqa_gpt41mini.yaml",
            "--dataset",
            "hotpotqa",
            "--train",
            str(tmp_path / "train.json"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["accounting"]["ok"] is False
    assert any("missing pricing table" in reason for reason in output["reasons"])


def test_cli_run_openai_fixed_pool_requires_tiktoken_costing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "no_tiktoken.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "fixed_pool": {
                    "task": "HotpotQA",
                    "target_model": "gpt-4.1-mini",
                    "n_trace_schemas": 1,
                    "n_random_schemas": 0,
                    "top_k_confirmation": 1,
                    "bootstrap_resamples": 10,
                    "randomization_swaps": 10,
                    "use_tiktoken_costing": False,
                },
                "train_examples": 1,
                "selection_examples": 1,
                "confirmation_examples": 1,
                "proposer": {"kind": "heuristic"},
            }
        ),
        encoding="utf-8",
    )
    for name in ("train", "selection", "confirmation"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "_id": name,
                        "question": "What is the answer?",
                        "answer": "alpha",
                        "context": [["Title", ["alpha is supported."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )

    exit_code = cli_main(
        [
            "run-openai-fixed-pool",
            "--config",
            str(config_path),
            "--dataset",
            "hotpotqa",
            "--train",
            str(tmp_path / "train.json"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
            "--input-price-per-million",
            "1.0",
            "--output-price-per-million",
            "2.0",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["accounting"]["ok"] is False
    assert any("tiktoken costing" in reason for reason in output["reasons"])
    assert not any("missing pricing table" in reason for reason in output["reasons"])


def test_causal_pilot_and_deployment_reports_write_artifacts(tmp_path):
    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=(),
        selection_examples=make_toy_examples("validation_selection", 3),
        confirmation_examples=make_toy_examples("validation_confirmation", 3),
        scorer=toy_scorer,
        config=FixedPoolConfig(
            task="toy_multihop",
            target_model="toy-model",
            seed=61,
            n_trace_schemas=1,
            n_random_schemas=0,
            top_k_confirmation=1,
            bootstrap_resamples=10,
            randomization_swaps=10,
            progress="none",
        ),
        artifact_dir=tmp_path / "run",
    )

    causal = build_causal_pilot_report(
        result=result,
        dataset="toy",
        model="toy-model",
        artifact_dir=tmp_path / "reports",
    )
    deployment = build_fixed_pool_deployment_report(
        result=result,
        artifact_dir=tmp_path / "reports",
    )

    assert causal.proceed
    assert causal.max_shuffle_drop > 0.0
    assert causal.empirical_status == "positive_schema_signal"
    assert causal.ablation_signal_interpretable
    assert causal.ablation_supports_primary_gain
    assert not causal.null_signal_warning
    assert deployment.serving_invariant
    assert Path(causal.artifacts["summary"]).exists()
    assert Path(deployment.artifacts["summary"]).exists()


def test_causal_pilot_reports_negative_null_signal_when_primary_loses_to_control():
    result = _fixed_pool_result_for_causal_pilot_null_signal()

    causal = build_causal_pilot_report(
        result=result,
        dataset="musique",
        model="gpt-4.1-mini",
    )

    assert not causal.proceed
    assert causal.empirical_status == "negative_or_no_primary_gain"
    assert causal.null_signal_warning
    assert not causal.ablation_signal_interpretable
    assert not causal.ablation_supports_primary_gain
    assert causal.best_control_vs_primary_delta == pytest.approx(0.075)
    assert "field ablations are not interpretable" in " ".join(causal.reasons)
    assert "control schema matched or beat" in " ".join(causal.reasons)


def test_causal_pilot_reports_control_primary_as_distinct_null_signal():
    result = _fixed_pool_result_for_causal_pilot_control_primary()

    causal = build_causal_pilot_report(
        result=result,
        dataset="musique",
        model="gpt-4.1-mini",
    )

    assert not causal.proceed
    assert causal.empirical_status == "control_selected_as_primary"
    assert causal.null_signal_warning
    assert causal.primary_is_control
    assert causal.primary_control_type == "validator_only"
    assert not causal.ablation_signal_interpretable
    assert not causal.ablation_supports_primary_gain
    assert causal.best_control_vs_primary_delta == pytest.approx(0.0)
    joined = " ".join(causal.reasons)
    assert "primary selected schema is a validator_only control" in joined
    assert "field ablations are not applicable" in joined
    assert "field ablation results are missing" not in joined


def test_budget_pareto_report_aggregates_summary_files(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "primary_confirmation_mean": 0.6,
                "cost_summary": {
                    "target_task_calls": 10,
                    "retriever_calls": 5,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "dollar_cost": 0.10,
                    "max_p95_latency_ms": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "primary_confirmation_mean": 0.7,
                "cost_summary": {
                    "target_task_calls": 10,
                    "retriever_calls": 5,
                    "prompt_tokens": 120,
                    "completion_tokens": 60,
                    "total_tokens": 180,
                    "dollar_cost": 0.12,
                    "max_p95_latency_ms": 14,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_budget_pareto_report(
        run_paths={"first": first, "second": second},
        artifact_dir=tmp_path / "pareto",
    )

    assert {row.method for row in report.rows} == {"first", "second"}
    assert "second" in report.pareto_methods
    assert Path(report.artifacts["csv"]).exists()


def test_external_prompt_optimizer_applies_prompt_patch(tmp_path):
    script = (
        "import json, os; "
        "data=json.load(open(os.environ['SCHEMAEVO_INPUT_PROGRAM'])); "
        "data['modules'][0]['prompt'] += '\\nPatched externally.'; "
        "json.dump(data, open(os.environ['SCHEMAEVO_OUTPUT_PROGRAM'], 'w'))"
    )
    optimizer = ExternalPromptOptimizer(
        name="test_prompt_optimizer",
        command=f"{sys.executable} -c \"{script}\"",
        artifact_dir=tmp_path,
    )

    optimized = optimizer(build_toy_program())

    assert "Patched externally." in optimized.modules[0].prompt
    assert (tmp_path / "test_prompt_optimizer_input_program.json").exists()
    assert (tmp_path / "test_prompt_optimizer_output_program.json").exists()


def test_composability_harness_runs_external_prompt_optimizer_then_schemaevo(tmp_path):
    def prompt_optimizer(program):
        optimized = program.clone()
        optimized.modules[0].prompt = optimized.modules[0].prompt + "\nExternal optimizer text."
        return optimized

    result = run_prompt_optimizer_then_schemaevo(
        base_program=build_toy_program(),
        prompt_optimizer=prompt_optimizer,
        prompt_eval_examples=make_toy_examples("prompt_eval", 3),
        schema_optimizer_examples=make_toy_examples("schema_train", 8),
        scorer=toy_scorer,
        schema_config=SchemaEvoConfig(
            task="toy_multihop",
            seed=29,
            max_program_rollouts=3,
            max_mutation_attempts=5,
            minibatch_size=4,
            initial_random_schemas=0,
            k_final=1,
        ),
        artifact_dir=tmp_path,
    )

    assert result.prompt_delta == pytest.approx(0.0)
    assert result.schemaevo_additive_delta == pytest.approx(1.0)
    assert result.schemaevo_eval_results
    assert result.best_schemaevo_eval_result is not None
    assert result.summary()["same_eval_examples"] is True
    assert result.summary()["paired_stats"]["schemaevo_vs_prompt"]["bootstrap"]["ci_low"] >= 1.0
    assert result.budget_summary["evaluation"]["target_task_calls"] > 0
    assert result.schemaevo_result.final_records
    assert (tmp_path / "composability_summary.json").exists()


def _fake_schema_value(field_name, schema):
    if field_name == "answer":
        return "alpha"
    if field_name == "label":
        return "SUPPORTED"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next(item for item in schema_type if item != "null")
    if "enum" in schema:
        return schema["enum"][0]
    if schema_type == "number":
        return 1.0
    if schema_type == "integer":
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        item_type = schema.get("items", {}).get("type")
        return [{"value": "control"}] if item_type == "object" else ["control"]
    if schema_type == "object":
        return {"value": "control"}
    return "control"
