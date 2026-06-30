from __future__ import annotations

import json

import pytest

from schemaevo.examples.toy_multihop import (
    build_toy_program,
    make_toy_examples,
    make_toy_traces,
    toy_scorer,
)
from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.scoring import evaluate_program
from schemaevo.optimizers.fixed_pool_schema import FixedPoolConfig, run_fixed_pool_schema_mvp
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, merge_schema_candidates, schema_evo_optimize
from schemaevo.programs.call_graph import assert_same_call_graph, extract_call_graph
from schemaevo.programs.compile_schema_program import CONTRACT_START, compile_schema_program
from schemaevo.schemas.candidate import ConsumptionRule, SchemaCandidate, SchemaField
from schemaevo.schemas.grammar import MutationOp, SchemaGrammar, assert_legal_mutation
from schemaevo.schemas.human_templates import make_hotpotqa_schema_candidate, make_human_minimal_schemas
from schemaevo.schemas.mutations import Mutation, apply_mutation
from schemaevo.schemas.proposer import OpenAISchemaProposer, TraceExample, propose_schemas_from_traces
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

    proposer = OpenAISchemaProposer(client=FakeClient())
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


def test_fixed_pool_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="top_k_confirmation"):
        FixedPoolConfig(top_k_confirmation=0)


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
