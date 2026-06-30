from __future__ import annotations

from schemaevo.eval.scoring import Scorer
from schemaevo.programs.base import (
    LMProgram,
    ModuleExecutionContext,
    ModuleSignature,
    ModuleSpec,
    ProgramExample,
    ProgramPrediction,
)
from schemaevo.schemas.proposer import TraceExample


def build_toy_program() -> LMProgram:
    planner = ModuleSpec(
        name="planner",
        signature=ModuleSignature(
            input_fields=("question", "context"),
            output_fields=("plan_summary",),
        ),
        prompt="Identify the multi-hop plan and pass only the original output fields.",
        model="toy-model",
        max_output_tokens=256,
        runner=_planner_runner,
        llm_calls=1,
        retriever_calls=1,
    )
    answerer = ModuleSpec(
        name="answerer",
        signature=ModuleSignature(
            input_fields=("question", "context", "plan_summary"),
            output_fields=("answer", "confidence"),
        ),
        prompt="Answer from the provided intermediate state.",
        model="toy-model",
        max_output_tokens=128,
        runner=_answerer_runner,
        llm_calls=1,
        retriever_calls=0,
    )
    return LMProgram(
        task="HotpotQA",
        modules=(planner, answerer),
        retriever_top_k=3,
        final_output_module="answerer",
    )


def make_toy_traces() -> tuple[TraceExample, ...]:
    return tuple(
        TraceExample(
            example_id=f"train_trace_{index}",
            split="train",
            module_name="planner",
            input_summary="bridge entity and next query intent were missing from the fixed schema",
            output_summary="planner should emit bridge_entity and next_query_intent for downstream answerer",
            score=0.0,
            errors=("missing bridge entity", "missing next query intent"),
        )
        for index in range(12)
    )


def make_toy_examples(split: str, n: int) -> tuple[ProgramExample, ...]:
    examples: list[ProgramExample] = []
    for index in range(n):
        bridge = "Ada Lovelace" if index % 2 == 0 else "Grace Hopper"
        answer = "analytical engine" if index % 2 == 0 else "compiler"
        examples.append(
            ProgramExample(
                example_id=f"{split}_{index}",
                split=split,
                inputs={
                    "question": f"What artifact is associated with the bridge entity in example {index}?",
                    "context": (
                        f"The first hop identifies {bridge}. The second hop links the entity to {answer}."
                    ),
                },
                expected={"answer": answer},
                metadata={
                    "bridge_entity": bridge,
                    "next_query_intent": f"find the artifact associated with {bridge}",
                    "missing_evidence_reason": "the downstream answerer needs the bridge entity",
                    "answer_type_expected": "other",
                    "question_type": "bridge",
                    "candidate_answer_constraints": answer,
                },
            )
        )
    return tuple(examples)


def toy_scorer(example: ProgramExample, prediction: ProgramPrediction) -> float:
    return 1.0 if prediction.final_output.get("answer") == example.expected.get("answer") else 0.0


def _planner_runner(
    state: dict,
    module: ModuleSpec,
    example: ProgramExample,
    context: ModuleExecutionContext,
) -> dict:
    output = {"plan_summary": "use the fixed two-module multi-hop path"}
    for field_spec in module.metadata.get("schemaevo_output_fields", []):
        name = field_spec["name"]
        output[name] = _semantic_or_default_value(field_spec, example)
    return output


def _answerer_runner(
    state: dict,
    module: ModuleSpec,
    example: ProgramExample,
    context: ModuleExecutionContext,
) -> dict:
    schema_fields = state["schema_fields"]
    bridge = schema_fields.get("bridge_entity")
    intent = schema_fields.get("next_query_intent")
    if bridge:
        context.record_field_use(
            producer_module=state["field_producers"].get("bridge_entity", "planner"),
            consumer_module=module.name,
            field_name="bridge_entity",
            behavior="used bridge entity to connect hops",
        )
    if intent:
        context.record_field_use(
            producer_module=state["field_producers"].get("next_query_intent", "planner"),
            consumer_module=module.name,
            field_name="next_query_intent",
            behavior="used query intent to preserve missing evidence need",
        )
    if bridge == example.metadata["bridge_entity"] and intent == example.metadata["next_query_intent"]:
        return {"answer": example.expected["answer"], "confidence": 1.0}
    return {"answer": "unknown", "confidence": 0.0}


def _semantic_or_default_value(field_spec: dict, example: ProgramExample):
    name = field_spec["name"]
    field_type = field_spec["type"]
    if name == "bridge_entity":
        return example.metadata["bridge_entity"]
    if name == "next_query_intent":
        return example.metadata["next_query_intent"]
    if name == "missing_evidence_reason":
        return example.metadata["missing_evidence_reason"]
    if name == "question_type":
        return example.metadata["question_type"]
    if name == "answer_type_expected":
        return example.metadata["answer_type_expected"]
    if name == "candidate_answer_constraints":
        return [example.metadata["candidate_answer_constraints"]]
    if name == "comparison_axes":
        return []
    if name == "evidence_needs":
        return [{"needed_fact": example.metadata["next_query_intent"], "resolved": False}]
    if field_type == "string":
        return "control"
    if field_type == "boolean":
        return False
    if field_type == "number":
        return 0.0
    if field_type == "integer":
        return 0
    if field_type == "enum":
        enum_values = field_spec.get("enum_values") or ["alpha"]
        return enum_values[0]
    if field_type == "array[string]":
        return ["control"]
    if field_type == "array[object]":
        return [{"control": True}]
    if field_type == "object":
        return {"control": True}
    return None


TOY_SCORER: Scorer = toy_scorer
