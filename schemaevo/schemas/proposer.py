from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from typing import Any, Protocol

from schemaevo.schemas.candidate import ConsumptionRule, SchemaCandidate, SchemaField
from schemaevo.schemas.grammar import SchemaGrammar
from schemaevo.schemas.human_templates import make_hotpotqa_schema_candidate, make_hover_schema_candidate


@dataclass(frozen=True)
class TraceExample:
    example_id: str
    split: str
    module_name: str
    input_summary: str
    output_summary: str
    score: float | None = None
    errors: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


class SchemaProposer(Protocol):
    def propose(
        self,
        *,
        traces: tuple[TraceExample, ...],
        task: str,
        module_names: tuple[str, ...],
        n: int,
        seed: int,
        schema_token_budget: int,
    ) -> list[SchemaCandidate]:
        ...


class HeuristicTraceSchemaProposer:
    """Deterministic train-trace-only proposer used when no LLM proposer is wired.

    This is not a competing optimization method. It gives the SchemaEvo pipeline a
    reproducible local proposal source and keeps the LLM proposer as a pluggable
    boundary for real experiments.
    """

    def propose(
        self,
        *,
        traces: tuple[TraceExample, ...],
        task: str,
        module_names: tuple[str, ...],
        n: int,
        seed: int,
        schema_token_budget: int,
    ) -> list[SchemaCandidate]:
        _assert_train_only(traces)
        if n <= 0:
            return []
        rng = random.Random(seed)
        base = _base_template(task, module_names, seed)
        producer = module_names[0]
        fields = list(base.module_fields[producer])
        trace_text = " ".join(
            f"{trace.input_summary} {trace.output_summary} {' '.join(trace.errors)}"
            for trace in traces
        ).lower()
        fields = _prioritize_fields(fields, trace_text)
        candidates: list[SchemaCandidate] = []
        for index in range(n):
            # Vary subset size and field order, but keep semantic fields from the task family.
            rng.shuffle(fields)
            subset_size = 1 + (index % max(1, min(len(fields), 6)))
            selected = tuple(fields[:subset_size])
            rules = tuple(
                ConsumptionRule(
                    consumer_module=module_names[-1],
                    field_name=field.name,
                    instruction=(
                        f"Consume `{field.name}` as typed intermediate evidence from `{producer}`."
                    ),
                    required_behavior=(
                        "Use this field to preserve information flow; do not add calls or hidden retries."
                    ),
                    fallback_if_missing="Fall back to the original prompt behavior and mark uncertainty.",
                )
                for field in selected
            )
            candidate = SchemaCandidate(
                schema_id=f"trace_schema_{index}",
                parent_schema_id=None,
                task=base.task,
                module_fields={producer: selected},
                consumption_rules=rules,
                validators={field.name: field.validation_rule or "" for field in selected},
                schema_token_budget=schema_token_budget,
                mutation_history=("heuristic_train_trace_proposal", f"subset_size={subset_size}"),
                proposer_seed=seed,
                control_type="schemaevo",
                metadata={
                    "proposal_index": index,
                    "trace_count": len(traces),
                    "split_source": "train_only",
                },
            ).with_id_from_content(prefix="trace")
            candidates.append(candidate)
        return _dedupe(candidates)[:n]


class OpenAISchemaProposer:
    """Reflective schema proposer backed by the OpenAI Responses API.

    The default model is `gpt-4.1-mini` per the project requirement. Tests can
    inject a fake client with a compatible `responses.create(...)` method.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        client: Any | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = client
        self.raise_on_error = raise_on_error
        self.last_errors: list[str] = []

    def propose(
        self,
        *,
        traces: tuple[TraceExample, ...],
        task: str,
        module_names: tuple[str, ...],
        n: int,
        seed: int,
        schema_token_budget: int,
    ) -> list[SchemaCandidate]:
        _assert_train_only(traces)
        self.last_errors = []
        if n <= 0:
            return []
        try:
            client = self.client or _make_openai_client()
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You propose SchemaEvo intermediate schema contracts for multi-module "
                            "LLM programs. Return only fields that preserve the fixed call graph: no "
                            "extra LLM calls, no extra retrieval, no self-consistency, no tools, no test "
                            "labels. Use snake_case field names and executable validator constraints "
                            "when possible."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": task,
                                "module_names": module_names,
                                "schema_count": n,
                                "schema_token_budget": schema_token_budget,
                                "train_traces": [trace.__dict__ for trace in traces],
                            },
                            sort_keys=True,
                            ensure_ascii=True,
                        ),
                    },
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "schemaevo_proposals",
                        "strict": True,
                        "schema": _proposal_json_schema(),
                    }
                },
            )
            payload = _extract_response_json(response)
            candidates = _candidates_from_openai_payload(
                payload=payload,
                task=task,
                module_names=module_names,
                seed=seed,
                schema_token_budget=schema_token_budget,
                model=self.model,
                errors=self.last_errors,
            )
        except Exception as exc:
            if self.raise_on_error:
                raise
            self.last_errors.append(f"openai_proposal_failed: {exc}")
            return []
        return _dedupe(candidates)[:n]


def propose_schemas_from_traces(
    *,
    traces: tuple[TraceExample, ...],
    task: str,
    module_names: tuple[str, ...],
    n: int,
    seed: int,
    schema_token_budget: int = 512,
    proposer: SchemaProposer | None = None,
) -> list[SchemaCandidate]:
    proposer = proposer or HeuristicTraceSchemaProposer()
    return proposer.propose(
        traces=traces,
        task=task,
        module_names=module_names,
        n=n,
        seed=seed,
        schema_token_budget=schema_token_budget,
    )


def _make_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAISchemaProposer")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the `openai` package to use OpenAISchemaProposer") from exc
    return OpenAI()


def _proposal_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schemas"],
        "properties": {
            "schemas": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rationale", "fields"],
                    "properties": {
                        "rationale": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "name",
                                    "type",
                                    "description",
                                    "required",
                                    "producer_module",
                                    "consumer_modules",
                                    "enum_values",
                                    "max_items",
                                    "max_tokens",
                                    "validator",
                                    "causal_hypothesis",
                                ],
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "string",
                                            "boolean",
                                            "number",
                                            "integer",
                                            "enum",
                                            "array[string]",
                                            "array[object]",
                                            "object",
                                        ],
                                    },
                                    "description": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "producer_module": {"type": "string"},
                                    "consumer_modules": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "string"},
                                    },
                                    "enum_values": {
                                        "type": ["array", "null"],
                                        "items": {"type": "string"},
                                    },
                                    "max_items": {"type": ["integer", "null"]},
                                    "max_tokens": {"type": ["integer", "null"]},
                                    "validator": {"type": "string"},
                                    "causal_hypothesis": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _extract_response_json(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not text:
        output = getattr(response, "output", None)
        if output:
            parts: list[str] = []
            for item in output:
                for content in getattr(item, "content", []) or []:
                    value = getattr(content, "text", None)
                    if value:
                        parts.append(value)
            text = "\n".join(parts)
    if not text and isinstance(response, dict):
        text = response.get("output_text")
    if not text:
        raise ValueError("OpenAI response did not contain output_text")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI schema proposal response must be a JSON object")
    return parsed


def _candidates_from_openai_payload(
    *,
    payload: dict[str, Any],
    task: str,
    module_names: tuple[str, ...],
    seed: int,
    schema_token_budget: int,
    model: str,
    errors: list[str] | None = None,
) -> list[SchemaCandidate]:
    module_set = set(module_names)
    grammar = SchemaGrammar(allowed_modules=module_names, max_schema_tokens=schema_token_budget)
    candidates: list[SchemaCandidate] = []
    for index, proposal in enumerate(payload.get("schemas", [])):
        fields_by_module: dict[str, list[SchemaField]] = {}
        for item in proposal.get("fields", []):
            try:
                producer = item["producer_module"]
                if producer not in module_set:
                    producer = module_names[0]
                consumers = tuple(
                    consumer for consumer in item["consumer_modules"] if consumer in module_set
                ) or (module_names[-1],)
                field = SchemaField(
                    name=item["name"],
                    type=item["type"],
                    description=item["description"],
                    required=bool(item["required"]),
                    producer_module=producer,
                    consumer_modules=consumers,
                    enum_values=tuple(item["enum_values"]) if item.get("enum_values") else None,
                    max_items=item.get("max_items"),
                    max_tokens=item.get("max_tokens"),
                    validation_rule=item.get("validator") or None,
                    causal_hypothesis=item.get("causal_hypothesis"),
                )
            except Exception as exc:
                if errors is not None:
                    errors.append(f"schema[{index}] field skipped: {exc}")
                continue
            fields_by_module.setdefault(producer, []).append(field)
        fields = {
            module_name: tuple(module_fields)
            for module_name, module_fields in fields_by_module.items()
        }
        if not any(fields.values()):
            if errors is not None:
                errors.append(f"schema[{index}] skipped: no valid fields")
            continue
        try:
            rules = tuple(
                ConsumptionRule(
                    consumer_module=consumer,
                    field_name=field.name,
                    instruction=f"Use `{field.name}` as typed evidence from `{field.producer_module}`.",
                    required_behavior="Consume the field without adding calls or changing retrieval.",
                    fallback_if_missing="Use original prompt behavior and preserve uncertainty.",
                )
                for module_fields in fields.values()
                for field in module_fields
                for consumer in field.consumer_modules
            )
            candidate = SchemaCandidate(
                schema_id=f"openai_schema_{index}",
                parent_schema_id=None,
                task=task,
                module_fields=fields,
                consumption_rules=rules,
                validators={
                    field.name: field.validation_rule or ""
                    for module_fields in fields.values()
                    for field in module_fields
                },
                schema_token_budget=schema_token_budget,
                mutation_history=("openai_reflective_schema_proposal",),
                proposer_seed=seed,
                control_type="schemaevo",
                metadata={
                    "proposal_index": index,
                    "proposer_model": model,
                    "rationale": proposal.get("rationale", ""),
                },
            ).with_id_from_content(prefix="openai")
        except Exception as exc:
            if errors is not None:
                errors.append(f"schema[{index}] skipped: {exc}")
            continue
        static_check = grammar.check_candidate(candidate)
        if not static_check.passed:
            if errors is not None:
                errors.append(f"schema[{index}] skipped: {'; '.join(static_check.errors)}")
            continue
        candidates.append(candidate)
    return candidates


def _assert_train_only(traces: tuple[TraceExample, ...]) -> None:
    bad = [trace.example_id for trace in traces if trace.split != "train"]
    if bad:
        raise ValueError(f"schema proposals may use train traces only; bad examples: {bad[:5]}")


def _base_template(task: str, module_names: tuple[str, ...], seed: int) -> SchemaCandidate:
    normalized = task.lower()
    if normalized in {"hover", "hovertask"}:
        return make_hover_schema_candidate(module_names=module_names, seed=seed)
    if normalized in {"hotpotqa", "hotpot", "toy_multihop"}:
        return make_hotpotqa_schema_candidate(module_names=module_names, seed=seed)
    raise ValueError(f"unsupported task for trace proposer: {task}")


def _prioritize_fields(fields: list[SchemaField], trace_text: str) -> list[SchemaField]:
    keywords = {
        "bridge_entity": ("bridge", "entity", "missing hop"),
        "next_query_intent": ("query", "retrieve", "next", "missing"),
        "evidence_conflict": ("conflict", "contradict"),
        "missing_evidence_reason": ("missing", "insufficient"),
        "claim_atoms": ("claim", "atom", "subclaim"),
        "hop_plan": ("hop", "plan", "retrieve"),
        "evidence_table": ("evidence", "sentence", "title"),
    }

    def score(field: SchemaField) -> tuple[int, str]:
        terms = keywords.get(field.name, ())
        return (sum(term in trace_text for term in terms), field.name)

    return sorted(fields, key=score, reverse=True)


def _dedupe(candidates: list[SchemaCandidate]) -> list[SchemaCandidate]:
    seen: set[str] = set()
    unique: list[SchemaCandidate] = []
    for candidate in candidates:
        if candidate.schema_id in seen:
            continue
        seen.add(candidate.schema_id)
        unique.append(candidate)
    return unique
