from __future__ import annotations

from schemaevo.schemas.candidate import ConsumptionRule, SchemaCandidate, SchemaField


def _module_pair(module_names: tuple[str, ...] | None) -> tuple[str, str]:
    modules = module_names or ("planner", "answerer")
    if len(modules) < 2:
        raise ValueError("at least two modules are required for a producer/consumer schema")
    return modules[0], modules[-1]


def _candidate(
    *,
    schema_id: str,
    task: str,
    producer: str,
    consumer: str,
    fields: tuple[SchemaField, ...],
    seed: int,
    control_type: str = "human",
) -> SchemaCandidate:
    rules = tuple(
        ConsumptionRule(
            consumer_module=consumer,
            field_name=field.name,
            instruction=f"Use `{field.name}` from `{producer}` when deciding the final response.",
            required_behavior=(
                "Treat the field as an intermediate evidence contract, not as an extra model call."
            ),
            fallback_if_missing="If the field is missing or invalid, fall back to the original prompt behavior.",
        )
        for field in fields
    )
    candidate = SchemaCandidate(
        schema_id=schema_id,
        parent_schema_id=None,
        task=task,
        module_fields={producer: fields},
        consumption_rules=rules,
        validators={field.name: field.validation_rule or "" for field in fields},
        schema_token_budget=512,
        mutation_history=("human_template",),
        proposer_seed=seed,
        control_type=control_type,
    )
    return candidate.with_id_from_content(prefix=schema_id)


def make_hover_schema_candidate(
    module_names: tuple[str, ...] | None = None,
    seed: int = 0,
    schema_id: str = "hover_human",
) -> SchemaCandidate:
    producer, consumer = _module_pair(module_names)
    fields = (
        SchemaField(
            name="claim_atoms",
            type="array[object]",
            description=(
                "Atomic claim decomposition with text, entities, relation, and evidence need for each atom."
            ),
            required=True,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_items=8,
            validation_rule="each atom should include text, entities, relation, and needs_evidence_from",
            evidence_scope="train-trace-derived claim decomposition only",
            causal_hypothesis="Downstream verifier can check each subclaim independently.",
        ),
        SchemaField(
            name="hop_plan",
            type="array[object]",
            description=(
                "Ordered retrieval or reasoning hop plan containing query_intent, anchor_entity, "
                "and missing_evidence_reason."
            ),
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_items=4,
            validation_rule="hop_index values should be positive integers when present",
            evidence_scope="retrieval intent, not gold evidence",
            causal_hypothesis="Query intent preserves the missing evidence bottleneck for later modules.",
        ),
        SchemaField(
            name="evidence_table",
            type="array[object]",
            description=(
                "Evidence rows with title, sentence, supported atom IDs, contradicted atom IDs, and confidence."
            ),
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_items=12,
            validation_rule="confidence values should be between 0 and 1 when present",
            evidence_scope="retrieved evidence only",
            causal_hypothesis="Structured evidence prevents unsupported final verdicts.",
        ),
        SchemaField(
            name="evidence_conflict",
            type="object",
            description="Whether retrieved evidence conflicts and a concise conflict description.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            validation_rule="object may contain has_conflict and conflict_description",
            evidence_scope="conflict state from available context",
            causal_hypothesis="Explicit conflict state reduces false supported verdicts.",
        ),
        SchemaField(
            name="final_verdict_preconditions",
            type="array[string]",
            description="Checklist required before emitting the final supported/not-supported verdict.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_items=6,
            validation_rule="short checklist items only",
            causal_hypothesis="Precondition checklist forces evidence completeness checks.",
        ),
    )
    return _candidate(
        schema_id=schema_id,
        task="HoVer",
        producer=producer,
        consumer=consumer,
        fields=fields,
        seed=seed,
    )


def make_hotpotqa_schema_candidate(
    module_names: tuple[str, ...] | None = None,
    seed: int = 0,
    schema_id: str = "hotpotqa_human",
) -> SchemaCandidate:
    producer, consumer = _module_pair(module_names)
    fields = (
        SchemaField(
            name="question_type",
            type="enum",
            description="Question class: bridge, comparison, yes_no, or other.",
            required=True,
            producer_module=producer,
            consumer_modules=(consumer,),
            enum_values=("bridge", "comparison", "yes_no", "other"),
            validation_rule="must be one of bridge, comparison, yes_no, other",
            causal_hypothesis="Question type changes downstream evidence aggregation strategy.",
        ),
        SchemaField(
            name="answer_type_expected",
            type="enum",
            description="Expected answer type such as person, place, date, number, organization, boolean, or other.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            enum_values=("person", "place", "date", "number", "organization", "boolean", "other"),
            validation_rule="must be a supported answer type enum value",
            causal_hypothesis="Answer type constrains final answer extraction.",
        ),
        SchemaField(
            name="bridge_entity",
            type="string",
            description="Most likely bridge entity needed to connect the first and second evidence hop.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_tokens=24,
            validation_rule="short entity mention or empty string",
            evidence_scope="candidate entity from question and retrieved context",
            causal_hypothesis="Bridge entity passes the missing multi-hop variable downstream.",
        ),
        SchemaField(
            name="next_query_intent",
            type="string",
            description="The missing fact or relation the downstream module should retrieve or reason about next.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_tokens=48,
            validation_rule="concise retrieval intent, not a full chain of thought",
            evidence_scope="intent only, not gold evidence",
            causal_hypothesis="Query intent improves evidence flow without extra retrieval calls.",
        ),
        SchemaField(
            name="missing_evidence_reason",
            type="string",
            description="Why the current context is insufficient to infer the answer.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_tokens=64,
            validation_rule="one concise reason",
            causal_hypothesis="Missing-evidence reason prevents premature final answers.",
        ),
        SchemaField(
            name="candidate_answer_constraints",
            type="array[string]",
            description="Constraints the final answer must satisfy.",
            required=False,
            producer_module=producer,
            consumer_modules=(consumer,),
            max_items=6,
            validation_rule="short constraints only",
            causal_hypothesis="Constraints remove answers inconsistent with earlier evidence.",
        ),
    )
    return _candidate(
        schema_id=schema_id,
        task="HotpotQA",
        producer=producer,
        consumer=consumer,
        fields=fields,
        seed=seed,
    )


def make_human_minimal_schemas(
    task: str,
    module_names: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[SchemaCandidate]:
    normalized = task.lower()
    if normalized in {"hover", "hovertask"}:
        full = make_hover_schema_candidate(module_names=module_names, seed=seed)
        producer, consumer = _module_pair(module_names)
        minimal = _candidate(
            schema_id="hover_human_minimal",
            task="HoVer",
            producer=producer,
            consumer=consumer,
            fields=full.module_fields[producer][:2],
            seed=seed,
        )
        return [minimal, full]
    if normalized in {"hotpotqa", "hotpot", "toy_multihop"}:
        full = make_hotpotqa_schema_candidate(module_names=module_names, seed=seed)
        producer, consumer = _module_pair(module_names)
        minimal_names = {"question_type", "bridge_entity", "next_query_intent"}
        minimal_fields = tuple(
            field for field in full.module_fields[producer] if field.name in minimal_names
        )
        minimal = _candidate(
            schema_id="hotpotqa_human_minimal",
            task="HotpotQA",
            producer=producer,
            consumer=consumer,
            fields=minimal_fields,
            seed=seed,
        )
        return [minimal, full]
    raise ValueError(f"unsupported task for human templates: {task}")


def make_validator_only_schema(
    *,
    task: str,
    module_names: tuple[str, ...],
    seed: int = 0,
    schema_token_budget: int = 512,
) -> SchemaCandidate:
    return SchemaCandidate(
        schema_id=f"validator_only_{task.lower()}",
        parent_schema_id=None,
        task=task,
        module_fields={module_name: () for module_name in module_names},
        consumption_rules=(),
        validators={},
        schema_token_budget=schema_token_budget,
        mutation_history=("validator_only_original_schema",),
        proposer_seed=seed,
        control_type="validator_only",
    ).with_id_from_content(prefix="validator_only")
