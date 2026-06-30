from __future__ import annotations

import random

from schemaevo.schemas.candidate import ConsumptionRule, FieldType, SchemaCandidate, SchemaField

_RANDOM_WORDS = (
    "format_slot",
    "unused_flag",
    "aux_marker",
    "latent_bucket",
    "tracking_note",
    "parse_hint",
    "opaque_key",
    "control_value",
    "scratch_label",
    "routing_stub",
    "neutral_token",
    "debug_count",
)

_TYPES: tuple[FieldType, ...] = (
    "string",
    "boolean",
    "number",
    "integer",
    "enum",
    "array[string]",
    "array[object]",
    "object",
)


def make_random_schema_controls(
    *,
    task: str,
    module_names: tuple[str, ...],
    n: int,
    seed: int,
    max_fields: int = 6,
    schema_token_budget: int = 512,
) -> list[SchemaCandidate]:
    if len(module_names) < 2:
        raise ValueError("random controls require at least two modules")
    rng = random.Random(seed)
    producer = module_names[0]
    consumer = module_names[-1]
    controls: list[SchemaCandidate] = []
    for index in range(n):
        field_count = rng.randint(1, max_fields)
        fields: list[SchemaField] = []
        used_names: set[str] = set()
        for field_index in range(field_count):
            base = rng.choice(_RANDOM_WORDS)
            name = f"{base}_{index}_{field_index}"
            while name in used_names:
                name = f"{base}_{index}_{field_index}_{rng.randint(0, 999)}"
            used_names.add(name)
            field_type = rng.choice(_TYPES)
            enum_values = ("alpha", "beta", "gamma") if field_type == "enum" else None
            fields.append(
                SchemaField(
                    name=name,
                    type=field_type,
                    description="Semantically neutral control field with matched schema capacity.",
                    required=False,
                    producer_module=producer,
                    consumer_modules=(consumer,),
                    enum_values=enum_values,
                    max_items=rng.randint(2, 6) if field_type.startswith("array") else None,
                    max_tokens=rng.randint(8, 48) if field_type == "string" else None,
                    validation_rule="control field; no task semantics",
                    causal_hypothesis="Should not improve evidence flow if semantics matter.",
                )
            )
        rules = tuple(
            ConsumptionRule(
                consumer_module=consumer,
                field_name=field.name,
                instruction=f"Ignore `{field.name}` unless it is useful under the original prompt.",
                required_behavior="Do not infer new evidence from this neutral control field.",
                fallback_if_missing="Use original prompt behavior.",
            )
            for field in fields
        )
        candidate = SchemaCandidate(
            schema_id=f"random_control_{index}",
            parent_schema_id=None,
            task=task,
            module_fields={producer: tuple(fields)},
            consumption_rules=rules,
            validators={field.name: field.validation_rule or "" for field in fields},
            schema_token_budget=schema_token_budget,
            mutation_history=("random_schema_control",),
            proposer_seed=seed,
            control_type="random",
            metadata={"control_index": index},
        ).with_id_from_content(prefix="random")
        controls.append(candidate)
    return controls
