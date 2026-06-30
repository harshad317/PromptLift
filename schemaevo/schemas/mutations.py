from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemaevo.schemas.candidate import ConsumptionRule, FieldType, SchemaCandidate, SchemaField
from schemaevo.schemas.grammar import MutationOp, assert_legal_mutation


@dataclass(frozen=True)
class Mutation:
    op: MutationOp
    module_name: str | None = None
    field_name: str | None = None
    payload: dict[str, Any] | None = None
    rationale: str = ""

    @classmethod
    def from_parts(
        cls,
        op: MutationOp | str,
        *,
        module_name: str | None = None,
        field_name: str | None = None,
        payload: dict[str, Any] | None = None,
        rationale: str = "",
    ) -> "Mutation":
        return cls(
            op=assert_legal_mutation(op),
            module_name=module_name,
            field_name=field_name,
            payload=payload or {},
            rationale=rationale,
        )


def apply_mutation(candidate: SchemaCandidate, mutation: Mutation) -> SchemaCandidate:
    payload = mutation.payload or {}
    fields_by_module = {module: list(fields) for module, fields in candidate.module_fields.items()}
    rules = list(candidate.consumption_rules)
    validators = dict(candidate.validators)
    op = mutation.op

    if op == MutationOp.ADD_FIELD:
        module_name = _required(mutation.module_name, "module_name")
        new_field = _field_from_payload(payload, module_name)
        fields_by_module.setdefault(module_name, []).append(new_field)
        for consumer in new_field.consumer_modules:
            rules.append(_default_rule(consumer, new_field.name, module_name))
        validators[new_field.name] = new_field.validation_rule or ""
    elif op == MutationOp.DROP_FIELD:
        module_name, field_name = _module_and_field(mutation)
        fields_by_module[module_name] = [
            field for field in fields_by_module.get(module_name, []) if field.name != field_name
        ]
        rules = [rule for rule in rules if rule.field_name != field_name]
        validators.pop(field_name, None)
    elif op == MutationOp.RENAME_FIELD:
        module_name, field_name = _module_and_field(mutation)
        new_name = payload["new_name"]
        fields_by_module[module_name] = [
            _replace_field(field, name=new_name) if field.name == field_name else field
            for field in fields_by_module.get(module_name, [])
        ]
        rules = [
            _replace_rule(rule, field_name=new_name) if rule.field_name == field_name else rule
            for rule in rules
        ]
        if field_name in validators:
            validators[new_name] = validators.pop(field_name)
    elif op == MutationOp.CHANGE_REQUIRED_OPTIONAL:
        module_name, field_name = _module_and_field(mutation)
        fields_by_module[module_name] = [
            _replace_field(field, required=not field.required) if field.name == field_name else field
            for field in fields_by_module.get(module_name, [])
        ]
    elif op == MutationOp.CHANGE_TYPE:
        module_name, field_name = _module_and_field(mutation)
        new_type: FieldType = payload["new_type"]
        enum_values = tuple(payload["enum_values"]) if new_type == "enum" else None
        fields_by_module[module_name] = [
            _replace_field(field, type=new_type, enum_values=enum_values)
            if field.name == field_name
            else field
            for field in fields_by_module.get(module_name, [])
        ]
    elif op == MutationOp.ADD_ENUM_VALUE:
        module_name, field_name = _module_and_field(mutation)
        value = payload["value"]
        fields_by_module[module_name] = [
            _replace_field(field, enum_values=tuple((*field.enum_values, value)))
            if field.name == field_name and field.enum_values and value not in field.enum_values
            else field
            for field in fields_by_module.get(module_name, [])
        ]
    elif op == MutationOp.DROP_ENUM_VALUE:
        module_name, field_name = _module_and_field(mutation)
        value = payload["value"]
        fields_by_module[module_name] = [
            _replace_field(
                field,
                enum_values=tuple(item for item in (field.enum_values or ()) if item != value),
            )
            if field.name == field_name and len(field.enum_values or ()) > 1
            else field
            for field in fields_by_module.get(module_name, [])
        ]
    elif op == MutationOp.SPLIT_FIELD:
        module_name, field_name = _module_and_field(mutation)
        source = _find_field(fields_by_module, module_name, field_name)
        new_names = tuple(payload["new_names"])
        replacement = [
            _replace_field(
                source,
                name=name,
                description=f"{source.description} Split component `{name}`.",
            )
            for name in new_names
        ]
        fields_by_module[module_name] = [
            field for field in fields_by_module.get(module_name, []) if field.name != field_name
        ] + replacement
        rules = [rule for rule in rules if rule.field_name != field_name]
        for new_field in replacement:
            validators[new_field.name] = new_field.validation_rule or ""
            for consumer in new_field.consumer_modules:
                rules.append(_default_rule(consumer, new_field.name, module_name))
        validators.pop(field_name, None)
    elif op == MutationOp.MERGE_FIELDS:
        module_name = _required(mutation.module_name, "module_name")
        names = set(payload["field_names"])
        source_fields = [field for field in fields_by_module.get(module_name, []) if field.name in names]
        if len(source_fields) < 2:
            raise ValueError("merge_fields requires at least two existing fields")
        merged = SchemaField(
            name=payload["new_name"],
            type=payload.get("new_type", "object"),
            description="Merged contract field: " + "; ".join(field.description for field in source_fields),
            required=any(field.required for field in source_fields),
            producer_module=module_name,
            consumer_modules=tuple(sorted({c for field in source_fields for c in field.consumer_modules})),
            validation_rule=payload.get("validation_rule", "merged field must preserve source semantics"),
            causal_hypothesis="Merged field should reduce fragmented downstream consumption.",
        )
        fields_by_module[module_name] = [
            field for field in fields_by_module.get(module_name, []) if field.name not in names
        ] + [merged]
        rules = [rule for rule in rules if rule.field_name not in names]
        for consumer in merged.consumer_modules:
            rules.append(_default_rule(consumer, merged.name, module_name))
        for name in names:
            validators.pop(name, None)
        validators[merged.name] = merged.validation_rule or ""
    elif op in {MutationOp.MOVE_FIELD_TO_EARLIER_MODULE, MutationOp.MOVE_FIELD_TO_LATER_MODULE}:
        module_name, field_name = _module_and_field(mutation)
        target_module = payload["target_module"]
        source = _find_field(fields_by_module, module_name, field_name)
        moved = _replace_field(source, producer_module=target_module)
        fields_by_module[module_name] = [
            field for field in fields_by_module.get(module_name, []) if field.name != field_name
        ]
        fields_by_module.setdefault(target_module, []).append(moved)
    elif op == MutationOp.ADD_DOWNSTREAM_CONSUMPTION_RULE:
        module_name = _required(mutation.module_name, "module_name")
        field_name = _required(mutation.field_name, "field_name")
        rules.append(
            ConsumptionRule(
                consumer_module=module_name,
                field_name=field_name,
                instruction=payload["instruction"],
                required_behavior=payload.get("required_behavior", "Consume the field when present."),
                fallback_if_missing=payload.get("fallback_if_missing", "Use original prompt behavior."),
            )
        )
    elif op == MutationOp.DROP_DOWNSTREAM_CONSUMPTION_RULE:
        module_name = _required(mutation.module_name, "module_name")
        field_name = _required(mutation.field_name, "field_name")
        rules = [
            rule
            for rule in rules
            if not (rule.consumer_module == module_name and rule.field_name == field_name)
        ]
    elif op == MutationOp.TIGHTEN_VALIDATOR:
        field_name = _required(mutation.field_name, "field_name")
        validators[field_name] = payload["validator"]
    elif op == MutationOp.RELAX_VALIDATOR:
        field_name = _required(mutation.field_name, "field_name")
        validators[field_name] = payload.get("validator", "")
    else:
        raise ValueError(f"unhandled mutation op: {op}")

    mutated = candidate.replace(
        parent_schema_id=candidate.schema_id,
        module_fields={module: tuple(fields) for module, fields in fields_by_module.items()},
        consumption_rules=tuple(rules),
        validators=validators,
        mutation_history=(
            *candidate.mutation_history,
            f"{op.value}:{mutation.module_name or ''}:{mutation.field_name or ''}",
        ),
    )
    return mutated.with_id_from_content(prefix="mut")


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _module_and_field(mutation: Mutation) -> tuple[str, str]:
    return _required(mutation.module_name, "module_name"), _required(mutation.field_name, "field_name")


def _field_from_payload(payload: dict[str, Any], module_name: str) -> SchemaField:
    return SchemaField(
        name=payload["name"],
        type=payload["type"],
        description=payload["description"],
        required=bool(payload.get("required", False)),
        producer_module=module_name,
        consumer_modules=tuple(payload["consumer_modules"]),
        enum_values=tuple(payload["enum_values"]) if payload.get("enum_values") else None,
        max_items=payload.get("max_items"),
        max_tokens=payload.get("max_tokens"),
        validation_rule=payload.get("validation_rule"),
        evidence_scope=payload.get("evidence_scope"),
        causal_hypothesis=payload.get("causal_hypothesis"),
    )


def _replace_field(field: SchemaField, **updates: Any) -> SchemaField:
    data = field.to_dict()
    data.update(updates)
    return SchemaField.from_dict(data)


def _replace_rule(rule: ConsumptionRule, **updates: Any) -> ConsumptionRule:
    data = rule.to_dict()
    data.update(updates)
    return ConsumptionRule.from_dict(data)


def _find_field(
    fields_by_module: dict[str, list[SchemaField]], module_name: str, field_name: str
) -> SchemaField:
    for field in fields_by_module.get(module_name, []):
        if field.name == field_name:
            return field
    raise ValueError(f"field not found: {module_name}.{field_name}")


def _default_rule(consumer: str, field_name: str, producer: str) -> ConsumptionRule:
    return ConsumptionRule(
        consumer_module=consumer,
        field_name=field_name,
        instruction=f"Use `{field_name}` emitted by `{producer}` when relevant.",
        required_behavior="Preserve fixed-call information flow without adding calls.",
        fallback_if_missing="Use original prompt behavior.",
    )
