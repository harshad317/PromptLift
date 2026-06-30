from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from schemaevo.schemas.candidate import FIELD_TYPES, FieldType, SchemaCandidate


class MutationOp(Enum):
    ADD_FIELD = "add_field"
    DROP_FIELD = "drop_field"
    RENAME_FIELD = "rename_field"
    CHANGE_REQUIRED_OPTIONAL = "change_required_optional"
    CHANGE_TYPE = "change_type"
    ADD_ENUM_VALUE = "add_enum_value"
    DROP_ENUM_VALUE = "drop_enum_value"
    SPLIT_FIELD = "split_field"
    MERGE_FIELDS = "merge_fields"
    MOVE_FIELD_TO_EARLIER_MODULE = "move_field_to_earlier_module"
    MOVE_FIELD_TO_LATER_MODULE = "move_field_to_later_module"
    ADD_DOWNSTREAM_CONSUMPTION_RULE = "add_downstream_consumption_rule"
    DROP_DOWNSTREAM_CONSUMPTION_RULE = "drop_downstream_consumption_rule"
    TIGHTEN_VALIDATOR = "tighten_validator"
    RELAX_VALIDATOR = "relax_validator"


FORBIDDEN_MUTATIONS: frozenset[str] = frozenset(
    {
        "ADD_LLM_CALL",
        "ADD_RETRIEVAL_CALL",
        "INCREASE_RETRIEVAL_TOP_K",
        "ADD_SELF_CONSISTENCY",
        "ADD_EXTERNAL_TOOL",
        "ADD_TEST_LABEL_OR_GOLD_EVIDENCE",
        "CHANGE_FINAL_METRIC",
        "CHANGE_TARGET_MODEL",
        "CHANGE_DATA_SPLIT",
    }
)


@dataclass(frozen=True)
class StaticCheckResult:
    passed: bool
    errors: tuple[str, ...] = ()

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError("; ".join(self.errors))


@dataclass(frozen=True)
class SchemaGrammar:
    allowed_modules: tuple[str, ...]
    allowed_types: tuple[FieldType, ...] = FIELD_TYPES  # type: ignore[assignment]
    max_fields_per_module: int = 12
    max_total_fields: int = 32
    max_schema_tokens: int = 512
    max_name_chars: int = 48
    allow_required_fields: bool = True

    def check_candidate(self, candidate: SchemaCandidate) -> StaticCheckResult:
        errors: list[str] = []
        module_set = set(self.allowed_modules)
        total_fields = len(candidate.all_fields)
        if total_fields > self.max_total_fields:
            errors.append(f"too many fields: {total_fields} > {self.max_total_fields}")
        if candidate.token_cost > min(candidate.schema_token_budget, self.max_schema_tokens):
            errors.append(
                f"schema token cost {candidate.token_cost} exceeds budget "
                f"{min(candidate.schema_token_budget, self.max_schema_tokens)}"
            )
        for module_name, fields in candidate.module_fields.items():
            if module_name not in module_set:
                errors.append(f"unknown producer module: {module_name}")
            if len(fields) > self.max_fields_per_module:
                errors.append(
                    f"module {module_name} has {len(fields)} fields; max is {self.max_fields_per_module}"
                )
            for field in fields:
                if len(field.name) > self.max_name_chars:
                    errors.append(f"field name too long: {field.name}")
                if field.type not in self.allowed_types:
                    errors.append(f"field {field.name} uses disallowed type {field.type}")
                if field.required and not self.allow_required_fields:
                    errors.append(f"required fields are disabled: {field.name}")
                unknown_consumers = set(field.consumer_modules) - module_set
                if unknown_consumers:
                    errors.append(
                        f"field {field.name} has unknown consumers: {sorted(unknown_consumers)}"
                    )
        for rule in candidate.consumption_rules:
            if rule.consumer_module not in module_set:
                errors.append(f"consumption rule has unknown consumer: {rule.consumer_module}")
        return StaticCheckResult(passed=not errors, errors=tuple(errors))


def assert_legal_mutation(op: MutationOp | str) -> MutationOp:
    if isinstance(op, str) and op in FORBIDDEN_MUTATIONS:
        raise ValueError(f"forbidden mutation requested: {op}")
    if isinstance(op, MutationOp):
        return op
    try:
        return MutationOp(op)
    except ValueError as exc:
        raise ValueError(f"unknown mutation op: {op}") from exc
