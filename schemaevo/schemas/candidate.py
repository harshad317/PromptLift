from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import re
from typing import Any, Literal, Optional, TypeAlias

FieldType: TypeAlias = Literal[
    "string",
    "boolean",
    "number",
    "integer",
    "enum",
    "array[string]",
    "array[object]",
    "object",
]

FIELD_TYPES: tuple[str, ...] = (
    "string",
    "boolean",
    "number",
    "integer",
    "enum",
    "array[string]",
    "array[object]",
    "object",
)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")


def _require_snake_case(name: str) -> None:
    if not _SNAKE_CASE.match(name):
        raise ValueError(f"field name must be snake_case: {name!r}")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, length: int = 16) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()[:length]


def approximate_token_count(text: str) -> int:
    """Cheap deterministic proxy for token-budget checks.

    The real experiment should replace this with provider tokenizer accounting.
    This proxy is intentionally conservative enough for schema pool pruning.
    """

    if not text:
        return 0
    if os.environ.get("SCHEMAEVO_USE_TIKTOKEN") == "1":
        encoding = _load_tiktoken_encoding()
        if encoding is not None:
            return len(encoding.encode(text))
    return max(1, (len(text) + 3) // 4)


_TIKTOKEN_ENCODING: Any | None = None
_TIKTOKEN_LOAD_ATTEMPTED = False


def _load_tiktoken_encoding() -> Any | None:
    global _TIKTOKEN_ENCODING, _TIKTOKEN_LOAD_ATTEMPTED
    if _TIKTOKEN_LOAD_ATTEMPTED:
        return _TIKTOKEN_ENCODING
    _TIKTOKEN_LOAD_ATTEMPTED = True
    try:
        import tiktoken

        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


@dataclass(frozen=True)
class SchemaField:
    name: str
    type: FieldType
    description: str
    required: bool
    producer_module: str
    consumer_modules: tuple[str, ...]
    enum_values: Optional[tuple[str, ...]] = None
    max_items: Optional[int] = None
    max_tokens: Optional[int] = None
    validation_rule: Optional[str] = None
    evidence_scope: Optional[str] = None
    causal_hypothesis: Optional[str] = None

    def __post_init__(self) -> None:
        _require_snake_case(self.name)
        if self.type not in FIELD_TYPES:
            raise ValueError(f"unsupported field type: {self.type!r}")
        if not self.description.strip():
            raise ValueError(f"field {self.name!r} must have a description")
        if not self.producer_module:
            raise ValueError(f"field {self.name!r} must have a producer module")
        if not self.consumer_modules:
            raise ValueError(f"field {self.name!r} must have at least one consumer module")
        if self.type == "enum" and not self.enum_values:
            raise ValueError(f"enum field {self.name!r} requires enum_values")
        if self.type != "enum" and self.enum_values:
            raise ValueError(f"non-enum field {self.name!r} cannot have enum_values")
        if self.max_items is not None and self.max_items <= 0:
            raise ValueError(f"field {self.name!r} max_items must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"field {self.name!r} max_tokens must be positive")

    @property
    def token_cost(self) -> int:
        text = " ".join(
            part
            for part in (
                self.name,
                self.type,
                self.description,
                " ".join(self.enum_values or ()),
                self.validation_rule or "",
                self.evidence_scope or "",
                self.causal_hypothesis or "",
            )
            if part
        )
        return approximate_token_count(text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "producer_module": self.producer_module,
            "consumer_modules": list(self.consumer_modules),
            "enum_values": list(self.enum_values) if self.enum_values else None,
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
            "validation_rule": self.validation_rule,
            "evidence_scope": self.evidence_scope,
            "causal_hypothesis": self.causal_hypothesis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaField":
        return cls(
            name=data["name"],
            type=data["type"],
            description=data["description"],
            required=bool(data["required"]),
            producer_module=data["producer_module"],
            consumer_modules=tuple(data["consumer_modules"]),
            enum_values=tuple(data["enum_values"]) if data.get("enum_values") else None,
            max_items=data.get("max_items"),
            max_tokens=data.get("max_tokens"),
            validation_rule=data.get("validation_rule"),
            evidence_scope=data.get("evidence_scope"),
            causal_hypothesis=data.get("causal_hypothesis"),
        )


@dataclass(frozen=True)
class ConsumptionRule:
    consumer_module: str
    field_name: str
    instruction: str
    required_behavior: str
    fallback_if_missing: str

    def __post_init__(self) -> None:
        _require_snake_case(self.field_name)
        if not self.consumer_module:
            raise ValueError("consumption rule requires consumer_module")
        if not self.instruction.strip():
            raise ValueError("consumption rule requires instruction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_module": self.consumer_module,
            "field_name": self.field_name,
            "instruction": self.instruction,
            "required_behavior": self.required_behavior,
            "fallback_if_missing": self.fallback_if_missing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsumptionRule":
        return cls(
            consumer_module=data["consumer_module"],
            field_name=data["field_name"],
            instruction=data["instruction"],
            required_behavior=data["required_behavior"],
            fallback_if_missing=data["fallback_if_missing"],
        )


@dataclass(frozen=True)
class SchemaCandidate:
    schema_id: str
    parent_schema_id: Optional[str]
    task: str
    module_fields: dict[str, tuple[SchemaField, ...]]
    consumption_rules: tuple[ConsumptionRule, ...]
    validators: dict[str, str]
    schema_token_budget: int
    mutation_history: tuple[str, ...]
    proposer_seed: int
    control_type: str = "schemaevo"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id is required")
        if not self.task:
            raise ValueError("task is required")
        seen: set[tuple[str, str]] = set()
        for module_name, fields in self.module_fields.items():
            if not module_name:
                raise ValueError("module field map contains empty module name")
            for schema_field in fields:
                if schema_field.producer_module != module_name:
                    raise ValueError(
                        f"field {schema_field.name!r} producer {schema_field.producer_module!r} "
                        f"does not match module map {module_name!r}"
                    )
                key = (module_name, schema_field.name)
                if key in seen:
                    raise ValueError(f"duplicate field in module {module_name!r}: {schema_field.name}")
                seen.add(key)
        field_names = {field.name for field in self.all_fields}
        for rule in self.consumption_rules:
            if rule.field_name not in field_names:
                raise ValueError(f"consumption rule references unknown field: {rule.field_name}")
        if self.schema_token_budget <= 0:
            raise ValueError("schema_token_budget must be positive")

    @property
    def all_fields(self) -> tuple[SchemaField, ...]:
        fields: list[SchemaField] = []
        for module_name in sorted(self.module_fields):
            fields.extend(self.module_fields[module_name])
        return tuple(fields)

    @property
    def evolved_field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.all_fields)

    @property
    def token_cost(self) -> int:
        field_cost = sum(field.token_cost for field in self.all_fields)
        rule_cost = sum(
            approximate_token_count(
                " ".join(
                    (
                        rule.consumer_module,
                        rule.field_name,
                        rule.instruction,
                        rule.required_behavior,
                        rule.fallback_if_missing,
                    )
                )
            )
            for rule in self.consumption_rules
        )
        validator_cost = approximate_token_count(_stable_json(self.validators))
        return field_cost + rule_cost + validator_cost

    def with_id_from_content(self, prefix: str = "schema") -> "SchemaCandidate":
        data = self.to_dict(include_schema_id=False)
        schema_id = f"{prefix}_{stable_hash(data)}"
        return self.replace(schema_id=schema_id)

    def replace(self, **updates: Any) -> "SchemaCandidate":
        data = {
            "schema_id": self.schema_id,
            "parent_schema_id": self.parent_schema_id,
            "task": self.task,
            "module_fields": self.module_fields,
            "consumption_rules": self.consumption_rules,
            "validators": self.validators,
            "schema_token_budget": self.schema_token_budget,
            "mutation_history": self.mutation_history,
            "proposer_seed": self.proposer_seed,
            "control_type": self.control_type,
            "metadata": self.metadata,
        }
        data.update(updates)
        return SchemaCandidate(**data)

    def to_dict(self, include_schema_id: bool = True) -> dict[str, Any]:
        data = {
            "parent_schema_id": self.parent_schema_id,
            "task": self.task,
            "module_fields": {
                module_name: [field.to_dict() for field in fields]
                for module_name, fields in sorted(self.module_fields.items())
            },
            "consumption_rules": [rule.to_dict() for rule in self.consumption_rules],
            "validators": dict(sorted(self.validators.items())),
            "schema_token_budget": self.schema_token_budget,
            "mutation_history": list(self.mutation_history),
            "proposer_seed": self.proposer_seed,
            "control_type": self.control_type,
            "metadata": self.metadata,
        }
        if include_schema_id:
            data = {"schema_id": self.schema_id, **data}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaCandidate":
        return cls(
            schema_id=data["schema_id"],
            parent_schema_id=data.get("parent_schema_id"),
            task=data["task"],
            module_fields={
                module_name: tuple(SchemaField.from_dict(item) for item in fields)
                for module_name, fields in data["module_fields"].items()
            },
            consumption_rules=tuple(
                ConsumptionRule.from_dict(item) for item in data.get("consumption_rules", [])
            ),
            validators=dict(data.get("validators", {})),
            schema_token_budget=int(data["schema_token_budget"]),
            mutation_history=tuple(data.get("mutation_history", ())),
            proposer_seed=int(data.get("proposer_seed", 0)),
            control_type=data.get("control_type", "schemaevo"),
            metadata=dict(data.get("metadata", {})),
        )


def make_schema_id(candidate: SchemaCandidate, prefix: str = "schema") -> str:
    return f"{prefix}_{stable_hash(candidate.to_dict(include_schema_id=False))}"
