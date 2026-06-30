from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from schemaevo.schemas.candidate import FieldType, SchemaCandidate, SchemaField

ValidationPolicy = Literal["fail", "deterministic_coercion"]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    parsed: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()
    deterministic_repair_applied: bool = False


class SchemaValidator:
    def __init__(self, schema_candidate: SchemaCandidate):
        self.schema_candidate = schema_candidate
        self.pydantic_models = build_pydantic_models(schema_candidate)

    def validate_module_output(
        self,
        module_name: str,
        raw_output: Any,
        policy: ValidationPolicy,
    ) -> ValidationResult:
        if module_name not in self.pydantic_models:
            return ValidationResult(valid=True, parsed=_as_mapping(raw_output))
        model = self.pydantic_models[module_name]
        fields = self.schema_candidate.module_fields[module_name]
        try:
            parsed = model.model_validate(raw_output)
            parsed_dict = parsed.model_dump()
            constraint_errors = validate_executable_constraints(
                parsed_dict,
                fields,
                self.schema_candidate.validators,
            )
            if constraint_errors:
                return ValidationResult(valid=False, errors=tuple(constraint_errors))
            return ValidationResult(valid=True, parsed=parsed_dict)
        except Exception as first_error:
            if policy != "deterministic_coercion":
                return ValidationResult(valid=False, errors=(str(first_error),))
            repaired = deterministic_type_coercion(raw_output, fields)
            try:
                parsed = model.model_validate(repaired)
                parsed_dict = parsed.model_dump()
                constraint_errors = validate_executable_constraints(
                    parsed_dict,
                    fields,
                    self.schema_candidate.validators,
                )
                if constraint_errors:
                    return ValidationResult(
                        valid=False,
                        errors=(str(first_error), *constraint_errors),
                    )
                return ValidationResult(
                    valid=True,
                    parsed=parsed_dict,
                    deterministic_repair_applied=True,
                )
            except Exception as second_error:
                return ValidationResult(valid=False, errors=(str(first_error), str(second_error)))


def build_pydantic_models(schema_candidate: SchemaCandidate) -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for module_name, fields in schema_candidate.module_fields.items():
        model_fields: dict[str, tuple[Any, Any]] = {}
        for field_spec in fields:
            py_type = _python_type(field_spec)
            default: Any
            if field_spec.required:
                default = Field(..., description=field_spec.description)
            else:
                py_type = Optional[py_type]
                default = Field(None, description=field_spec.description)
            model_fields[field_spec.name] = (py_type, default)
        models[module_name] = create_model(
            f"{module_name.title().replace('_', '')}SchemaEvoModel",
            __config__=ConfigDict(extra="allow"),
            **model_fields,
        )
    return models


def deterministic_type_coercion(raw_output: Any, fields: tuple[SchemaField, ...]) -> dict[str, Any]:
    data = _as_mapping(raw_output)
    repaired = dict(data)
    for field_spec in fields:
        if field_spec.name not in repaired or repaired[field_spec.name] is None:
            continue
        repaired[field_spec.name] = _coerce_value(repaired[field_spec.name], field_spec)
    return repaired


def validate_executable_constraints(
    parsed_output: dict[str, Any],
    fields: tuple[SchemaField, ...],
    validators: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for field_spec in fields:
        value = parsed_output.get(field_spec.name)
        if value is None:
            continue
        constraints = _parse_constraints(
            validators.get(field_spec.name) or field_spec.validation_rule or ""
        )
        if field_spec.max_tokens is not None:
            constraints.setdefault("max_tokens", field_spec.max_tokens)
        if field_spec.max_items is not None:
            constraints.setdefault("max_items", field_spec.max_items)
        errors.extend(_check_constraints(field_spec.name, value, constraints))
        errors.extend(_check_known_object_shape(field_spec, value))
    return errors


def _parse_constraints(rule: str) -> dict[str, Any]:
    rule = rule.strip()
    if not rule:
        return {}
    if rule.startswith("{"):
        try:
            parsed = json.loads(rule)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    constraints: dict[str, Any] = {}
    for part in re.split(r"[;,]", rule):
        item = part.strip()
        if not item:
            continue
        if item == "non_empty":
            constraints["non_empty"] = True
            continue
        if item.startswith("regex=") or item.startswith("regex:"):
            constraints["regex"] = item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1]
            continue
        if item.startswith("min=") or item.startswith("min:"):
            constraints["min"] = float(item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1])
            continue
        if item.startswith("max=") or item.startswith("max:"):
            constraints["max"] = float(item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1])
            continue
        if item.startswith("max_tokens=") or item.startswith("max_tokens:"):
            constraints["max_tokens"] = int(
                item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1]
            )
            continue
        if item.startswith("max_items=") or item.startswith("max_items:"):
            constraints["max_items"] = int(
                item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1]
            )
            continue
        if item.startswith("one_of=") or item.startswith("one_of:"):
            raw = item.split("=", 1)[1] if "=" in item else item.split(":", 1)[1]
            constraints["one_of"] = [value.strip() for value in raw.split("|") if value.strip()]
    return constraints


def _check_constraints(field_name: str, value: Any, constraints: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if constraints.get("non_empty"):
        if value == "" or value == [] or value == {}:
            errors.append(f"{field_name}: must be non-empty")
    if "regex" in constraints and isinstance(value, str):
        if not re.search(str(constraints["regex"]), value):
            errors.append(f"{field_name}: does not match regex {constraints['regex']!r}")
    if "min" in constraints and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < float(constraints["min"]):
            errors.append(f"{field_name}: {value} < min {constraints['min']}")
    if "max" in constraints and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > float(constraints["max"]):
            errors.append(f"{field_name}: {value} > max {constraints['max']}")
    if "max_tokens" in constraints and isinstance(value, str):
        if max(1, (len(value) + 3) // 4) > int(constraints["max_tokens"]):
            errors.append(f"{field_name}: exceeds max_tokens {constraints['max_tokens']}")
    if "max_items" in constraints and isinstance(value, list):
        if len(value) > int(constraints["max_items"]):
            errors.append(f"{field_name}: has {len(value)} items > max_items {constraints['max_items']}")
    if "one_of" in constraints and str(value) not in set(map(str, constraints["one_of"])):
        errors.append(f"{field_name}: {value!r} not in one_of {constraints['one_of']}")
    return errors


def _check_known_object_shape(field_spec: SchemaField, value: Any) -> list[str]:
    shape = _known_object_shape(field_spec.name)
    if not shape:
        return []
    if field_spec.type == "array[object]":
        if not isinstance(value, list):
            return [f"{field_spec.name}: must be an array of objects"]
        errors: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{field_spec.name}[{index}]: must be an object")
                continue
            errors.extend(
                _check_object_properties(
                    path=f"{field_spec.name}[{index}]",
                    value=item,
                    shape=shape,
                )
            )
        return errors
    if field_spec.type == "object":
        if not isinstance(value, dict):
            return [f"{field_spec.name}: must be an object"]
        return _check_object_properties(path=field_spec.name, value=value, shape=shape)
    return []


def _check_object_properties(
    *,
    path: str,
    value: dict[str, Any],
    shape: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for key, expected_type in shape.items():
        if key not in value:
            errors.append(f"{path}.{key}: missing required property")
            continue
        if not _matches_shape_type(value[key], expected_type):
            errors.append(f"{path}.{key}: expected {expected_type}")
    return errors


def _matches_shape_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "array[string]":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if expected_type == "array[integer]":
        return isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    return True


def _known_object_shape(field_name: str) -> dict[str, str]:
    if field_name == "claim_atoms":
        return {
            "text": "string",
            "entities": "array[string]",
            "relation": "string",
            "needs_evidence_from": "string",
        }
    if field_name == "hop_plan":
        return {
            "hop_index": "integer",
            "query_intent": "string",
            "anchor_entity": "string",
            "missing_evidence_reason": "string",
        }
    if field_name == "evidence_table":
        return {
            "title": "string",
            "sentence": "string",
            "supports_atom_ids": "array[integer]",
            "contradicts_atom_ids": "array[integer]",
            "confidence": "number",
        }
    if field_name == "evidence_conflict":
        return {
            "has_conflict": "boolean",
            "conflict_description": "string",
        }
    if field_name == "bridge_entities":
        return {
            "surface": "string",
            "role": "string",
            "confidence": "number",
        }
    if field_name == "evidence_needs":
        return {
            "needed_fact": "string",
            "source_hint": "string",
            "resolved": "boolean",
        }
    return {}


def _as_mapping(raw_output: Any) -> dict[str, Any]:
    if raw_output is None:
        return {}
    if isinstance(raw_output, dict):
        return dict(raw_output)
    if isinstance(raw_output, BaseModel):
        return raw_output.model_dump()
    if isinstance(raw_output, str):
        stripped = raw_output.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("string module output is not a JSON object")
    raise ValueError(f"module output must be a mapping or JSON object string, got {type(raw_output).__name__}")


def _python_type(field_spec: SchemaField) -> Any:
    field_type: FieldType = field_spec.type
    if field_type == "string":
        return str
    if field_type == "boolean":
        return bool
    if field_type == "number":
        return float
    if field_type == "integer":
        return int
    if field_type == "array[string]":
        return list[str]
    if field_type == "array[object]":
        return list[dict[str, Any]]
    if field_type == "object":
        return dict[str, Any]
    if field_type == "enum":
        values = field_spec.enum_values or ()
        # Literal requires values as positional args at runtime.
        return Literal.__getitem__(values)  # type: ignore[attr-defined]
    valid = ", ".join(get_args(FieldType)) if hasattr(FieldType, "__args__") else "supported field type"
    raise ValueError(f"unsupported field type {field_type!r}; expected {valid}")


def _coerce_value(value: Any, field_spec: SchemaField) -> Any:
    if field_spec.type == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return value
    if field_spec.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return value
    if field_spec.type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value
    if field_spec.type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value
    if field_spec.type == "array[string]":
        if isinstance(value, list):
            return [str(item) for item in value if isinstance(item, (str, int, float, bool))]
        if isinstance(value, str):
            if value.strip().startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return value
    if field_spec.type == "array[object]":
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return value
    if field_spec.type == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return value
    if field_spec.type == "enum":
        if isinstance(value, str):
            for enum_value in field_spec.enum_values or ():
                if value == enum_value or value.strip().lower() == enum_value.lower():
                    return enum_value
        return value
    return value


def summarize_validation_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors())
    return str(error)
