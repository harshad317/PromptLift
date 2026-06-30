from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any

from schemaevo.programs.base import (
    LMProgram,
    ModuleExecutionContext,
    ModuleSignature,
    ModuleSpec,
    ProgramExample,
)


@dataclass(frozen=True)
class OpenAIModuleConfig:
    name: str
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    prompt: str
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 1024
    temperature: float | None = None
    output_field_types: dict[str, str] = field(default_factory=dict)
    llm_calls: int = 1
    retriever_calls: int = 0
    demo_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def openai_modules_to_lm_program(
    *,
    task: str,
    modules: tuple[OpenAIModuleConfig, ...],
    final_output_module: str,
    retriever_top_k: int = 0,
    client: Any | None = None,
) -> LMProgram:
    """Build an `LMProgram` whose module runners call the OpenAI Responses API."""

    specs = tuple(_module_spec_from_config(config, client=client) for config in modules)
    return LMProgram(
        task=task,
        modules=specs,
        retriever_top_k=retriever_top_k,
        final_output_module=final_output_module,
        metadata={"adapter": "openai_responses"},
    )


def _module_spec_from_config(config: OpenAIModuleConfig, *, client: Any | None) -> ModuleSpec:
    metadata = dict(config.metadata)
    metadata["demo_ids"] = tuple(config.demo_ids)
    metadata["adapter"] = "openai_responses"
    metadata["openai_output_field_types"] = dict(config.output_field_types)
    metadata["temperature"] = config.temperature
    return ModuleSpec(
        name=config.name,
        signature=ModuleSignature(
            input_fields=config.input_fields,
            output_fields=config.output_fields,
        ),
        prompt=config.prompt,
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        runner=_make_runner(client),
        llm_calls=config.llm_calls,
        retriever_calls=config.retriever_calls,
        metadata=metadata,
    )


def _make_runner(client: Any | None):
    api_client = client

    def run(
        state: dict[str, Any],
        spec: ModuleSpec,
        example: ProgramExample,
        context: ModuleExecutionContext,
    ) -> dict[str, Any]:
        nonlocal api_client
        if api_client is None:
            api_client = _make_openai_client()
        module_input = _build_module_input(state=state, spec=spec, example=example)
        create_kwargs: dict[str, Any] = {
            "model": spec.model,
            "input": [
                {
                    "role": "system",
                    "content": spec.prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "module_name": spec.name,
                            "task": context.task,
                            "example_id": example.example_id,
                            "inputs": module_input,
                            "schema_fields": state.get("schema_fields", {}),
                            "prior_module_outputs": state.get("module_outputs", {}),
                            "required_output_fields": list(spec.signature.output_fields),
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                        default=str,
                    ),
                },
            ],
            "max_output_tokens": spec.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{_sanitize_schema_name(spec.name)}_output",
                    "strict": True,
                    "schema": _module_output_json_schema(spec),
                }
            },
        }
        temperature = spec.metadata.get("temperature")
        if temperature is not None:
            create_kwargs["temperature"] = float(temperature)
        response = api_client.responses.create(**create_kwargs)
        return _extract_response_json(response)

    return run


def _make_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI module runners")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the `openai` package to use OpenAI module runners") from exc
    return OpenAI()


def _build_module_input(
    *,
    state: dict[str, Any],
    spec: ModuleSpec,
    example: ProgramExample,
) -> dict[str, Any]:
    available: dict[str, Any] = {}
    available.update(state.get("inputs", {}))
    available.update(state.get("schema_fields", {}))
    for output in state.get("module_outputs", {}).values():
        if isinstance(output, dict):
            available.update(output)
    available["example_metadata"] = example.metadata
    return {field: available.get(field) for field in spec.signature.input_fields}


def _module_output_json_schema(spec: ModuleSpec) -> dict[str, Any]:
    schema_fields = {
        item["name"]: item
        for item in spec.metadata.get("schemaevo_output_fields", [])
        if isinstance(item, dict) and "name" in item
    }
    configured_types = spec.metadata.get("openai_output_field_types", {})
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name in spec.signature.output_fields:
        schema_field = schema_fields.get(field_name)
        field_type = (
            schema_field.get("type")
            if schema_field
            else configured_types.get(field_name, "string")
        )
        required_field = bool(schema_field.get("required", True)) if schema_field else True
        field_schema = _json_schema_for_field_type(str(field_type), schema_field)
        if not required_field:
            field_schema = _nullable_schema(field_schema)
        properties[field_name] = field_schema
        required.append(field_name)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _json_schema_for_field_type(field_type: str, schema_field: dict[str, Any] | None) -> dict[str, Any]:
    if field_type == "boolean":
        return {"type": "boolean"}
    if field_type == "number":
        return {"type": "number"}
    if field_type == "integer":
        return {"type": "integer"}
    if field_type == "array[string]":
        return {"type": "array", "items": {"type": "string"}}
    if field_type == "array[object]":
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}
    if field_type == "object":
        return {"type": "object", "additionalProperties": True}
    if field_type == "enum" and schema_field and schema_field.get("enum_values"):
        return {"type": "string", "enum": list(schema_field["enum_values"])}
    return {"type": "string"}


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
    return schema


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
        raise ValueError("OpenAI module response did not contain output_text")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI module response must be a JSON object")
    return parsed


def _sanitize_schema_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    return cleaned or "module"
