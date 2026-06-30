from __future__ import annotations

from schemaevo.programs.base import LMProgram, ModuleSignature
from schemaevo.programs.call_graph import assert_same_call_graph, extract_call_graph
from schemaevo.schemas.candidate import ConsumptionRule, SchemaCandidate, SchemaField

CONTRACT_START = "\n\n[SCHEMAEVO_CONTRACT_START]\n"
CONTRACT_END = "[SCHEMAEVO_CONTRACT_END]\n"


def compile_schema_program(
    *,
    base_program: LMProgram,
    schema: SchemaCandidate,
    freeze_prompt_text: bool,
    allow_only_schema_contract_insert: bool,
) -> LMProgram:
    patched = base_program.clone()
    original_call_graph = extract_call_graph(base_program)
    patched.schema_candidate = schema
    for module in patched.modules:
        module_schema_fields = schema.module_fields.get(module.name, ())
        module.metadata["schemaevo_output_fields"] = [
            field.to_dict() for field in module_schema_fields
        ]
        module.metadata["schemaevo_consumption_rules"] = [
            rule.to_dict() for rule in schema.consumption_rules if rule.consumer_module == module.name
        ]
        module.signature = patch_signature(
            original_signature=module.signature,
            additional_output_fields=module_schema_fields,
        )
        contract_text = render_schema_contract(
            module_name=module.name,
            fields=module_schema_fields,
            validators=schema.validators,
        )
        consumption_text = render_consumption_rules(
            module_name=module.name,
            rules=schema.consumption_rules,
        )
        original_prompt = module.prompt
        if freeze_prompt_text:
            module.prompt = insert_schema_contract_only(
                original_prompt=module.prompt,
                contract_text=contract_text,
                consumption_text=consumption_text,
            )
        else:
            module.prompt = insert_schema_and_mutated_prompt(
                original_prompt=module.prompt,
                contract_text=contract_text,
                consumption_text=consumption_text,
                mutated_prompt=schema.metadata.get("mutated_prompt"),
            )
        if allow_only_schema_contract_insert and freeze_prompt_text:
            _assert_only_schema_block_added(original_prompt, module.prompt)
    assert extract_call_graph(patched) == original_call_graph
    assert_same_call_graph(patched, base_program)
    return patched


def patch_signature(
    *,
    original_signature: ModuleSignature,
    additional_output_fields: tuple[SchemaField, ...],
) -> ModuleSignature:
    return original_signature.with_additional_outputs(tuple(field.name for field in additional_output_fields))


def render_schema_contract(
    *,
    module_name: str,
    fields: tuple[SchemaField, ...],
    validators: dict[str, str],
) -> str:
    if not fields:
        return f"Module `{module_name}` keeps the original output contract. No extra fields are added."
    lines = [
        f"Module `{module_name}` must emit these additional typed fields in the same module call:",
    ]
    for field in fields:
        required = "required" if field.required else "optional"
        enum_text = f" enum={list(field.enum_values)}" if field.enum_values else ""
        max_items = f" max_items={field.max_items}" if field.max_items else ""
        max_tokens = f" max_tokens={field.max_tokens}" if field.max_tokens else ""
        validator = validators.get(field.name) or field.validation_rule or "local schema validator"
        lines.append(
            f"- {field.name}: type={field.type} {required}.{enum_text}{max_items}{max_tokens} "
            f"Description: {field.description} Validator: {validator}"
        )
    lines.append("Do not add LLM calls, retrieval calls, self-consistency, or external tools.")
    return "\n".join(lines)


def render_consumption_rules(
    *,
    module_name: str,
    rules: tuple[ConsumptionRule, ...],
) -> str:
    relevant = [rule for rule in rules if rule.consumer_module == module_name]
    if not relevant:
        return ""
    lines = [f"Module `{module_name}` must consume upstream schema fields as follows:"]
    for rule in relevant:
        lines.append(
            f"- {rule.field_name}: {rule.instruction} Required behavior: "
            f"{rule.required_behavior} Fallback: {rule.fallback_if_missing}"
        )
    lines.append("If a field is missing or invalid, do not repair with an extra LLM call in primary evaluation.")
    return "\n".join(lines)


def insert_schema_contract_only(
    *,
    original_prompt: str,
    contract_text: str,
    consumption_text: str,
) -> str:
    parts = [CONTRACT_START, contract_text]
    if consumption_text:
        parts.extend(["\n\n", consumption_text])
    parts.append("\n" + CONTRACT_END)
    return original_prompt.rstrip() + "".join(parts)


def insert_schema_and_mutated_prompt(
    *,
    original_prompt: str,
    contract_text: str,
    consumption_text: str,
    mutated_prompt: str | None,
) -> str:
    base = mutated_prompt if mutated_prompt is not None else original_prompt
    return insert_schema_contract_only(
        original_prompt=base,
        contract_text=contract_text,
        consumption_text=consumption_text,
    )


def _assert_only_schema_block_added(original_prompt: str, patched_prompt: str) -> None:
    expected_prefix = original_prompt.rstrip()
    if not patched_prompt.startswith(expected_prefix):
        raise AssertionError("frozen prompt text changed before schema-contract insertion")
    suffix = patched_prompt[len(expected_prefix) :]
    if CONTRACT_START not in suffix or CONTRACT_END not in suffix:
        raise AssertionError("schema contract markers are missing")
