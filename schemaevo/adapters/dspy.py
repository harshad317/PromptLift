from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from schemaevo.programs.base import (
    LMProgram,
    ModuleExecutionContext,
    ModuleSignature,
    ModuleSpec,
    ProgramExample,
)


@dataclass(frozen=True)
class DSpyModuleConfig:
    name: str
    module: Callable[..., Any]
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    prompt: str = ""
    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 1024
    llm_calls: int = 1
    retriever_calls: int = 0
    demo_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def dspy_program_to_lm_program(
    *,
    task: str,
    modules: tuple[DSpyModuleConfig, ...],
    final_output_module: str,
    retriever_top_k: int = 0,
) -> LMProgram:
    """Wrap DSPy-style callables as SchemaEvo `ModuleSpec`s.

    The adapter intentionally depends only on Python call semantics. If the
    supplied DSPy modules are configured with a real LM, their normal `__call__`
    path runs that model; SchemaEvo only adds typed schema contracts around the
    module boundary.
    """

    specs = tuple(_module_spec_from_config(config) for config in modules)
    return LMProgram(
        task=task,
        modules=specs,
        retriever_top_k=retriever_top_k,
        final_output_module=final_output_module,
        metadata={"adapter": "dspy"},
    )


def _module_spec_from_config(config: DSpyModuleConfig) -> ModuleSpec:
    metadata = dict(config.metadata)
    metadata["demo_ids"] = tuple(config.demo_ids)
    metadata["adapter"] = "dspy"
    return ModuleSpec(
        name=config.name,
        signature=ModuleSignature(
            input_fields=config.input_fields,
            output_fields=config.output_fields,
        ),
        prompt=config.prompt,
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        runner=_make_runner(config.module, config.output_fields),
        llm_calls=config.llm_calls,
        retriever_calls=config.retriever_calls,
        metadata=metadata,
    )


def _make_runner(
    module: Callable[..., Any],
    output_fields: tuple[str, ...],
):
    def run(
        state: dict[str, Any],
        spec: ModuleSpec,
        example: ProgramExample,
        context: ModuleExecutionContext,
    ) -> dict[str, Any]:
        kwargs = _build_kwargs(state=state, spec=spec, example=example)
        raw = module(**kwargs)
        return _coerce_dspy_output(raw, output_fields=output_fields)

    return run


def _build_kwargs(
    *,
    state: dict[str, Any],
    spec: ModuleSpec,
    example: ProgramExample,
) -> dict[str, Any]:
    available: dict[str, Any] = {}
    available.update(state.get("inputs", {}))
    available.update(state.get("schema_fields", {}))
    for module_output in state.get("module_outputs", {}).values():
        if isinstance(module_output, Mapping):
            available.update(module_output)
    available["example"] = example
    return {field: available.get(field) for field in spec.signature.input_fields}


def _coerce_dspy_output(raw: Any, *, output_fields: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        data = dict(raw)
    elif hasattr(raw, "toDict"):
        data = dict(raw.toDict())
    elif hasattr(raw, "_store") and isinstance(raw._store, Mapping):
        data = dict(raw._store)
    else:
        data = {
            field: getattr(raw, field)
            for field in output_fields
            if hasattr(raw, field)
        }
    if not data and len(output_fields) == 1:
        return {output_fields[0]: raw}
    return data
