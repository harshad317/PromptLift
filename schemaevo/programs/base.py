from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Protocol

from schemaevo.eval.cost_ledger import CostLedger, CostLedgerEntry, CostMeter
from schemaevo.eval.logging import (
    CallContext,
    FieldUseEvent,
    LogSink,
    MemoryLogSink,
    OutputPayloadStore,
    hash_obj,
    make_module_log,
)
from schemaevo.schemas.candidate import SchemaCandidate
from schemaevo.schemas.validators import SchemaValidator, ValidationPolicy


@dataclass(frozen=True)
class ModuleSignature:
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]

    def with_additional_outputs(self, field_names: tuple[str, ...]) -> "ModuleSignature":
        outputs = list(self.output_fields)
        for field_name in field_names:
            if field_name not in outputs:
                outputs.append(field_name)
        return ModuleSignature(input_fields=self.input_fields, output_fields=tuple(outputs))


@dataclass(frozen=True)
class ProgramExample:
    example_id: str
    split: str
    inputs: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModuleExecutionContext:
    def __init__(self, *, run_id: str, task: str, example_id: str, schema_id: str) -> None:
        self.run_id = run_id
        self.task = task
        self.example_id = example_id
        self.schema_id = schema_id
        self.field_use_events: list[FieldUseEvent] = []

    def record_field_use(
        self,
        *,
        producer_module: str,
        consumer_module: str,
        field_name: str,
        behavior: str,
    ) -> None:
        self.field_use_events.append(
            FieldUseEvent(
                run_id=self.run_id,
                task=self.task,
                example_id=self.example_id,
                schema_id=self.schema_id,
                producer_module=producer_module,
                consumer_module=consumer_module,
                field_name=field_name,
                behavior=behavior,
            )
        )


ModuleRunner = Callable[
    [dict[str, Any], "ModuleSpec", ProgramExample, ModuleExecutionContext],
    dict[str, Any],
]


@dataclass
class ModuleSpec:
    name: str
    signature: ModuleSignature
    prompt: str
    model: str
    max_output_tokens: int
    runner: ModuleRunner | None = None
    llm_calls: int = 1
    retriever_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ModuleSpec":
        return ModuleSpec(
            name=self.name,
            signature=deepcopy(self.signature),
            prompt=self.prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            runner=self.runner,
            llm_calls=self.llm_calls,
            retriever_calls=self.retriever_calls,
            metadata=deepcopy(self.metadata),
        )


class FieldIntervention(Protocol):
    def before_module(
        self,
        *,
        module_name: str,
        state: dict[str, Any],
        example: ProgramExample,
    ) -> None:
        ...


@dataclass
class ProgramPrediction:
    run_id: str
    example_id: str
    candidate_id: str
    schema_id: str
    final_output: dict[str, Any]
    module_outputs: dict[str, dict[str, Any]]
    valid: bool
    validation_errors: tuple[str, ...]
    module_logs: list[Any]
    field_use_events: list[FieldUseEvent]
    target_task_calls: int
    retriever_calls: int
    prompt_tokens: int
    completion_tokens: int
    dollar_cost: float
    latency_ms: int


@dataclass
class LMProgram:
    task: str
    modules: tuple[ModuleSpec, ...]
    retriever_top_k: int
    final_output_module: str
    schema_candidate: SchemaCandidate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "LMProgram":
        return LMProgram(
            task=self.task,
            modules=tuple(module.clone() for module in self.modules),
            retriever_top_k=self.retriever_top_k,
            final_output_module=self.final_output_module,
            schema_candidate=self.schema_candidate,
            metadata=deepcopy(self.metadata),
        )

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    @property
    def calls_per_example(self) -> int:
        return sum(module.llm_calls for module in self.modules)

    @property
    def retriever_calls_per_example(self) -> int:
        return sum(module.retriever_calls for module in self.modules)

    def run(
        self,
        example: ProgramExample,
        *,
        run_id: str,
        method: str,
        candidate_id: str,
        seed: int,
        validation_policy: ValidationPolicy = "deterministic_coercion",
        log_sink: LogSink | None = None,
        cost_meter: CostMeter | None = None,
        cost_ledger: CostLedger | None = None,
        payload_store: OutputPayloadStore | None = None,
        field_intervention: FieldIntervention | None = None,
    ) -> ProgramPrediction:
        schema = self.schema_candidate
        schema_id = schema.schema_id if schema else "original_schema"
        schema_hash = hash_obj(schema.to_dict() if schema else {})
        validator = SchemaValidator(schema) if schema else None
        log_sink = log_sink or MemoryLogSink()
        cost_meter = cost_meter or CostMeter()
        payload_store = payload_store or OutputPayloadStore(None)

        state: dict[str, Any] = {
            "inputs": dict(example.inputs),
            "expected": dict(example.expected),
            "metadata": dict(example.metadata),
            "module_outputs": {},
            "schema_fields": {},
            "field_producers": {},
        }
        logs: list[Any] = []
        validation_errors: list[str] = []
        global_call_index = 0
        target_task_calls = 0
        retriever_calls = 0
        execution_context = ModuleExecutionContext(
            run_id=run_id,
            task=self.task,
            example_id=example.example_id,
            schema_id=schema_id,
        )

        for within_index, module in enumerate(self.modules, start=1):
            if field_intervention:
                field_intervention.before_module(
                    module_name=module.name,
                    state=state,
                    example=example,
                )
            if module.runner is None:
                raise RuntimeError(f"module {module.name!r} has no runner")
            module_input_snapshot = {
                "inputs": deepcopy(state["inputs"]),
                "schema_fields": deepcopy(state["schema_fields"]),
                "module_outputs": deepcopy(state["module_outputs"]),
            }
            global_call_index += module.llm_calls
            target_task_calls += module.llm_calls
            retrieval_before = retriever_calls
            retriever_calls += module.retriever_calls
            latency_start = time.perf_counter()
            raw_output = module.runner(state, module, example, execution_context)
            payload_path = payload_store.write(run_id, example.example_id, module.name, raw_output)
            validation_result = (
                validator.validate_module_output(module.name, raw_output, validation_policy)
                if validator
                else None
            )
            valid_json = True
            deterministic_repair = False
            parsed_output = dict(raw_output)
            errors: list[str] = []
            if validation_result:
                valid_json = validation_result.valid
                deterministic_repair = validation_result.deterministic_repair_applied
                parsed_output = validation_result.parsed or {}
                errors = list(validation_result.errors)
                validation_errors.extend(f"{module.name}: {error}" for error in errors)
            state["module_outputs"][module.name] = parsed_output
            schema_output_names = {
                field["name"] for field in module.metadata.get("schemaevo_output_fields", [])
            }
            for field_name, value in parsed_output.items():
                if field_name in schema_output_names:
                    state["schema_fields"][field_name] = value
                    state["field_producers"][field_name] = module.name
            prompt = module.prompt
            context = CallContext(
                run_id=run_id,
                task=self.task,
                example_id=example.example_id,
                seed=seed,
                method=method,
                candidate_id=candidate_id,
                schema_id=schema_id,
                module_name=module.name,
                schema_hash=schema_hash,
                call_index_global=global_call_index,
                call_index_within_example=within_index,
                retrieval_calls_before=retrieval_before,
                retrieval_calls_after=retriever_calls,
                valid_json=valid_json,
                validation_errors=errors,
                deterministic_repair_applied=deterministic_repair,
                output_payload_path=payload_path,
            )
            prompt_tokens_probe = prompt + str(module_input_snapshot)
            completion_tokens_probe = str(raw_output)
            prompt_tokens = cost_meter.count_tokens(model=module.model, text=prompt_tokens_probe)
            completion_tokens = cost_meter.count_tokens(model=module.model, text=completion_tokens_probe)
            dollar_cost = cost_meter.compute(
                model=module.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            log = make_module_log(
                context=context,
                module_prompt=prompt,
                module_input=module_input_snapshot,
                module_output=raw_output,
                latency_start=latency_start,
                cost=dollar_cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            log_sink.write(log)
            logs.append(log)
            if cost_ledger:
                cost_ledger.add(
                    CostLedgerEntry(
                        run_id=run_id,
                        method=method,
                        candidate_id=candidate_id,
                        model=module.model,
                        call_type="target_task",
                        prompt_tokens=log.prompt_tokens,
                        completion_tokens=log.completion_tokens,
                        cached_tokens=log.cached_tokens,
                        dollar_cost=log.dollar_cost,
                        latency_ms=log.latency_ms,
                    )
                )

        final_output = dict(state["module_outputs"].get(self.final_output_module, {}))
        return ProgramPrediction(
            run_id=run_id,
            example_id=example.example_id,
            candidate_id=candidate_id,
            schema_id=schema_id,
            final_output=final_output,
            module_outputs=deepcopy(state["module_outputs"]),
            valid=not validation_errors,
            validation_errors=tuple(validation_errors),
            module_logs=logs,
            field_use_events=list(execution_context.field_use_events),
            target_task_calls=target_task_calls,
            retriever_calls=retriever_calls,
            prompt_tokens=sum(log.prompt_tokens for log in logs),
            completion_tokens=sum(log.completion_tokens for log in logs),
            dollar_cost=sum(log.dollar_cost for log in logs),
            latency_ms=sum(log.latency_ms for log in logs),
        )
