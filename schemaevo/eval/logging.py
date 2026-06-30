from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Any, Protocol

from schemaevo.schemas.candidate import approximate_token_count


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_obj(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hash_text(rendered)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0


@dataclass
class ModuleCallLog:
    run_id: str
    task: str
    example_id: str
    seed: int
    method: str
    candidate_id: str
    schema_id: str
    module_name: str
    call_index_global: int
    call_index_within_example: int
    input_hash: str
    output_hash: str
    prompt_hash: str
    schema_hash: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    dollar_cost: float
    latency_ms: int
    retrieval_calls_before: int
    retrieval_calls_after: int
    retrieved_doc_ids: list[str]
    valid_json: bool
    validation_errors: list[str]
    deterministic_repair_applied: bool
    llm_repair_call_count: int
    output_payload_path: str


@dataclass(frozen=True)
class FieldUseEvent:
    run_id: str
    task: str
    example_id: str
    schema_id: str
    producer_module: str
    consumer_module: str
    field_name: str
    behavior: str


@dataclass
class CallContext:
    run_id: str
    task: str
    example_id: str
    seed: int
    method: str
    candidate_id: str
    schema_id: str
    module_name: str
    schema_hash: str
    call_index_global: int
    call_index_within_example: int
    retrieval_calls_before: int = 0
    retrieval_calls_after: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    valid_json: bool = True
    validation_errors: list[str] = field(default_factory=list)
    deterministic_repair_applied: bool = False
    llm_repair_call_count: int = 0
    output_payload_path: str = ""


class LogSink(Protocol):
    def write(self, log: ModuleCallLog) -> None:
        ...


class MemoryLogSink:
    def __init__(self) -> None:
        self.logs: list[ModuleCallLog] = []

    def write(self, log: ModuleCallLog) -> None:
        self.logs.append(log)


class JSONLLogSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, log: ModuleCallLog) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(log), sort_keys=True, ensure_ascii=True) + "\n")


class OutputPayloadStore:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, example_id: str, module_name: str, payload: Any) -> str:
        if self.root is None:
            return ""
        path = self.root / run_id / example_id
        path.mkdir(parents=True, exist_ok=True)
        output_path = path / f"{module_name}.json"
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(payload, tmp, sort_keys=True, ensure_ascii=True, default=str, indent=2)
            tmp.write("\n")
        tmp_path.replace(output_path)
        return str(output_path)


def make_module_log(
    *,
    context: CallContext,
    module_prompt: str,
    module_input: Any,
    module_output: Any,
    latency_start: float,
    cost: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cached_tokens: int = 0,
) -> ModuleCallLog:
    latency_ms = int((time.perf_counter() - latency_start) * 1000)
    computed_prompt_tokens = (
        prompt_tokens
        if prompt_tokens is not None
        else approximate_token_count(module_prompt + json.dumps(module_input, default=str))
    )
    output_text = json.dumps(module_output, sort_keys=True, ensure_ascii=True, default=str)
    usage = Usage(
        prompt_tokens=computed_prompt_tokens,
        completion_tokens=completion_tokens
        if completion_tokens is not None
        else approximate_token_count(output_text),
        cached_tokens=cached_tokens,
    )
    return ModuleCallLog(
        run_id=context.run_id,
        task=context.task,
        example_id=context.example_id,
        seed=context.seed,
        method=context.method,
        candidate_id=context.candidate_id,
        schema_id=context.schema_id,
        module_name=context.module_name,
        call_index_global=context.call_index_global,
        call_index_within_example=context.call_index_within_example,
        input_hash=hash_obj(module_input),
        output_hash=hash_obj(module_output),
        prompt_hash=hash_text(module_prompt),
        schema_hash=context.schema_hash,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        dollar_cost=cost,
        latency_ms=latency_ms,
        retrieval_calls_before=context.retrieval_calls_before,
        retrieval_calls_after=context.retrieval_calls_after,
        retrieved_doc_ids=list(context.retrieved_doc_ids),
        valid_json=context.valid_json,
        validation_errors=list(context.validation_errors),
        deterministic_repair_applied=context.deterministic_repair_applied,
        llm_repair_call_count=context.llm_repair_call_count,
        output_payload_path=context.output_payload_path,
    )
