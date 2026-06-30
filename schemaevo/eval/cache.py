from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from schemaevo.eval.logging import FieldUseEvent, ModuleCallLog, hash_obj
from schemaevo.programs.base import LMProgram, ProgramExample, ProgramPrediction


class RolloutCache:
    """Content-addressed rollout cache for repeated schema/example evaluations."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self.memory: dict[str, ProgramPrediction] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        *,
        program: LMProgram,
        example: ProgramExample,
        seed: int,
        intervention_id: str = "none",
    ) -> str:
        return hash_obj(
            {
                "program": program_fingerprint(program),
                "example_id": example.example_id,
                "seed": seed,
                "intervention_id": intervention_id,
            }
        )

    def get(self, key: str) -> ProgramPrediction | None:
        if key in self.memory:
            return self.memory[key]
        if not self.root:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            prediction = prediction_from_dict(json.load(handle))
        self.memory[key] = prediction
        return prediction

    def set(self, key: str, prediction: ProgramPrediction) -> None:
        self.memory[key] = prediction
        if not self.root:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(prediction_to_dict(prediction), handle, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        tmp_path.replace(path)

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / f"{key}.json"


def program_fingerprint(program: LMProgram) -> dict[str, Any]:
    return {
        "task": program.task,
        "retriever_top_k": program.retriever_top_k,
        "final_output_module": program.final_output_module,
        "schema": program.schema_candidate.to_dict() if program.schema_candidate else None,
        "modules": [
            {
                "name": module.name,
                "signature": {
                    "input_fields": module.signature.input_fields,
                    "output_fields": module.signature.output_fields,
                },
                "prompt": module.prompt,
                "model": module.model,
                "max_output_tokens": module.max_output_tokens,
                "llm_calls": module.llm_calls,
                "retriever_calls": module.retriever_calls,
            }
            for module in program.modules
        ],
    }


def prediction_to_dict(prediction: ProgramPrediction) -> dict[str, Any]:
    data = asdict(prediction)
    data["module_logs"] = [asdict(log) if hasattr(log, "__dataclass_fields__") else log for log in prediction.module_logs]
    data["field_use_events"] = [asdict(event) for event in prediction.field_use_events]
    return data


def prediction_from_dict(data: dict[str, Any]) -> ProgramPrediction:
    return ProgramPrediction(
        run_id=data["run_id"],
        example_id=data["example_id"],
        candidate_id=data["candidate_id"],
        schema_id=data["schema_id"],
        final_output=dict(data["final_output"]),
        module_outputs={name: dict(output) for name, output in data["module_outputs"].items()},
        valid=bool(data["valid"]),
        validation_errors=tuple(data["validation_errors"]),
        module_logs=[ModuleCallLog(**log) for log in data["module_logs"]],
        field_use_events=[FieldUseEvent(**event) for event in data["field_use_events"]],
        target_task_calls=int(data["target_task_calls"]),
        retriever_calls=int(data["retriever_calls"]),
        prompt_tokens=int(data["prompt_tokens"]),
        completion_tokens=int(data["completion_tokens"]),
        dollar_cost=float(data["dollar_cost"]),
        latency_ms=int(data["latency_ms"]),
    )
