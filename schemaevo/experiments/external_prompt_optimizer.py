from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from schemaevo.programs.base import LMProgram
from schemaevo.programs.call_graph import assert_same_call_graph


@dataclass(frozen=True)
class ExternalPromptOptimizer:
    name: str
    command: str
    artifact_dir: str | Path | None = None
    timeout_seconds: int = 3600
    allow_demo_changes: bool = False

    def __call__(self, program: LMProgram) -> LMProgram:
        root = (
            Path(self.artifact_dir).resolve()
            if self.artifact_dir
            else Path(tempfile.mkdtemp(prefix="schemaevo_promptopt_")).resolve()
        )
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / f"{self.name}_input_program.json"
        output_path = root / f"{self.name}_output_program.json"
        input_path.write_text(
            json.dumps(program_to_prompt_spec(program), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "SCHEMAEVO_PROMPT_OPTIMIZER": self.name,
            "SCHEMAEVO_INPUT_PROGRAM": str(input_path),
            "SCHEMAEVO_OUTPUT_PROGRAM": str(output_path),
        }
        subprocess.run(
            self.command,
            shell=True,
            check=True,
            cwd=str(root),
            env=env,
            timeout=self.timeout_seconds,
        )
        if not output_path.exists():
            raise FileNotFoundError(
                f"external optimizer {self.name!r} did not write {output_path}"
            )
        optimized = apply_prompt_spec(program, json.loads(output_path.read_text(encoding="utf-8")))
        if self.allow_demo_changes:
            _assert_same_deployment_graph(optimized, program)
        else:
            assert_same_call_graph(optimized, program)
        return optimized


def program_to_prompt_spec(program: LMProgram) -> dict[str, Any]:
    return {
        "task": program.task,
        "retriever_top_k": program.retriever_top_k,
        "final_output_module": program.final_output_module,
        "metadata": dict(program.metadata),
        "modules": [
            {
                "name": module.name,
                "prompt": module.prompt,
                "model": module.model,
                "max_output_tokens": module.max_output_tokens,
                "llm_calls": module.llm_calls,
                "retriever_calls": module.retriever_calls,
                "input_fields": list(module.signature.input_fields),
                "output_fields": list(module.signature.output_fields),
                "metadata": dict(module.metadata),
            }
            for module in program.modules
        ],
    }


def apply_prompt_spec(program: LMProgram, spec: dict[str, Any]) -> LMProgram:
    by_name = {
        str(module_spec.get("name")): module_spec
        for module_spec in spec.get("modules", [])
        if isinstance(module_spec, dict)
    }
    optimized = program.clone()
    for module in optimized.modules:
        patch = by_name.get(module.name, {})
        if "prompt" in patch:
            module.prompt = str(patch["prompt"])
        metadata = patch.get("metadata")
        if isinstance(metadata, dict):
            for key in ("demo_ids", "few_shot_demo_ids"):
                if key in metadata:
                    raw_ids = metadata[key]
                    if isinstance(raw_ids, str):
                        module.metadata[key] = (raw_ids,)
                    elif isinstance(raw_ids, (list, tuple)):
                        module.metadata[key] = tuple(str(item) for item in raw_ids)
    return optimized


def _assert_same_deployment_graph(candidate: LMProgram, baseline: LMProgram) -> None:
    if candidate.module_names != baseline.module_names:
        raise AssertionError("module order changed")
    if candidate.calls_per_example != baseline.calls_per_example:
        raise AssertionError("LLM call count changed")
    if candidate.retriever_calls_per_example != baseline.retriever_calls_per_example:
        raise AssertionError("retriever call count changed")
    if candidate.retriever_top_k != baseline.retriever_top_k:
        raise AssertionError("retriever top-k changed")
    for left, right in zip(candidate.modules, baseline.modules):
        if left.signature != right.signature:
            raise AssertionError(f"module signature changed for {left.name}")
        if left.max_output_tokens != right.max_output_tokens:
            raise AssertionError(f"max_output_tokens changed for {left.name}")
