from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from schemaevo.adapters.openai import OpenAIModuleConfig, openai_modules_to_lm_program
from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.datasets.musique import load_musique_examples
from schemaevo.datasets.scorers import hotpotqa_exact_match, hover_label_accuracy, musique_exact_match
from schemaevo.eval.scoring import Scorer
from schemaevo.optimizers.fixed_pool_schema import FixedPoolConfig, FixedPoolResult, run_fixed_pool_schema_mvp
from schemaevo.programs.base import LMProgram, ProgramExample
from schemaevo.schemas.proposer import HeuristicTraceSchemaProposer, OpenAISchemaProposer, SchemaProposer, TraceExample

DatasetName = Literal["hotpotqa", "hover", "musique"]


@dataclass(frozen=True)
class OpenAIFixedPoolBenchmarkConfig:
    dataset: DatasetName
    train_path: str | Path
    selection_path: str | Path
    confirmation_path: str | Path
    smoke_path: str | Path | None = None
    heldout_path: str | Path | None = None
    train_limit: int | None = None
    smoke_limit: int | None = None
    selection_limit: int | None = None
    confirmation_limit: int | None = None
    heldout_limit: int | None = None
    model: str = "gpt-4.1-mini"
    temperature: float | None = None
    retriever_top_k: int = 0


def run_openai_fixed_pool_benchmark(
    *,
    benchmark_config: OpenAIFixedPoolBenchmarkConfig,
    fixed_pool_config: FixedPoolConfig,
    proposer: SchemaProposer | None = None,
    artifact_dir: str | Path | None = None,
    client: Any | None = None,
) -> FixedPoolResult:
    program = build_openai_benchmark_program(benchmark_config, client=client)
    scorer = _scorer_for_dataset(benchmark_config.dataset)
    train_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.train_path,
        split="train",
        limit=benchmark_config.train_limit,
    )
    smoke_examples = (
        _load_examples(
            dataset=benchmark_config.dataset,
            path=benchmark_config.smoke_path,
            split="validation_smoke",
            limit=benchmark_config.smoke_limit,
        )
        if benchmark_config.smoke_path
        else ()
    )
    selection_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.selection_path,
        split="validation_selection",
        limit=benchmark_config.selection_limit,
    )
    confirmation_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.confirmation_path,
        split="validation_confirmation",
        limit=benchmark_config.confirmation_limit,
    )
    heldout_examples = (
        _load_examples(
            dataset=benchmark_config.dataset,
            path=benchmark_config.heldout_path,
            split="final_test",
            limit=benchmark_config.heldout_limit,
        )
        if benchmark_config.heldout_path
        else ()
    )
    train_traces = make_train_traces_from_examples(
        dataset=benchmark_config.dataset,
        examples=train_examples,
    )
    return run_fixed_pool_schema_mvp(
        base_program=program,
        train_traces=train_traces,
        smoke_examples=smoke_examples,
        selection_examples=selection_examples,
        confirmation_examples=confirmation_examples,
        heldout_test_examples=heldout_examples,
        scorer=scorer,
        config=fixed_pool_config,
        proposer=proposer or HeuristicTraceSchemaProposer(),
        artifact_dir=artifact_dir,
    )


def build_openai_benchmark_program(
    config: OpenAIFixedPoolBenchmarkConfig,
    *,
    client: Any | None = None,
) -> LMProgram:
    if config.dataset in {"hotpotqa", "musique"}:
        task_name = "HotpotQA" if config.dataset == "hotpotqa" else "MuSiQue"
        modules = (
            OpenAIModuleConfig(
                name="planner",
                input_fields=("question", "context"),
                output_fields=("plan_summary",),
                output_field_types={"plan_summary": "string"},
                prompt=(
                    f"Read the {task_name} context and write a concise plan_summary capturing the specific facts "
                    "needed to answer the question: the key entities, dates, numbers, and relationships, "
                    "including the bridge fact that links the hops. The downstream answerer will NOT see "
                    "the source documents, so include every fact required to answer. Return JSON only."
                ),
                model=config.model,
                max_output_tokens=512,
                temperature=config.temperature,
                retriever_calls=1 if config.retriever_top_k else 0,
            ),
            OpenAIModuleConfig(
                # Interface-bottleneck setup: the answerer does NOT see raw context.
                # It must rely on what the planner distilled (plan_summary + SchemaEvo fields),
                # so the planner->answerer schema is the only information channel.
                name="answerer",
                input_fields=("question", "plan_summary"),
                output_fields=("answer", "confidence"),
                output_field_types={"answer": "string", "confidence": "number"},
                prompt=(
                    f"Answer the {task_name} question using ONLY the planner's plan and any SchemaEvo fields. "
                    "You do not have the source documents; rely on the distilled plan/evidence provided. "
                    "Give the shortest exact answer span only - a name, entity, number, or 'yes'/'no'. "
                    "Do not write a sentence or any explanation. "
                    "Return JSON with answer (the span) and confidence."
                ),
                model=config.model,
                max_output_tokens=256,
                temperature=config.temperature,
            ),
        )
        return openai_modules_to_lm_program(
            task=task_name,
            modules=modules,
            final_output_module="answerer",
            retriever_top_k=config.retriever_top_k,
            client=client,
        )
    if config.dataset == "hover":
        modules = (
            OpenAIModuleConfig(
                name="planner",
                input_fields=("claim", "context"),
                output_fields=("plan_summary",),
                output_field_types={"plan_summary": "string"},
                prompt=(
                    "Plan the evidence checks needed to verify the HoVer claim from the provided context. "
                    "Return JSON only."
                ),
                model=config.model,
                max_output_tokens=512,
                temperature=config.temperature,
                retriever_calls=1 if config.retriever_top_k else 0,
            ),
            OpenAIModuleConfig(
                name="verifier",
                input_fields=("claim", "context", "plan_summary"),
                output_fields=("label", "confidence"),
                output_field_types={"label": "string", "confidence": "number"},
                prompt=(
                    "Verify the HoVer claim using the context, plan, and any SchemaEvo fields. "
                    "Return JSON with label and confidence."
                ),
                model=config.model,
                max_output_tokens=256,
                temperature=config.temperature,
            ),
        )
        return openai_modules_to_lm_program(
            task="HoVer",
            modules=modules,
            final_output_module="verifier",
            retriever_top_k=config.retriever_top_k,
            client=client,
        )
    raise ValueError(f"unsupported dataset: {config.dataset}")


def make_train_traces_from_examples(
    *,
    dataset: DatasetName,
    examples: tuple[ProgramExample, ...],
) -> tuple[TraceExample, ...]:
    traces: list[TraceExample] = []
    for example in examples:
        if dataset in {"hotpotqa", "musique"}:
            input_summary = f"question={example.inputs.get('question', '')}"
            output_summary = (
                "The schema should preserve multi-hop bridge entities, next retrieval intent, "
                "evidence needs, and answer constraints across the planner->answerer boundary."
            )
        elif dataset == "hover":
            input_summary = f"claim={example.inputs.get('claim', '')}"
            output_summary = (
                "The schema should preserve claim atoms, evidence requirements, conflict state, "
                "and verification label constraints across the planner->verifier boundary."
            )
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
        traces.append(
            TraceExample(
                example_id=example.example_id,
                split="train",
                module_name="planner",
                input_summary=input_summary[:1024],
                output_summary=output_summary,
                score=None,
                errors=("train_only_schema_proposal_trace",),
                metadata={"dataset": dataset},
            )
        )
    return tuple(traces)


def make_proposer(kind: str, *, model: str, temperature: float, max_output_tokens: int) -> SchemaProposer:
    if kind == "heuristic":
        return HeuristicTraceSchemaProposer()
    if kind == "openai":
        return OpenAISchemaProposer(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"unknown proposer kind: {kind}")


def _load_examples(
    *,
    dataset: DatasetName,
    path: str | Path,
    split: str,
    limit: int | None,
) -> tuple[ProgramExample, ...]:
    if dataset == "hotpotqa":
        return load_hotpotqa_examples(path, split=split, limit=limit)
    if dataset == "musique":
        return load_musique_examples(path, split=split, limit=limit)
    if dataset == "hover":
        return load_hover_examples(path, split=split, limit=limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def _scorer_for_dataset(dataset: DatasetName) -> Scorer:
    if dataset == "hotpotqa":
        return hotpotqa_exact_match
    if dataset == "musique":
        return musique_exact_match
    if dataset == "hover":
        return hover_label_accuracy
    raise ValueError(f"unsupported dataset: {dataset}")
