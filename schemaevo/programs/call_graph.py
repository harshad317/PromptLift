from __future__ import annotations

from dataclasses import dataclass

from schemaevo.programs.base import LMProgram


@dataclass(frozen=True)
class CallGraph:
    modules: tuple[str, ...]
    llm_calls_by_module: tuple[tuple[str, int], ...]
    retriever_calls_by_module: tuple[tuple[str, int], ...]
    demo_ids_by_module: tuple[tuple[str, tuple[str, ...]], ...]
    edges: tuple[tuple[str, str], ...]
    retriever_top_k: int


def extract_call_graph(program: LMProgram) -> CallGraph:
    modules = program.module_names
    edges = tuple(zip(modules, modules[1:]))
    return CallGraph(
        modules=modules,
        llm_calls_by_module=tuple((module.name, module.llm_calls) for module in program.modules),
        retriever_calls_by_module=tuple((module.name, module.retriever_calls) for module in program.modules),
        demo_ids_by_module=tuple(
            (module.name, _demo_ids(module.metadata)) for module in program.modules
        ),
        edges=edges,
        retriever_top_k=program.retriever_top_k,
    )


def assert_same_call_graph(candidate: LMProgram, baseline: LMProgram) -> None:
    candidate_graph = extract_call_graph(candidate)
    baseline_graph = extract_call_graph(baseline)
    if candidate_graph != baseline_graph:
        raise AssertionError(
            "call graph changed under schema compilation:\n"
            f"baseline={baseline_graph}\n"
            f"candidate={candidate_graph}"
        )
    if candidate.calls_per_example != baseline.calls_per_example:
        raise AssertionError(
            f"LLM call count changed: {candidate.calls_per_example} != {baseline.calls_per_example}"
        )
    if candidate.retriever_calls_per_example != baseline.retriever_calls_per_example:
        raise AssertionError(
            "retriever call count changed: "
            f"{candidate.retriever_calls_per_example} != {baseline.retriever_calls_per_example}"
        )
    if candidate.retriever_top_k != baseline.retriever_top_k:
        raise AssertionError(
            f"retriever top-k changed: {candidate.retriever_top_k} != {baseline.retriever_top_k}"
        )


def _demo_ids(metadata: dict[str, object]) -> tuple[str, ...]:
    raw_ids = metadata.get("demo_ids", metadata.get("few_shot_demo_ids", ()))
    if isinstance(raw_ids, str):
        return (raw_ids,)
    if isinstance(raw_ids, (list, tuple)):
        return tuple(str(item) for item in raw_ids)
    return ()
