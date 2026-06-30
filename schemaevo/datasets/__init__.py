"""Dataset loaders for real SchemaEvo evaluations."""

from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.datasets.scorers import hotpotqa_exact_match, hover_label_accuracy

__all__ = [
    "hotpotqa_exact_match",
    "hover_label_accuracy",
    "load_hotpotqa_examples",
    "load_hover_examples",
]
