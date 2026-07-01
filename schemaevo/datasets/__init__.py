"""Dataset loaders for real SchemaEvo evaluations."""

from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.datasets.musique import load_musique_examples
from schemaevo.datasets.scorers import hotpotqa_exact_match, hover_label_accuracy, musique_exact_match

__all__ = [
    "hotpotqa_exact_match",
    "hover_label_accuracy",
    "load_hotpotqa_examples",
    "load_hover_examples",
    "load_musique_examples",
    "musique_exact_match",
]
