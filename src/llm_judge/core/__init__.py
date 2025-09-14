"""Core modules for LLM-as-Judge library."""

from llm_judge.core.category import (
    CategoryDefinition,
    CharacteristicProperty,
    PropertyType,
    Example,
    EvaluationRubric,
    Range,
)
from llm_judge.core.evaluation import (
    EvaluationEngine,
    EvaluationResult,
    ComparisonResult,
    ComparisonEvaluator,
    PropertyScore,
)
from llm_judge.core.judge import Judge

__all__ = [
    "CategoryDefinition",
    "CharacteristicProperty",
    "PropertyType",
    "Example",
    "EvaluationRubric",
    "Range",
    "EvaluationEngine",
    "EvaluationResult",
    "ComparisonResult",
    "ComparisonEvaluator",
    "PropertyScore",
    "Judge",
]