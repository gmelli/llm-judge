"""
LLM-as-Judge Library

A robust framework for evaluating content using Large Language Models as judges.
"""

from llm_judge.core.category import (
    CategoryDefinition,
    CharacteristicProperty,
    PropertyType,
    Example,
    EvaluationRubric,
)
from llm_judge.core.evaluation import (
    EvaluationEngine,
    EvaluationResult,
    ComparisonResult,
    ComparisonEvaluator,
)
from llm_judge.core.judge import Judge
from llm_judge.categories.registry import CategoryRegistry
from llm_judge.providers.base import JudgeProvider, ProviderResult
from llm_judge.utils.consensus import ConsensusJudge, ConsensusMode

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "CategoryDefinition",
    "CharacteristicProperty",
    "PropertyType",
    "Example",
    "EvaluationRubric",
    "EvaluationEngine",
    "EvaluationResult",
    "ComparisonResult",
    "ComparisonEvaluator",
    "Judge",
    # Registry
    "CategoryRegistry",
    # Providers
    "JudgeProvider",
    "ProviderResult",
    # Consensus
    "ConsensusJudge",
    "ConsensusMode",
]