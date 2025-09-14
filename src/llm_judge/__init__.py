"""
LLM-as-Judge Library

A robust framework for evaluating content using Large Language Models as judges.
"""

from llm_judge.categories.registry import CategoryRegistry
from llm_judge.core.category import (
    CategoryDefinition,
    CharacteristicProperty,
    EvaluationRubric,
    Example,
    PropertyType,
    Range,
)
from llm_judge.core.evaluation import (
    ComparisonEvaluator,
    ComparisonResult,
    EvaluationEngine,
    EvaluationResult,
)
from llm_judge.core.judge import Judge
from llm_judge.providers.base import JudgeProvider, ProviderResult
from llm_judge.providers.mock import MockConfig
from llm_judge.utils.consensus import ConsensusJudge, ConsensusMode

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "CategoryDefinition",
    "CharacteristicProperty",
    "PropertyType",
    "Range",
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
    "MockConfig",
    # Consensus
    "ConsensusJudge",
    "ConsensusMode",
]
