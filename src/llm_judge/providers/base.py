"""
Base provider interface for LLM-as-Judge implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_judge.core.category import CategoryDefinition, Example


@dataclass
class ProviderResult:
    """Result from an LLM provider evaluation."""

    category_match: bool
    confidence: float
    scores: Dict[str, float]
    reasoning: str
    raw_response: str
    model: str
    tokens_used: int = 0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class JudgeProvider(ABC):
    """Abstract base class for LLM judge providers."""

    def __init__(self, model: str, temperature: float = 0.1, **kwargs):
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """
        Evaluate content against category using LLM.

        Args:
            content: The content to evaluate
            category: The category definition
            examples: Optional examples for few-shot learning

        Returns:
            ProviderResult with evaluation details
        """
        pass

    def prepare_prompt(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> str:
        """
        Build evaluation prompt with examples and properties.

        Args:
            content: The content to evaluate
            category: The category definition
            examples: Optional examples for few-shot learning

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Evaluate if the following content belongs to category: {category.name}",
            f"\nCategory Definition: {category.description}",
        ]

        # Add characteristic properties
        if category.characteristic_properties:
            prompt_parts.append("\nCharacteristic Properties that MUST be satisfied:")
            for prop in category.characteristic_properties:
                prop_desc = f"- {prop.name}: {prop.formal_definition}"
                if prop.property_type.value == "necessary":
                    prop_desc += " (REQUIRED)"
                elif prop.property_type.value == "sufficient":
                    prop_desc += " (SUFFICIENT ALONE)"
                prompt_parts.append(prop_desc)

        # Add examples if provided
        if examples or category.positive_examples or category.negative_examples:
            # Use provided examples or fall back to category examples
            pos_examples = examples if examples else category.positive_examples[:3]
            neg_examples = category.negative_examples[:2]

            if pos_examples:
                prompt_parts.append("\nPositive Examples:")
                for i, ex in enumerate(pos_examples[:3], 1):
                    prompt_parts.append(f"{i}. {ex.content[:200]}...")

            if neg_examples:
                prompt_parts.append("\nNegative Examples:")
                for i, ex in enumerate(neg_examples[:2], 1):
                    prompt_parts.append(f"{i}. {ex.content[:200]}...")

        # Add content to evaluate
        prompt_parts.append(f"\nContent to Evaluate:\n{content}")

        # Add instructions
        prompt_parts.append(
            "\nProvide your evaluation with:"
            "\n1. Whether the content matches the category (Yes/No)"
            "\n2. Confidence score (0-1)"
            "\n3. Scores for each characteristic property"
            "\n4. Brief reasoning for your judgment"
        )

        return "\n".join(prompt_parts)

    def _format_properties(
        self, properties: List[Any]
    ) -> str:
        """Format characteristic properties for prompt."""
        formatted = []
        for prop in properties:
            formatted.append(f"- {prop.name}: {prop.formal_definition}")
        return "\n".join(formatted)

    def _format_examples(
        self, examples: List[Example]
    ) -> str:
        """Format examples for prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            # Truncate long examples
            content = ex.content[:500] + "..." if len(ex.content) > 500 else ex.content
            formatted.append(f"{i}. {content}")
            if ex.annotations:
                formatted.append(f"   Notes: {'; '.join(ex.annotations.values())}")
        return "\n".join(formatted)

    @abstractmethod
    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
    ) -> List[ProviderResult]:
        """
        Evaluate multiple contents in batch.

        Args:
            contents: List of contents to evaluate
            category: The category definition

        Returns:
            List of ProviderResults
        """
        pass

    def validate_response(self, response: str) -> bool:
        """
        Validate that the LLM response is properly formatted.

        Args:
            response: Raw LLM response

        Returns:
            True if response is valid
        """
        # Basic validation - can be overridden by specific providers
        required_elements = ["yes", "no", "confidence", "reasoning"]
        response_lower = response.lower()
        return any(elem in response_lower for elem in required_elements)