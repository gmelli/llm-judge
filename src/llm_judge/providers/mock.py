"""
Mock provider for testing and development.
"""

import random
from typing import List, Optional

from llm_judge.core.category import CategoryDefinition, Example
from llm_judge.providers.base import JudgeProvider, ProviderResult


class MockProvider(JudgeProvider):
    """Mock LLM provider for testing."""

    def __init__(self, model: str = "mock-model", temperature: float = 0.1, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.always_match = kwargs.get("always_match", False)
        self.always_fail = kwargs.get("always_fail", False)
        self.random_seed = kwargs.get("random_seed", None)

        if self.random_seed:
            random.seed(self.random_seed)

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """
        Mock evaluation that returns deterministic or random results.

        Args:
            content: The content to evaluate
            category: The category definition
            examples: Optional examples for few-shot learning

        Returns:
            Mock ProviderResult
        """
        # Determine match based on settings
        if self.always_match:
            category_match = True
            confidence = 0.95
        elif self.always_fail:
            category_match = False
            confidence = 0.95
        else:
            # Random evaluation based on content length as a simple heuristic
            content_length = len(content)
            category_match = content_length > 100
            confidence = min(0.9, content_length / 500) if category_match else 0.3

        # Generate mock scores for properties
        scores = {}
        for prop in category.characteristic_properties:
            if category_match:
                scores[prop.name] = random.uniform(0.6, 1.0)
            else:
                scores[prop.name] = random.uniform(0.0, 0.5)

        # Generate mock reasoning
        if category_match:
            reasoning = (
                f"The content appears to match the '{category.name}' category. "
                f"It satisfies most characteristic properties with high confidence."
            )
        else:
            reasoning = (
                f"The content does not appear to match the '{category.name}' category. "
                f"Several key properties are not satisfied."
            )

        return ProviderResult(
            category_match=category_match,
            confidence=confidence,
            scores=scores,
            reasoning=reasoning,
            raw_response=f"Mock evaluation for {category.name}",
            model=self.model,
            tokens_used=len(content.split()) * 2,  # Mock token count
            cost=0.0001 * len(content.split()),  # Mock cost calculation
        )

    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
    ) -> List[ProviderResult]:
        """
        Mock batch evaluation.

        Args:
            contents: List of contents to evaluate
            category: The category definition

        Returns:
            List of mock ProviderResults
        """
        results = []
        for content in contents:
            result = await self.evaluate(content, category)
            results.append(result)
        return results