"""
OpenAI provider implementation for LLM-as-Judge.
"""

import json
import os
from typing import Any, List, Optional

from llm_judge.core.category import CategoryDefinition, Example
from llm_judge.providers.base import JudgeProvider, ProviderResult


class OpenAIProvider(JudgeProvider):
    """OpenAI GPT-based judge provider.

    Integrates with OpenAI's GPT models to provide LLM-as-Judge evaluations.
    Supports structured JSON output and includes cost estimation based on
    token usage.

    Requires the 'openai' package and a valid OpenAI API key.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model to use (default "gpt-4-turbo-preview").
            temperature: Sampling temperature for generation.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            **kwargs: Additional configuration parameters.

        Raises:
            ImportError: If the 'openai' package is not installed.
        """
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy load OpenAI client.

        Returns:
            Configured OpenAI client instance.

        Raises:
            ImportError: If the 'openai' package is not available.
        """
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "OpenAI provider requires 'openai' package. "
                    "Install with: pip install llm-judge[providers]"
                ) from e
        return self._client

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """
        Evaluate content using OpenAI GPT models.

        Args:
            content: The content to evaluate
            category: The category definition
            examples: Optional examples for few-shot learning

        Returns:
            ProviderResult with evaluation details
        """
        # Prepare the evaluation prompt
        prompt = self.prepare_prompt(content, category, examples)

        # Create structured output prompt
        system_prompt = """You are an expert evaluator. Analyze the content and provide a JSON response with:
{
    "category_match": boolean,
    "confidence": float (0-1),
    "scores": {property_name: score, ...},
    "reasoning": "brief explanation"
}"""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            # Parse response
            result_data = json.loads(response.choices[0].message.content)

            # Calculate token usage and cost
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = self._calculate_cost(tokens_used)

            return ProviderResult(
                category_match=result_data.get("category_match", False),
                confidence=result_data.get("confidence", 0.0),
                scores=result_data.get("scores", {}),
                reasoning=result_data.get("reasoning", ""),
                raw_response=response.choices[0].message.content,
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
            )

        except Exception as e:
            # Handle errors gracefully
            return ProviderResult(
                category_match=False,
                confidence=0.0,
                scores={},
                reasoning=f"Evaluation failed: {str(e)}",
                raw_response="",
                model=self.model,
                tokens_used=0,
                cost=0.0,
            )

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
        # For now, process sequentially
        # Could be optimized with concurrent requests
        results = []
        for content in contents:
            result = await self.evaluate(content, category)
            results.append(result)
        return results

    def _calculate_cost(self, tokens: int) -> float:
        """
        Calculate approximate cost based on token usage.

        Args:
            tokens: Number of tokens used

        Returns:
            Estimated cost in USD
        """
        # Approximate pricing (as of 2024)
        # These should be updated based on current OpenAI pricing
        pricing = {
            "gpt-4-turbo-preview": 0.01 / 1000,  # $0.01 per 1K tokens
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.001 / 1000,  # $0.001 per 1K tokens
        }

        rate = pricing.get(self.model, 0.01 / 1000)
        return tokens * rate
