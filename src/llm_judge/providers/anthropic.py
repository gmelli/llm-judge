"""
Anthropic Claude provider implementation for LLM-as-Judge.
"""

import json
import os
from typing import List, Optional

from llm_judge.core.category import CategoryDefinition, Example
from llm_judge.providers.base import JudgeProvider, ProviderResult


class AnthropicProvider(JudgeProvider):
    """Anthropic Claude-based judge provider."""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "Anthropic provider requires 'anthropic' package. "
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
        Evaluate content using Anthropic Claude models.

        Args:
            content: The content to evaluate
            category: The category definition
            examples: Optional examples for few-shot learning

        Returns:
            ProviderResult with evaluation details
        """
        # Prepare the evaluation prompt
        prompt = self.prepare_prompt(content, category, examples)

        # Add JSON output instruction
        json_prompt = f"""{prompt}

Please provide your evaluation as a JSON object with this structure:
{{
    "category_match": true/false,
    "confidence": 0.0-1.0,
    "scores": {{"property_name": score, ...}},
    "reasoning": "your explanation"
}}"""

        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": json_prompt}],
            )

            # Extract JSON from response
            response_text = response.content[0].text if response.content else ""

            # Try to parse JSON from the response
            # Claude might include markdown formatting
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result_data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                result_data = {
                    "category_match": False,
                    "confidence": 0.0,
                    "scores": {},
                    "reasoning": "Unable to parse response",
                }

            # Calculate token usage and cost
            tokens_used = (
                response.usage.input_tokens + response.usage.output_tokens
                if hasattr(response, "usage")
                else 0
            )
            cost = self._calculate_cost(tokens_used)

            return ProviderResult(
                category_match=result_data.get("category_match", False),
                confidence=result_data.get("confidence", 0.0),
                scores=result_data.get("scores", {}),
                reasoning=result_data.get("reasoning", ""),
                raw_response=response_text,
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
        # Process sequentially for now
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
        # These should be updated based on current Anthropic pricing
        pricing = {
            "claude-3-opus-20240229": 0.015 / 1000,  # $0.015 per 1K tokens input
            "claude-3-sonnet-20240229": 0.003 / 1000,  # $0.003 per 1K tokens
            "claude-3-haiku-20240307": 0.00025 / 1000,  # $0.00025 per 1K tokens
            "claude-2.1": 0.008 / 1000,  # $0.008 per 1K tokens
        }

        rate = pricing.get(self.model, 0.01 / 1000)
        return tokens * rate
