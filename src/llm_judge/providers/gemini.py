"""
Google Gemini provider implementation for LLM-as-Judge.
"""

import json
import os
from typing import List, Optional, Any

from llm_judge.core.category import CategoryDefinition, Example
from llm_judge.providers.base import JudgeProvider, ProviderResult


class GeminiProvider(JudgeProvider):
    """Google Gemini-based judge provider.

    Integrates with Google's Gemini models to provide LLM-as-Judge evaluations.
    Handles JSON parsing from Gemini responses and provides cost estimation
    based on character/token usage.

    Requires the 'google-generativeai' package and a valid Google API key.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Gemini provider.

        Args:
            model: Gemini model to use (default "gemini-1.5-pro").
            temperature: Sampling temperature for generation.
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            **kwargs: Additional configuration parameters.

        Raises:
            ImportError: If the 'google-generativeai' package is not installed.
        """
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Lazy load Google Generative AI client.

        Returns:
            Configured Google GenerativeModel instance.

        Raises:
            ImportError: If the 'google-generativeai' package is not available.
        """
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError as e:
                raise ImportError(
                    "Gemini provider requires 'google-generativeai' package. "
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
        Evaluate content using Google Gemini models.

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
}}

Return ONLY the JSON object, no additional text."""

        try:
            # Call Gemini API
            response = self.client.generate_content(
                json_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1000,
                },
            )

            # Extract text from response
            response_text = response.text if response else ""

            # Try to parse JSON from the response
            # Gemini might include markdown formatting
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result_data = json.loads(json_str)
            else:
                # Try direct parsing
                try:
                    result_data = json.loads(response_text.strip())
                except json.JSONDecodeError:
                    # Fallback if no JSON found
                    result_data = {
                        "category_match": False,
                        "confidence": 0.0,
                        "scores": {},
                        "reasoning": "Unable to parse response",
                    }

            # Calculate token usage and cost (Gemini doesn't provide token counts easily)
            # Estimate based on text length
            estimated_tokens = len(json_prompt.split()) + len(response_text.split())
            cost = self._calculate_cost(estimated_tokens)

            return ProviderResult(
                category_match=result_data.get("category_match", False),
                confidence=result_data.get("confidence", 0.0),
                scores=result_data.get("scores", {}),
                reasoning=result_data.get("reasoning", ""),
                raw_response=response_text,
                model=self.model,
                tokens_used=estimated_tokens,
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
        # Gemini has batch API but it's more complex
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
        # Approximate pricing for Gemini (as of 2024)
        # Note: Gemini pricing is per character, converting approximately
        pricing = {
            "gemini-1.5-pro": 0.00125 / 1000,  # $0.00125 per 1K characters
            "gemini-1.5-flash": 0.0003 / 1000,  # $0.0003 per 1K characters
            "gemini-1.0-pro": 0.0005 / 1000,  # $0.0005 per 1K characters
        }

        rate = pricing.get(self.model, 0.001 / 1000)
        # Rough conversion: 1 token ≈ 4 characters
        return tokens * 4 * rate