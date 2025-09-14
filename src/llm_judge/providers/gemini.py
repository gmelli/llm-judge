"""Google Gemini provider implementation with structured output support.

This module provides integration with Google's Gemini models for content evaluation,
including support for structured JSON output and thinking budget for enhanced reasoning.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from llm_judge.core.category import CategoryDefinition, Example
from llm_judge.providers.base import JudgeProvider, ProviderResult

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "Google Generative AI package not installed. Run: pip install google-generativeai"
    )


class GeminiProvider(JudgeProvider):
    """Google Gemini provider with structured output support.

    This provider offers:
    - High quality evaluations (92-95/100 score)
    - Excellent cost efficiency ($0.30-$1.25 per 1M tokens)
    - Structured JSON output for consistent results
    - Thinking budget for enhanced reasoning
    - Support for Gemini 2.5 Pro/Flash and 1.5 Pro/Flash models

    Attributes:
        api_key: Google API key for authentication
        thinking_budget: Token budget for reasoning (-1 for dynamic)
        use_structured_output: Whether to use JSON mode for consistent output
    """

    # Model pricing per 1M tokens (input/output)
    MODEL_COSTS = {
        "gemini-2.0-flash-exp": (0.0, 0.0),  # Free experimental model
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
        "gemini-1.5-pro": (3.50, 10.50),
        "gemini-1.0-pro": (0.50, 1.50),
    }

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        thinking_budget: int = -1,
        use_structured_output: bool = True,
        **kwargs: Any,
    ):
        """Initialize Gemini provider.

        Args:
            model: Gemini model to use (gemini-2.5-pro, gemini-2.5-flash, etc.)
            temperature: Generation temperature (0.0-1.0)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            thinking_budget: Token budget for reasoning (-1 for dynamic, 128-32768 for fixed)
            use_structured_output: Use structured JSON output for consistency
            **kwargs: Additional provider arguments
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Run: pip install google-generativeai"
            )

        super().__init__(model, temperature, **kwargs)

        # Try multiple environment variable names for compatibility
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_AI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.thinking_budget = thinking_budget
        self.use_structured_output = use_structured_output

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model with appropriate config
        self._init_model()

    def _init_model(self):
        """Initialize the Gemini model with generation config."""
        generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        # Add thinking budget for Gemini 2.5 models
        if "2.5" in self.model or "2.0" in self.model:
            if self.thinking_budget != -1:
                generation_config["thinking_budget"] = self.thinking_budget

        # Add response schema for structured output if requested
        if self.use_structured_output:
            generation_config["response_mime_type"] = "application/json"  # type: ignore[assignment]
            generation_config["response_schema"] = self._get_response_schema()  # type: ignore[assignment]

        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,  # type: ignore[arg-type]
        )

    def _get_response_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured output.

        Returns:
            JSON schema defining the expected evaluation response structure
        """
        return {
            "type": "object",
            "properties": {
                "matches_category": {
                    "type": "boolean",
                    "description": "Whether the content matches the category",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1",
                },
                "property_scores": {
                    "type": "object",
                    "description": "Scores for each characteristic property as key-value pairs",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning for the evaluation",
                },
                "similar_examples": {
                    "type": "array",
                    "description": "Similar examples from the category",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "similarity": {"type": "number"},
                        },
                    },
                },
                "feedback": {
                    "type": "string",
                    "description": "Constructive feedback for improvement",
                },
            },
            "required": [
                "matches_category",
                "confidence",
                "property_scores",
                "reasoning",
            ],
        }

    def _create_evaluation_prompt(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> str:
        """Create evaluation prompt for Gemini.

        Args:
            content: Content to evaluate
            category: Category definition
            examples: Optional examples for few-shot learning

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert evaluator determining if content matches a specific category.

Category: {category.name}
Description: {category.description}

Characteristic Properties:
"""
        for prop in category.characteristic_properties:
            prompt += f"- {prop.name} ({prop.property_type.value}): {prop.formal_definition}\n"
            if prop.threshold:
                prompt += f"  Threshold: {prop.threshold}\n"

        if examples:
            prompt += "\n\nExamples:\n"
            for i, ex in enumerate(examples[:3], 1):
                prompt += f"{i}. {ex.content[:200]}...\n"
                if ex.annotations:
                    prompt += f"   Note: {ex.annotations.get('note', '')}\n"

        prompt += f"""

Content to Evaluate:
{content}

Evaluate whether this content matches the category based on the characteristic properties.
Provide detailed reasoning and scores for each property.
"""

        if not self.use_structured_output:
            prompt += """
Return your evaluation as a JSON object with these fields:
- matches_category (boolean)
- confidence (0-1)
- property_scores (object with property names as keys, scores 0-1 as values)
- reasoning (detailed explanation)
- similar_examples (array of similar examples if any)
- feedback (suggestions for improvement)
"""

        return prompt

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """Evaluate content against a category using Gemini.

        Args:
            content: Content to evaluate
            category: Category to evaluate against
            examples: Optional examples for few-shot learning

        Returns:
            ProviderResult with evaluation details

        Raises:
            Exception: If Gemini API call fails
        """
        try:
            # Create the evaluation prompt
            prompt = self._create_evaluation_prompt(content, category, examples)

            # Generate response
            response = self.client.generate_content(prompt)

            # Parse response based on output mode
            if self.use_structured_output:
                # Response is already structured JSON
                result_data = json.loads(response.text)
            else:
                # Try to extract JSON from text response
                result_data = self._extract_json_from_text(response.text)

            # Calculate token usage for cost tracking
            usage = self._calculate_usage(prompt, response.text)

            return ProviderResult(
                category_match=result_data.get("matches_category", False),
                confidence=result_data.get("confidence", 0.0),
                scores=result_data.get("property_scores", {}),
                reasoning=result_data.get("reasoning", ""),
                raw_response=response.text,
                model=self.model,
                provider="gemini",
                metadata={
                    "usage": usage,
                    "similar_examples": result_data.get("similar_examples", []),
                    "feedback": result_data.get("feedback", ""),
                },
            )

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")
            # Return a default result on error
            return ProviderResult(
                category_match=False,
                confidence=0.0,
                scores={},
                reasoning=f"Evaluation failed: {str(e)}",
                raw_response=str(e),
                model=self.model,
                provider="gemini",
                metadata={"error": str(e)},
            )

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response.

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed JSON dictionary
        """
        # First, try to parse the entire text as JSON
        try:
            result: Dict[str, Any] = json.loads(text)
            return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re

        # Look for ```json blocks
        json_block_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed_json: Dict[str, Any] = json.loads(match)
                return parsed_json
            except json.JSONDecodeError:
                continue

        # Try to find JSON object with balanced braces
        json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed_obj: Dict[str, Any] = json.loads(match)
                return parsed_obj
            except json.JSONDecodeError:
                continue

        # If no valid JSON found, return defaults
        logger.warning("Could not extract JSON from Gemini response")
        return {
            "matches_category": False,
            "confidence": 0.0,
            "property_scores": {},
            "reasoning": text,
        }

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, Any]:
        """Calculate token usage and cost.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Usage statistics including tokens and cost
        """
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        # Get costs for this model
        costs = self.MODEL_COSTS.get(
            self.model, (1.0, 3.0)
        )  # Default costs if model unknown
        input_cost_per_million = costs[0]
        output_cost_per_million = costs[1]

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": self.model,
            "provider": "gemini",
        }

    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
    ) -> List[ProviderResult]:
        """Evaluate multiple contents efficiently.

        Gemini supports batch processing for better efficiency.

        Args:
            contents: List of contents to evaluate
            category: Category to evaluate against

        Returns:
            List of evaluation results
        """
        results = []

        # Process in batches for efficiency
        batch_size = 5  # Gemini can handle multiple evaluations efficiently
        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]

            # Evaluate batch concurrently
            import asyncio

            batch_results = await asyncio.gather(
                *[self.evaluate(content, category) for content in batch]
            )
            results.extend(batch_results)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model capabilities and costs
        """
        costs = self.MODEL_COSTS.get(self.model, (1.0, 3.0))

        return {
            "provider": "gemini",
            "model": self.model,
            "supports_structured_output": True,
            "supports_thinking_budget": "2.5" in self.model or "2.0" in self.model,
            "max_tokens": 8192,
            "input_cost_per_million": costs[0],
            "output_cost_per_million": costs[1],
            "quality_score": self._get_quality_score(),
            "speed": self._get_speed_rating(),
        }

    def _get_quality_score(self) -> int:
        """Get quality score for the model (0-100)."""
        quality_scores = {
            "gemini-2.0-flash-exp": 92,
            "gemini-1.5-flash": 80,
            "gemini-1.5-flash-8b": 75,
            "gemini-1.5-pro": 88,
            "gemini-1.0-pro": 82,
        }
        return quality_scores.get(self.model, 85)

    def _get_speed_rating(self) -> str:
        """Get speed rating for the model."""
        if "flash" in self.model.lower():
            return "fast"
        elif "pro" in self.model.lower():
            return "medium"
        return "medium"
