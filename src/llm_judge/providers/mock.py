"""Enhanced mock provider for testing without API calls.

This module provides a sophisticated mock provider that simulates realistic
LLM behavior for testing and development purposes.
"""

import asyncio
import random
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from llm_judge.providers.base import JudgeProvider, ProviderResult
from llm_judge.core.category import CategoryDefinition, Example, PropertyType


@dataclass
class MockConfig:
    """Configuration for mock provider behavior.

    Attributes:
        response_delay: Simulated API delay in seconds
        failure_rate: Probability of simulated failures (0-1)
        quality_mode: Quality level of responses (low, medium, high)
        model_behavior: Which real model to simulate
        deterministic: Whether to use deterministic responses (for testing)
    """
    response_delay: float = 0.1
    failure_rate: float = 0.0
    quality_mode: str = "medium"
    model_behavior: str = "gpt-4"
    deterministic: bool = False


class MockProvider(JudgeProvider):
    """Enhanced mock provider for testing without API costs.

    This provider simulates realistic LLM behavior including:
    - Variable response quality based on configuration
    - Simulated API delays
    - Deterministic mode for consistent testing
    - Cost simulation based on model type
    - Realistic error scenarios
    """

    # Simulated model behaviors and costs
    MODEL_PROFILES = {
        "gpt-4": {
            "quality": 0.95,
            "speed": "slow",
            "cost_per_1k": 0.03,
            "typical_confidence": 0.85,
        },
        "gpt-3.5-turbo": {
            "quality": 0.75,
            "speed": "fast",
            "cost_per_1k": 0.001,
            "typical_confidence": 0.70,
        },
        "claude-3-opus": {
            "quality": 0.96,
            "speed": "medium",
            "cost_per_1k": 0.015,
            "typical_confidence": 0.88,
        },
        "gemini-1.5-flash": {
            "quality": 0.80,
            "speed": "very-fast",
            "cost_per_1k": 0.0001,
            "typical_confidence": 0.75,
        },
    }

    def __init__(
        self,
        model: str = "mock-gpt-4",
        temperature: float = 0.1,
        config: Optional[MockConfig] = None,
        **kwargs: Any,
    ):
        """Initialize enhanced mock provider.

        Args:
            model: Mock model name
            temperature: Temperature setting (affects response variability)
            config: Mock configuration settings
            **kwargs: Additional provider arguments
        """
        super().__init__(model, temperature, **kwargs)
        self._mock_config = config or MockConfig()
        # Keep self.config as dict for backward compatibility with type checker
        self.config: Dict[str, Any] = {
            "response_delay": self._mock_config.response_delay,
            "failure_rate": self._mock_config.failure_rate,
            "quality_mode": self._mock_config.quality_mode,
            "model_behavior": self._mock_config.model_behavior,
            "deterministic": self._mock_config.deterministic,
        }

        # Determine which model to simulate
        self.simulated_model = self._get_simulated_model(model)
        self.model_profile = self.MODEL_PROFILES.get(
            self.simulated_model,
            self.MODEL_PROFILES["gpt-3.5-turbo"]
        )

        # Response cache for deterministic mode
        self.response_cache: Dict[str, ProviderResult] = {}

    def _get_simulated_model(self, model: str) -> str:
        """Extract the model being simulated from the mock model name."""
        if "gpt-4" in model.lower():
            return "gpt-4"
        elif "gpt-3" in model.lower() or "turbo" in model.lower():
            return "gpt-3.5-turbo"
        elif "claude" in model.lower() or "opus" in model.lower():
            return "claude-3-opus"
        elif "gemini" in model.lower() or "flash" in model.lower():
            return "gemini-1.5-flash"
        else:
            return self.config["model_behavior"]

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """Simulate evaluation with realistic behavior.

        Args:
            content: Content to evaluate
            category: Category to evaluate against
            examples: Optional examples for few-shot learning

        Returns:
            Simulated ProviderResult with realistic scores and reasoning
        """
        # Check cache in deterministic mode
        if self.config["deterministic"]:
            cache_key = self._get_cache_key(content, category)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

        # Simulate API delay
        await self._simulate_delay()

        # Simulate failures
        if await self._should_fail():
            return self._create_error_result()

        # Generate realistic evaluation
        result = self._generate_evaluation(content, category, examples)

        # Cache in deterministic mode
        if self.config["deterministic"]:
            self.response_cache[cache_key] = result

        return result

    def _get_cache_key(self, content: str, category: CategoryDefinition) -> str:
        """Generate cache key for deterministic responses."""
        key_str = f"{content}:{category.name}:{self.model}:{self.temperature}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _simulate_delay(self):
        """Simulate realistic API delay based on model speed."""
        base_delay = self.config["response_delay"]

        speed_multipliers = {
            "very-fast": 0.5,
            "fast": 1.0,
            "medium": 1.5,
            "slow": 2.0,
        }

        speed = self.model_profile.get("speed", "medium")
        multiplier = speed_multipliers.get(speed, 1.0)

        # Add some variability unless in deterministic mode
        if not self.config["deterministic"]:
            multiplier *= random.uniform(0.8, 1.2)

        await asyncio.sleep(base_delay * multiplier)

    async def _should_fail(self) -> bool:
        """Determine if this request should simulate a failure."""
        if self.config["deterministic"]:
            return False
        return random.random() < self.config["failure_rate"]

    def _create_error_result(self) -> ProviderResult:
        """Create a realistic error result."""
        error_messages = [
            "Rate limit exceeded. Please try again later.",
            "Model overloaded. Request timeout.",
            "Invalid API response format.",
            "Context length exceeded.",
        ]

        error_msg = random.choice(error_messages)

        return ProviderResult(
            category_match=False,
            confidence=0.0,
            scores={},
            reasoning=f"Evaluation failed: {error_msg}",
            raw_response="",
            model=self.model,
            provider="mock",
            error=error_msg,
        )

    def _generate_evaluation(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]],
    ) -> ProviderResult:
        """Generate realistic evaluation based on content analysis."""
        # Analyze content characteristics
        content_lower = content.lower()
        word_count = len(content.split())

        # Base quality from model profile
        base_quality = self.model_profile["quality"]

        # Evaluate each property
        property_scores = {}
        total_weight = 0
        weighted_score = 0

        for prop in category.characteristic_properties:
            # Simulate property evaluation with some intelligence
            score = self._evaluate_property(content, prop, base_quality)
            property_scores[prop.name] = score

            # Weight the scores
            weight = prop.weight if prop.weight else 1.0
            total_weight += weight
            weighted_score += score * weight

        # Calculate overall match score
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.5

        # Add quality-based noise unless deterministic
        if not self.config["deterministic"]:
            noise = random.gauss(0, (1 - base_quality) * 0.1)
            overall_score = max(0, min(1, overall_score + noise))

        # Determine match based on threshold
        matches = overall_score > 0.5

        # Generate confidence based on model profile and score extremity
        base_confidence = self.model_profile["typical_confidence"]
        if self.config["deterministic"]:
            confidence = base_confidence
        else:
            # Higher confidence for extreme scores
            extremity = abs(overall_score - 0.5) * 2
            confidence = base_confidence * (0.7 + 0.3 * extremity)
            confidence = max(0.3, min(0.95, confidence + random.gauss(0, 0.05)))

        # Generate realistic reasoning
        reasoning = self._generate_reasoning(
            content, category, property_scores, matches, overall_score
        )

        # Generate feedback if not matching
        feedback = None
        if not matches:
            feedback = self._generate_feedback(property_scores, category)

        # Find similar examples (simulate)
        similar_examples = []
        if examples and len(examples) > 0:
            # Pick 1-2 random examples as "similar"
            num_similar = min(2, len(examples))
            for ex in random.sample(examples, num_similar):
                similar_examples.append({
                    "content": ex.content[:100] + "...",
                    "similarity": random.uniform(0.6, 0.9),
                })

        # Calculate simulated cost
        input_tokens = len(content) // 4 + 200  # Rough estimate
        output_tokens = len(reasoning) // 4 + 50
        total_tokens = input_tokens + output_tokens

        cost_per_1k = self.model_profile.get("cost_per_1k", 0.001)
        total_cost = (total_tokens / 1000) * cost_per_1k

        # Create usage statistics
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": (input_tokens / 1000) * cost_per_1k * 0.3,
            "output_cost": (output_tokens / 1000) * cost_per_1k * 0.7,
            "total_cost": total_cost,
            "model": self.model,
            "provider": "mock",
        }

        return ProviderResult(
            category_match=matches,
            confidence=confidence,
            scores=property_scores,
            reasoning=reasoning,
            raw_response=f"Mock response for {category.name} evaluation",
            model=self.model,
            provider="mock",
            tokens_used=total_tokens,
            cost=total_cost,
            usage=usage,
            similar_examples=similar_examples,
            feedback=feedback,
            metadata={
                "simulated_model": self.simulated_model,
                "quality_mode": self.config["quality_mode"],
                "word_count": word_count,
            },
        )

    def _evaluate_property(
        self,
        content: str,
        prop: Any,
        base_quality: float,
    ) -> float:
        """Simulate intelligent property evaluation."""
        # Use property's measurement function if available
        if hasattr(prop, 'measurement_function') and prop.measurement_function:
            try:
                measured_value = prop.measurement_function(content)
                if hasattr(prop, 'meets_threshold'):
                    if prop.meets_threshold(measured_value):
                        return random.uniform(0.7, 1.0) * base_quality
                    else:
                        return random.uniform(0.0, 0.4) * base_quality
            except:
                pass

        # Fallback to keyword-based simulation
        content_lower = content.lower()
        prop_name_lower = prop.name.lower()

        # Check for property-related keywords
        keywords_found = 0
        keywords = prop_name_lower.split('_') + [prop_name_lower]
        for keyword in keywords:
            if keyword in content_lower:
                keywords_found += 1

        if keywords_found > 0:
            base_score = 0.6 + (0.2 * min(keywords_found, 2))
        else:
            base_score = 0.3

        # Apply quality factor and add controlled randomness
        score = base_score * base_quality
        if not self.config["deterministic"]:
            score += random.gauss(0, 0.1)

        return max(0, min(1, score))

    def _generate_reasoning(
        self,
        content: str,
        category: CategoryDefinition,
        scores: Dict[str, float],
        matches: bool,
        overall_score: float,
    ) -> str:
        """Generate realistic reasoning for the evaluation."""
        reasoning_parts = []

        # Opening assessment
        if matches:
            reasoning_parts.append(
                f"The content appears to match the '{category.name}' category "
                f"with an overall score of {overall_score:.2f}."
            )
        else:
            reasoning_parts.append(
                f"The content does not sufficiently match the '{category.name}' category "
                f"(score: {overall_score:.2f})."
            )

        # Property analysis
        high_scoring = [p for p, s in scores.items() if s > 0.7]
        low_scoring = [p for p, s in scores.items() if s < 0.3]

        if high_scoring:
            reasoning_parts.append(
                f"Strong alignment found with: {', '.join(high_scoring)}."
            )

        if low_scoring:
            reasoning_parts.append(
                f"Weak alignment with: {', '.join(low_scoring)}."
            )

        # Add quality-specific insights
        if self.config["quality_mode"] == "high":
            word_count = len(content.split())
            reasoning_parts.append(
                f"Content analysis shows {word_count} words with "
                f"{len(scores)} properties evaluated."
            )

        return " ".join(reasoning_parts)

    def _generate_feedback(
        self,
        scores: Dict[str, float],
        category: CategoryDefinition,
    ) -> str:
        """Generate improvement feedback for non-matching content."""
        feedback_parts = ["To better match this category:"]

        # Find lowest scoring properties
        sorted_props = sorted(scores.items(), key=lambda x: x[1])

        for prop_name, score in sorted_props[:3]:
            if score < 0.5:
                # Find the property definition
                prop_def = next(
                    (p for p in category.characteristic_properties if p.name == prop_name),
                    None
                )
                if prop_def:
                    feedback_parts.append(
                        f"- Improve '{prop_name}': {prop_def.formal_definition}"
                    )

        return "\n".join(feedback_parts)

    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
    ) -> List[ProviderResult]:
        """Evaluate multiple contents with simulated batch processing.

        Args:
            contents: List of contents to evaluate
            category: Category to evaluate against

        Returns:
            List of evaluation results
        """
        # Simulate batch processing with some efficiency gain
        results = []

        # Reduce delay for batch processing
        original_delay = self.config["response_delay"]
        self.config["response_delay"] = original_delay * 0.7

        try:
            for content in contents:
                result = await self.evaluate(content, category)
                results.append(result)
        finally:
            # Restore original delay
            self.config["response_delay"] = original_delay

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model.

        Returns:
            Dictionary with model capabilities and simulated costs
        """
        return {
            "provider": "mock",
            "model": self.model,
            "simulated_model": self.simulated_model,
            "supports_streaming": False,
            "max_tokens": 4096,
            "cost_per_1k_tokens": self.model_profile.get("cost_per_1k", 0.001),
            "quality_score": int(self.model_profile["quality"] * 100),
            "speed": self.model_profile["speed"],
            "config": {
                "response_delay": self.config["response_delay"],
                "failure_rate": self.config["failure_rate"],
                "quality_mode": self.config["quality_mode"],
                "deterministic": self.config["deterministic"],
            },
        }