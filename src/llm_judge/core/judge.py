"""
Main Judge class that orchestrates evaluations.
"""

import asyncio
from typing import Dict, List, Optional, Union

from llm_judge.core.category import CategoryDefinition
from llm_judge.core.evaluation import (
    ComparisonEvaluator,
    ComparisonResult,
    EvaluationEngine,
    EvaluationResult,
)
from llm_judge.providers.base import JudgeProvider
from llm_judge.utils.consensus import ConsensusJudge, ConsensusMode


class Judge:
    """
    Main interface for LLM-as-Judge evaluations.
    """

    def __init__(
        self,
        provider: Optional[Union[str, JudgeProvider, List[JudgeProvider]]] = None,
        temperature: float = 0.1,
        consensus_mode: Optional[ConsensusMode] = None,
        cache_enabled: bool = True,
        **kwargs,
    ):
        """
        Initialize Judge with provider configuration.

        Args:
            provider: Provider name, instance, or list of providers
            temperature: Temperature for LLM generation
            consensus_mode: Mode for multi-provider consensus
            cache_enabled: Whether to cache evaluation results
            **kwargs: Additional provider configuration
        """
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, EvaluationResult] = {}
        self.evaluation_engine = EvaluationEngine()
        self.comparison_evaluator = ComparisonEvaluator(self.evaluation_engine)

        # Set up providers
        self.providers = self._setup_providers(provider, **kwargs)
        self.consensus_mode = consensus_mode

        # Set up consensus judge if multiple providers
        if isinstance(self.providers, list) and len(self.providers) > 1:
            self.consensus_judge = ConsensusJudge(
                self.providers,
                consensus_mode or ConsensusMode.MAJORITY,
            )
        else:
            self.consensus_judge = None

    def _setup_providers(
        self, provider: Union[str, JudgeProvider, List[JudgeProvider]], **kwargs
    ) -> Union[JudgeProvider, List[JudgeProvider]]:
        """Set up provider(s) based on input."""
        if isinstance(provider, list):
            return provider
        elif isinstance(provider, JudgeProvider):
            return provider
        elif isinstance(provider, str):
            return self._create_provider(provider, **kwargs)
        else:
            # Default to mock provider for testing
            return self._create_provider("mock", **kwargs)

    def _create_provider(self, provider_name: str, **kwargs) -> JudgeProvider:
        """Create a provider instance by name."""
        # Import providers dynamically to avoid circular imports
        provider_map = {
            "openai": "llm_judge.providers.openai.OpenAIProvider",
            "anthropic": "llm_judge.providers.anthropic.AnthropicProvider",
            "gemini": "llm_judge.providers.gemini.GeminiProvider",
            "google": "llm_judge.providers.gemini.GeminiProvider",  # Alias
            "mock": "llm_judge.providers.mock.MockProvider",
        }

        if provider_name not in provider_map:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Dynamic import
        module_path, class_name = provider_map[provider_name].rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[class_name])
            provider_class = getattr(module, class_name)
            return provider_class(temperature=self.temperature, **kwargs)
        except (ImportError, AttributeError):
            # Fall back to mock provider if import fails
            from llm_judge.providers.mock import MockProvider

            return MockProvider(temperature=self.temperature, **kwargs)

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        return_feedback: bool = True,
        use_cache: bool = None,
    ) -> EvaluationResult:
        """
        Evaluate content against a category.

        Args:
            content: The content to evaluate
            category: The category definition
            return_feedback: Whether to include detailed feedback
            use_cache: Whether to use cached results (overrides instance setting)

        Returns:
            EvaluationResult with scores and optional feedback
        """
        # Check cache
        cache_key = f"{content[:100]}_{category.name}"
        use_cache = self.cache_enabled if use_cache is None else use_cache

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Perform evaluation
        if self.consensus_judge:
            # Use consensus evaluation
            result = await self.consensus_judge.evaluate(content, category)
        else:
            # Single provider evaluation
            provider = (
                self.providers[0]
                if isinstance(self.providers, list)
                else self.providers
            )

            # Get LLM judgment
            provider_result = await provider.evaluate(content, category)

            # Combine with engine evaluation
            engine_result = self.evaluation_engine.evaluate(content, category)

            # Merge results
            result = self._merge_results(
                engine_result, provider_result, return_feedback
            )

        # Cache result
        if use_cache:
            self.cache[cache_key] = result

        return result

    def _merge_results(
        self, engine_result: EvaluationResult, provider_result, return_feedback: bool
    ) -> EvaluationResult:
        """Merge results from engine and provider evaluations."""
        # Update engine result with provider information
        engine_result.model_used = provider_result.model
        engine_result.evaluation_cost = provider_result.cost

        # Combine confidence scores
        engine_result.confidence = (
            engine_result.confidence * 0.5 + provider_result.confidence * 0.5
        )

        # Add provider reasoning to feedback if requested
        if return_feedback and provider_result.reasoning:
            engine_result.feedback += f"\n\nLLM Assessment: {provider_result.reasoning}"

        # Update metadata
        engine_result.metadata.update(
            {
                "provider_scores": provider_result.scores,
                "tokens_used": provider_result.tokens_used,
            }
        )

        return engine_result

    async def compare(
        self,
        original: str,
        enhanced: str,
        target_category: CategoryDefinition,
    ) -> ComparisonResult:
        """
        Compare original and enhanced content.

        Args:
            original: Original content
            enhanced: Enhanced/modified content
            target_category: Category to evaluate against

        Returns:
            ComparisonResult with detailed comparison
        """
        return self.comparison_evaluator.compare_before_after(
            original, enhanced, target_category
        )

    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
        max_concurrent: int = 5,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple contents in batch.

        Args:
            contents: List of contents to evaluate
            category: The category definition
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of EvaluationResults
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(content: str) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(content, category)

        # Run evaluations concurrently
        tasks = [evaluate_with_semaphore(content) for content in contents]
        results = await asyncio.gather(*tasks)

        return results

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "hits": getattr(self, "_cache_hits", 0),
            "misses": getattr(self, "_cache_misses", 0),
        }
