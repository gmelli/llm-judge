"""
Consensus mechanisms for multi-provider evaluations.
"""

import asyncio
from enum import Enum
from typing import List, Optional

from llm_judge.core.category import CategoryDefinition
from llm_judge.core.evaluation import EvaluationResult, PropertyScore
from llm_judge.providers.base import JudgeProvider


class ConsensusMode(Enum):
    """Modes for reaching consensus among multiple judges."""

    UNANIMOUS = "unanimous"  # All judges must agree
    MAJORITY = "majority"  # Simple majority
    WEIGHTED = "weighted"  # Weighted average
    STRICT = "strict"  # At least 75% agreement


class ConsensusJudge:
    """Handles consensus among multiple LLM judges."""

    def __init__(
        self,
        providers: List[JudgeProvider],
        consensus_mode: ConsensusMode = ConsensusMode.MAJORITY,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize consensus judge.

        Args:
            providers: List of judge providers
            consensus_mode: How to reach consensus
            weights: Optional weights for weighted consensus
        """
        self.providers = providers
        self.consensus_mode = consensus_mode
        self.weights = weights or [1.0] * len(providers)

        if len(self.weights) != len(self.providers):
            raise ValueError("Number of weights must match number of providers")

    async def evaluate(
        self, content: str, category: CategoryDefinition
    ) -> EvaluationResult:
        """
        Evaluate content using multiple providers and reach consensus.

        Args:
            content: The content to evaluate
            category: The category definition

        Returns:
            Consensus EvaluationResult
        """
        # Get evaluations from all providers
        evaluations = await asyncio.gather(*[
            provider.evaluate(content, category)
            for provider in self.providers
        ])

        # Apply consensus strategy
        if self.consensus_mode == ConsensusMode.UNANIMOUS:
            return self._unanimous_consensus(evaluations, category)
        elif self.consensus_mode == ConsensusMode.MAJORITY:
            return self._majority_vote(evaluations, category)
        elif self.consensus_mode == ConsensusMode.WEIGHTED:
            return self._weighted_average(evaluations, category)
        elif self.consensus_mode == ConsensusMode.STRICT:
            return self._strict_consensus(evaluations, category)
        else:
            raise ValueError(f"Unknown consensus mode: {self.consensus_mode}")

    def _unanimous_consensus(
        self, evaluations: List, category: CategoryDefinition
    ) -> EvaluationResult:
        """All judges must agree for positive classification."""
        # Check if all agree on category match
        all_match = all(eval.category_match for eval in evaluations)

        # Average confidence only if all agree
        avg_confidence = (
            sum(eval.confidence for eval in evaluations) / len(evaluations)
            if all_match else 0.0
        )

        # Combine scores
        combined_scores = self._combine_scores(evaluations)

        # Generate feedback
        if all_match:
            feedback = f"All {len(evaluations)} judges agree: content matches category"
        else:
            dissenting = sum(1 for eval in evaluations if not eval.category_match)
            feedback = f"{dissenting} judge(s) dissent: content does not match category"

        return EvaluationResult(
            category=category.name,
            membership_score=1.0 if all_match else 0.0,
            property_scores=combined_scores,
            similar_examples=[],
            feedback=feedback,
            confidence=avg_confidence,
            metadata={
                "consensus_mode": "unanimous",
                "num_judges": len(evaluations),
                "agreement": all_match,
            }
        )

    def _majority_vote(
        self, evaluations: List, category: CategoryDefinition
    ) -> EvaluationResult:
        """Simple majority determines classification."""
        # Count votes
        positive_votes = sum(1 for eval in evaluations if eval.category_match)
        majority_threshold = len(evaluations) / 2

        matches = positive_votes > majority_threshold

        # Average confidence
        avg_confidence = sum(eval.confidence for eval in evaluations) / len(evaluations)

        # Combine scores
        combined_scores = self._combine_scores(evaluations)

        # Generate feedback
        feedback = (
            f"{positive_votes}/{len(evaluations)} judges vote: "
            f"content {'matches' if matches else 'does not match'} category"
        )

        return EvaluationResult(
            category=category.name,
            membership_score=positive_votes / len(evaluations),
            property_scores=combined_scores,
            similar_examples=[],
            feedback=feedback,
            confidence=avg_confidence,
            metadata={
                "consensus_mode": "majority",
                "num_judges": len(evaluations),
                "positive_votes": positive_votes,
                "agreement": matches,
            }
        )

    def _weighted_average(
        self, evaluations: List, category: CategoryDefinition
    ) -> EvaluationResult:
        """Weighted average of all judge scores."""
        # Calculate weighted scores
        total_weight = sum(self.weights)

        weighted_match_score = sum(
            (1.0 if eval.category_match else 0.0) * weight
            for eval, weight in zip(evaluations, self.weights)
        ) / total_weight

        weighted_confidence = sum(
            eval.confidence * weight
            for eval, weight in zip(evaluations, self.weights)
        ) / total_weight

        # Combine scores with weights
        combined_scores = self._combine_scores(evaluations, use_weights=True)

        # Generate feedback
        feedback = (
            f"Weighted consensus ({len(evaluations)} judges): "
            f"membership score = {weighted_match_score:.2f}"
        )

        return EvaluationResult(
            category=category.name,
            membership_score=weighted_match_score,
            property_scores=combined_scores,
            similar_examples=[],
            feedback=feedback,
            confidence=weighted_confidence,
            metadata={
                "consensus_mode": "weighted",
                "num_judges": len(evaluations),
                "weights": self.weights,
            }
        )

    def _strict_consensus(
        self, evaluations: List, category: CategoryDefinition
    ) -> EvaluationResult:
        """Requires 75% or more agreement for positive classification."""
        # Count positive votes
        positive_votes = sum(1 for eval in evaluations if eval.category_match)
        agreement_ratio = positive_votes / len(evaluations)
        matches = agreement_ratio >= 0.75

        # Average confidence
        avg_confidence = sum(eval.confidence for eval in evaluations) / len(evaluations)

        # Combine scores
        combined_scores = self._combine_scores(evaluations)

        # Generate feedback
        feedback = (
            f"Strict consensus ({positive_votes}/{len(evaluations)} agree): "
            f"content {'matches' if matches else 'does not match'} category"
        )

        return EvaluationResult(
            category=category.name,
            membership_score=agreement_ratio,
            property_scores=combined_scores,
            similar_examples=[],
            feedback=feedback,
            confidence=avg_confidence,
            metadata={
                "consensus_mode": "strict",
                "num_judges": len(evaluations),
                "agreement_ratio": agreement_ratio,
                "agreement": matches,
            }
        )

    def _combine_scores(
        self, evaluations: List, use_weights: bool = False
    ) -> dict:
        """Combine property scores from multiple evaluations."""
        combined = {}

        # Collect all property names
        all_properties = set()
        for eval in evaluations:
            if hasattr(eval, 'scores') and eval.scores:
                all_properties.update(eval.scores.keys())

        # Average scores for each property
        for prop in all_properties:
            scores = []
            weights_to_use = []

            for i, eval in enumerate(evaluations):
                if hasattr(eval, 'scores') and prop in eval.scores:
                    scores.append(eval.scores[prop])
                    if use_weights:
                        weights_to_use.append(self.weights[i])
                    else:
                        weights_to_use.append(1.0)

            if scores:
                if use_weights:
                    avg_score = sum(s * w for s, w in zip(scores, weights_to_use)) / sum(weights_to_use)
                else:
                    avg_score = sum(scores) / len(scores)

                combined[prop] = PropertyScore(
                    property_name=prop,
                    value=avg_score,
                    meets_threshold=avg_score > 0.5,  # Simple threshold
                    weight=1.0,
                    raw_score=avg_score,
                )

        return combined