"""
Evaluation engine and result classes for the LLM-as-Judge framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_judge.core.category import CategoryDefinition, Example


@dataclass
class PropertyScore:
    """Score for a single characteristic property."""

    property_name: str
    value: Any
    meets_threshold: bool
    weight: float
    raw_score: float = 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating content against a category."""

    category: str
    membership_score: float  # 0-1 score for category membership
    property_scores: Dict[str, PropertyScore]
    similar_examples: List[Tuple[Example, float]]  # (example, similarity_score)
    feedback: str = ""
    confidence: float = 1.0
    model_used: Optional[str] = None
    evaluation_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def matches_category(self) -> bool:
        """Check if content matches the category (membership_score > 0.5).

        Returns:
            True if the membership score is greater than 0.5, indicating the
            content likely belongs to the category.
        """
        return self.membership_score > 0.5

    @property
    def failed_properties(self) -> List[str]:
        """Get list of properties that failed to meet threshold.

        Returns:
            List of property names that did not meet their threshold requirements.
        """
        return [
            name
            for name, score in self.property_scores.items()
            if not score.meets_threshold
        ]

    @property
    def passed_properties(self) -> List[str]:
        """Get list of properties that met threshold.

        Returns:
            List of property names that successfully met their threshold requirements.
        """
        return [
            name
            for name, score in self.property_scores.items()
            if score.meets_threshold
        ]

    def generate_improvement_suggestions(self) -> str:
        """Generate suggestions for improving category membership.

        Returns:
            A formatted string with actionable suggestions for improving
            category membership, or a confirmation if all requirements are met.
        """
        suggestions = []

        for prop_name, score in self.property_scores.items():
            if not score.meets_threshold:
                suggestions.append(
                    f"- Improve {prop_name}: current value {score.value}, "
                    f"needs to meet threshold"
                )

        if suggestions:
            return "To improve category membership:\n" + "\n".join(suggestions)
        return "Content meets all category requirements."


@dataclass
class ComparisonResult:
    """Result of comparing two pieces of content."""

    overall_improvement: bool
    property_improvements: Dict[str, Dict[str, Any]]
    recommendation: str
    original_score: float
    modified_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percentage(self) -> float:
        """Calculate percentage improvement."""
        if self.original_score == 0:
            return 100.0 if self.modified_score > 0 else 0.0
        return ((self.modified_score - self.original_score) / self.original_score) * 100

    @property
    def improved_properties(self) -> List[str]:
        """Get list of properties that improved."""
        return [
            prop
            for prop, delta in self.property_improvements.items()
            if delta.get("improved", False)
        ]

    @property
    def regressed_properties(self) -> List[str]:
        """Get list of properties that regressed."""
        return [
            prop
            for prop, delta in self.property_improvements.items()
            if not delta.get("improved", True) and delta.get("delta", 0) < 0
        ]

    @property
    def missing_properties(self) -> List[str]:
        """Get list of properties still not meeting threshold."""
        return [
            prop
            for prop, delta in self.property_improvements.items()
            if not delta.get("meets_threshold", False)
        ]


class EvaluationEngine:
    """Main evaluation engine for assessing content against categories.

    This engine provides the core functionality for evaluating content against
    category definitions using characteristic properties, example similarity,
    and evaluation rubrics.

    Attributes:
        cache: Dictionary for caching evaluation results.
        evaluation_history: List of all evaluation results for analysis.
    """

    def __init__(self):
        """Initialize the evaluation engine.

        Sets up empty cache and evaluation history tracking.
        """
        self.cache = {}
        self.evaluation_history = []

    def evaluate(self, content: str, category: CategoryDefinition) -> EvaluationResult:
        """
        Evaluate content against a category definition.

        Args:
            content: The content to evaluate
            category: The category definition to evaluate against

        Returns:
            EvaluationResult with scores and feedback
        """
        # 1. Measure all characteristic properties
        property_scores = self._measure_properties(content, category)

        # 2. Compare with examples
        example_similarity = self._compute_example_similarity(
            content, category.positive_examples, category.negative_examples
        )

        # 3. Apply category-specific rubric
        rubric_score = 0.0
        if category.evaluation_rubric:
            # Get individual scores for rubric criteria
            criteria_scores = self._evaluate_criteria(content, category)
            rubric_score = category.evaluation_rubric.score(content, criteria_scores)

        # 4. Compute overall category membership
        membership_score = self._compute_membership(
            property_scores, example_similarity, rubric_score
        )

        # 5. Generate feedback
        feedback = self._generate_feedback(property_scores, membership_score)

        result = EvaluationResult(
            category=category.name,
            membership_score=membership_score,
            property_scores=property_scores,
            similar_examples=example_similarity["top_similar"],
            feedback=feedback,
        )

        # Store in history
        self.evaluation_history.append(result)

        return result

    def _measure_properties(
        self, content: str, category: CategoryDefinition
    ) -> Dict[str, PropertyScore]:
        """Measure all characteristic properties for the content.

        Args:
            content: The text content to evaluate.
            category: Category definition containing properties to measure.

        Returns:
            Dictionary mapping property names to PropertyScore objects.
            Failed measurements are recorded with null values and zero scores.
        """
        property_scores = {}

        for prop in category.characteristic_properties:
            try:
                value = prop.measure(content)
                meets_threshold = prop.meets_threshold(value)

                property_scores[prop.name] = PropertyScore(
                    property_name=prop.name,
                    value=value,
                    meets_threshold=meets_threshold,
                    weight=prop.weight,
                    raw_score=1.0 if meets_threshold else 0.0,
                )
            except Exception:
                # Handle measurement errors gracefully
                property_scores[prop.name] = PropertyScore(
                    property_name=prop.name,
                    value=None,
                    meets_threshold=False,
                    weight=prop.weight,
                    raw_score=0.0,
                )

        return property_scores

    def _compute_example_similarity(
        self,
        content: str,
        positive_examples: List[Example],
        negative_examples: List[Example],
    ) -> Dict[str, Any]:
        """Compute similarity between content and examples.

        Args:
            content: The text content to compare against examples.
            positive_examples: Examples that belong to the target category.
            negative_examples: Examples that do not belong to the target category.

        Returns:
            Dictionary containing:
                - top_similar: Top 3 most similar positive examples with scores
                - positive_avg: Average similarity to positive examples
                - negative_avg: Average similarity to negative examples

        Note:
            Uses simple word-overlap similarity. In production, this would use
            embeddings or more sophisticated similarity measures.
        """
        # Simplified similarity computation
        # In production, this would use embeddings or more sophisticated methods

        positive_similarities = []
        for example in positive_examples[:5]:  # Limit to top 5
            similarity = self._simple_similarity(content, example.content)
            positive_similarities.append((example, similarity))

        negative_similarities = []
        for example in negative_examples[:5]:  # Limit to top 5
            similarity = self._simple_similarity(content, example.content)
            negative_similarities.append((example, similarity))

        # Sort by similarity
        positive_similarities.sort(key=lambda x: x[1], reverse=True)
        negative_similarities.sort(key=lambda x: x[1], reverse=True)

        return {
            "top_similar": positive_similarities[:3],
            "positive_avg": (
                sum(s for _, s in positive_similarities) / len(positive_similarities)
                if positive_similarities
                else 0
            ),
            "negative_avg": (
                sum(s for _, s in negative_similarities) / len(negative_similarities)
                if negative_similarities
                else 0
            ),
        }

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap.

        Args:
            text1: First text string to compare.
            text2: Second text string to compare.

        Returns:
            Jaccard similarity coefficient (0.0 to 1.0) based on word overlap.
            Returns 0.0 if either text is empty.

        Examples:
            >>> engine = EvaluationEngine()
            >>> engine._simple_similarity("hello world", "hello there")
            0.33333...  # 1 common word out of 3 total unique words
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _evaluate_criteria(
        self, content: str, category: CategoryDefinition
    ) -> Dict[str, float]:
        """Evaluate individual rubric criteria.

        Args:
            content: The text content to evaluate.
            category: Category definition containing evaluation rubric.

        Returns:
            Dictionary mapping criterion names to scores (0.0-1.0).
            Returns empty dict if no rubric is defined.

        Note:
            Current implementation returns placeholder scores.
            In production, this would call specific evaluators for each criterion.
        """
        # This would be implemented based on specific criteria
        # For now, return placeholder scores
        if not category.evaluation_rubric:
            return {}

        criteria_scores = {}
        for criterion in category.evaluation_rubric.scoring_criteria:
            # Placeholder: In production, this would call specific evaluators
            criteria_scores[criterion] = 0.5

        return criteria_scores

    def _compute_membership(
        self,
        property_scores: Dict[str, PropertyScore],
        example_similarity: Dict[str, Any],
        rubric_score: float,
    ) -> float:
        """Compute overall category membership score."""
        # Weighted combination of different signals

        # 1. Property score (60% weight)
        property_weight = 0.6
        total_property_weight = sum(ps.weight for ps in property_scores.values())
        if total_property_weight > 0:
            property_score = (
                sum(ps.raw_score * ps.weight for ps in property_scores.values())
                / total_property_weight
            )
        else:
            property_score = 0.0

        # 2. Example similarity (20% weight)
        example_weight = 0.2
        positive_sim = example_similarity.get("positive_avg", 0)
        negative_sim = example_similarity.get("negative_avg", 0)
        example_score = max(0, positive_sim - negative_sim)

        # 3. Rubric score (20% weight)
        rubric_weight = 0.2

        # Combine scores
        membership = (
            property_score * property_weight
            + example_score * example_weight
            + rubric_score * rubric_weight
        )

        return min(1.0, max(0.0, membership))

    def _generate_feedback(
        self, property_scores: Dict[str, PropertyScore], membership_score: float
    ) -> str:
        """Generate human-readable feedback about the evaluation."""
        feedback_parts = []

        # Overall assessment
        if membership_score > 0.7:
            feedback_parts.append("Content strongly matches the category.")
        elif membership_score > 0.5:
            feedback_parts.append("Content moderately matches the category.")
        elif membership_score > 0.3:
            feedback_parts.append("Content weakly matches the category.")
        else:
            feedback_parts.append("Content does not match the category well.")

        # Property feedback
        failed_props = [
            name for name, score in property_scores.items() if not score.meets_threshold
        ]

        if failed_props:
            feedback_parts.append(f"Failed properties: {', '.join(failed_props)}")

        passed_props = [
            name for name, score in property_scores.items() if score.meets_threshold
        ]

        if passed_props:
            feedback_parts.append(
                f"Met properties: {', '.join(passed_props[:3])}"
                + ("..." if len(passed_props) > 3 else "")
            )

        return " ".join(feedback_parts)


class ComparisonEvaluator:
    """Evaluator for comparing content before and after modifications.

    This class provides functionality to compare two versions of content
    (original vs. modified) against the same category to determine improvements
    or regressions in category membership.

    Attributes:
        engine: EvaluationEngine instance used for individual evaluations.
    """

    def __init__(self, engine: Optional[EvaluationEngine] = None):
        """Initialize the comparison evaluator.

        Args:
            engine: Optional EvaluationEngine instance. If None, creates a new one.
        """
        self.engine = engine or EvaluationEngine()

    def compare_before_after(
        self,
        original: str,
        modified: str,
        category: CategoryDefinition,
    ) -> ComparisonResult:
        """
        Compare original and modified content against a category.

        Args:
            original: Original content
            modified: Modified content
            category: Category to evaluate against

        Returns:
            ComparisonResult with detailed comparison
        """
        # Evaluate both versions
        original_eval = self.engine.evaluate(original, category)
        modified_eval = self.engine.evaluate(modified, category)

        # Compute improvements/regressions for each property
        property_deltas = {}
        for prop_name in original_eval.property_scores:
            orig_score = original_eval.property_scores[prop_name]
            mod_score = modified_eval.property_scores[prop_name]

            delta = 0
            if isinstance(orig_score.value, (int, float)) and isinstance(
                mod_score.value, (int, float)
            ):
                delta = mod_score.value - orig_score.value  # type: ignore[assignment]

            property_deltas[prop_name] = {
                "delta": delta,
                "improved": mod_score.meets_threshold
                and not orig_score.meets_threshold,
                "significant": (
                    abs(delta) > 0.1 if isinstance(delta, float) else delta != 0
                ),
                "meets_threshold": mod_score.meets_threshold,
                "original_value": orig_score.value,
                "modified_value": mod_score.value,
            }

        # Generate recommendation
        recommendation = self._generate_recommendation(
            property_deltas, original_eval, modified_eval
        )

        return ComparisonResult(
            overall_improvement=modified_eval.membership_score
            > original_eval.membership_score,
            property_improvements=property_deltas,
            recommendation=recommendation,
            original_score=original_eval.membership_score,
            modified_score=modified_eval.membership_score,
        )

    def _generate_recommendation(
        self,
        property_deltas: Dict[str, Dict[str, Any]],
        original_eval: EvaluationResult,
        modified_eval: EvaluationResult,
    ) -> str:
        """Generate recommendation based on comparison."""
        if modified_eval.membership_score > original_eval.membership_score + 0.1:
            return "Significant improvement detected. Modified version recommended."
        elif modified_eval.membership_score > original_eval.membership_score:
            return "Minor improvement detected. Modified version slightly better."
        elif modified_eval.membership_score < original_eval.membership_score - 0.1:
            return "Regression detected. Original version recommended."
        else:
            return "No significant difference. Both versions comparable."
