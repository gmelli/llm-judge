"""
Core category definitions for the LLM-as-Judge framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class PropertyType(Enum):
    """Types of characteristic properties for category membership."""

    NECESSARY = "necessary"  # Must be present for category membership
    SUFFICIENT = "sufficient"  # Alone determines category membership
    TYPICAL = "typical"  # Usually present but not required


@dataclass
class Range:
    """Represents a range of acceptable values."""

    min: Optional[float] = None
    max: Optional[float] = None

    def contains(self, value: float) -> bool:
        """Check if a value is within the range."""
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True


@dataclass
class CharacteristicProperty:
    """
    Formal property that must hold for category membership.
    """

    name: str  # e.g., "citation_density"
    property_type: PropertyType  # NECESSARY, SUFFICIENT, TYPICAL
    formal_definition: str  # Mathematical/logical definition
    measurement_function: Callable[[str], Any]  # How to measure this property
    threshold: Union[float, Range]  # Acceptable values
    weight: float = 1.0  # Importance in category determination

    def measure(self, content: str) -> Any:
        """Measure the property value for given content."""
        return self.measurement_function(content)

    def meets_threshold(self, value: Any) -> bool:
        """Check if a value meets the threshold requirement."""
        if isinstance(self.threshold, Range):
            return self.threshold.contains(value)
        elif isinstance(self.threshold, (int, float)):
            return value >= self.threshold
        elif isinstance(self.threshold, bool):
            return value == self.threshold
        else:
            return value == self.threshold


@dataclass
class Example:
    """
    Concrete example with annotations.
    """

    content: str  # The actual example
    category: str  # Which category it belongs to
    properties: Dict[str, Any] = field(default_factory=dict)  # Measured property values
    annotations: Dict[str, str] = field(default_factory=dict)  # Human explanations
    confidence: float = 1.0  # How exemplary this example is (0-1)
    source: str = ""  # Where example came from


@dataclass
class EvaluationRubric:
    """
    Category-specific evaluation criteria.
    """

    name: str
    description: str
    scoring_criteria: Dict[str, float]  # Criteria and their weights
    scale: tuple = (0, 1)  # Min and max scores

    def score(self, content: str, scores: Dict[str, float]) -> float:
        """
        Calculate overall score based on individual criteria scores.

        Args:
            content: The content to evaluate
            scores: Individual scores for each criterion

        Returns:
            Weighted average score normalized to scale
        """
        if not scores:
            return self.scale[0]

        total_weight = sum(self.scoring_criteria.values())
        if total_weight == 0:
            return self.scale[0]

        weighted_sum = sum(
            scores.get(criterion, 0) * weight
            for criterion, weight in self.scoring_criteria.items()
        )

        # Normalize to scale
        raw_score = weighted_sum / total_weight
        return self.scale[0] + raw_score * (self.scale[1] - self.scale[0])


@dataclass
class CategoryDefinition:
    """
    Defines a category with formal characteristic properties and examples.
    """

    name: str  # e.g., "Academic Writing"
    description: str  # Human-readable description
    characteristic_properties: List[CharacteristicProperty]  # Formal properties
    positive_examples: List[Example] = field(default_factory=list)  # Exemplars
    negative_examples: List[Example] = field(default_factory=list)  # Counter-examples
    parent_category: Optional[str] = None  # Hierarchical organization
    evaluation_rubric: Optional[EvaluationRubric] = None  # Category-specific criteria
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def get_necessary_properties(self) -> List[CharacteristicProperty]:
        """Get all necessary properties for this category."""
        return [
            prop
            for prop in self.characteristic_properties
            if prop.property_type == PropertyType.NECESSARY
        ]

    def get_sufficient_properties(self) -> List[CharacteristicProperty]:
        """Get all sufficient properties for this category."""
        return [
            prop
            for prop in self.characteristic_properties
            if prop.property_type == PropertyType.SUFFICIENT
        ]

    def get_typical_properties(self) -> List[CharacteristicProperty]:
        """Get all typical properties for this category."""
        return [
            prop
            for prop in self.characteristic_properties
            if prop.property_type == PropertyType.TYPICAL
        ]

    def add_property(self, property: CharacteristicProperty) -> None:
        """Add a new characteristic property to the category."""
        self.characteristic_properties.append(property)

    def add_positive_example(self, example: Example) -> None:
        """Add a positive example to the category."""
        self.positive_examples.append(example)

    def add_negative_example(self, example: Example) -> None:
        """Add a negative example to the category."""
        self.negative_examples.append(example)
