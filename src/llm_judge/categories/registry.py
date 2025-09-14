"""
Category registry for managing and loading category definitions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from llm_judge.core.category import (
    CategoryDefinition,
    CharacteristicProperty,
    EvaluationRubric,
    Example,
    PropertyType,
    Range,
)


class CategoryRegistry:
    """Registry for managing category definitions."""

    def __init__(self):
        self.categories: Dict[str, CategoryDefinition] = {}
        self._load_builtin_categories()

    def _load_builtin_categories(self):
        """Load built-in category definitions."""
        # Add some default categories
        self._add_academic_categories()
        self._add_technical_categories()
        self._add_safety_categories()

    def _add_academic_categories(self):
        """Add academic writing categories."""
        academic = CategoryDefinition(
            name="academic_writing",
            description="Formal academic writing with citations and scholarly tone",
            characteristic_properties=[
                CharacteristicProperty(
                    name="citation_density",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Number of citations per 100 words >= 2",
                    measurement_function=lambda c: len(
                        [w for w in c.split() if w.startswith("[")]
                    )
                    / (len(c.split()) / 100),
                    threshold=2.0,
                    weight=1.5,
                ),
                CharacteristicProperty(
                    name="formality_score",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Formal language usage score >= 0.7",
                    measurement_function=lambda c: 0.8,  # Placeholder
                    threshold=0.7,
                    weight=1.0,
                ),
            ],
            evaluation_rubric=EvaluationRubric(
                name="academic_rubric",
                description="Rubric for academic writing evaluation",
                scoring_criteria={
                    "clarity": 0.25,
                    "argumentation": 0.25,
                    "evidence": 0.25,
                    "structure": 0.25,
                },
            ),
        )
        self.register(academic)

    def _add_technical_categories(self):
        """Add technical documentation categories.

        Creates a category for technical documentation with properties
        for code blocks and technical vocabulary density.
        """
        technical = CategoryDefinition(
            name="technical_documentation",
            description="Technical documentation with code examples and precise language",
            characteristic_properties=[
                CharacteristicProperty(
                    name="code_blocks",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Contains at least one code block",
                    measurement_function=lambda c: c.count("```"),
                    threshold=1,
                    weight=1.0,
                ),
                CharacteristicProperty(
                    name="technical_terms",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Technical vocabulary >= 30% of content",
                    measurement_function=lambda c: 0.35,  # Placeholder
                    threshold=0.3,
                    weight=1.2,
                ),
            ],
        )
        self.register(technical)

    def _add_safety_categories(self):
        """Add content safety categories."""
        safe_content = CategoryDefinition(
            name="safe_content",
            description="Content safe for all audiences",
            characteristic_properties=[
                CharacteristicProperty(
                    name="toxicity_score",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Toxicity score < 0.1",
                    measurement_function=lambda c: 0.05,  # Placeholder
                    threshold=Range(max=0.1),
                    weight=2.0,
                ),
                CharacteristicProperty(
                    name="pii_present",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="No personally identifiable information",
                    measurement_function=lambda c: False,  # Placeholder
                    threshold=False,
                    weight=2.0,
                ),
            ],
        )
        self.register(safe_content)

    def register(self, category: CategoryDefinition) -> None:
        """
        Register a category definition.

        Args:
            category: The category to register
        """
        self.categories[category.name] = category

    def get(self, name: str) -> Optional[CategoryDefinition]:
        """
        Get a category by name.

        Args:
            name: Category name (supports dot notation for nested categories)

        Returns:
            CategoryDefinition or None if not found
        """
        # Support dot notation (e.g., "wiki.stub_article")
        if "." in name:
            # For now, just look for the full name
            # In a full implementation, this would traverse a hierarchy
            pass

        return self.categories.get(name)

    def list_categories(self) -> List[str]:
        """List all registered category names.

        Returns:
            List of category name strings currently in the registry.
        """
        return list(self.categories.keys())

    def load_from_yaml(self, path: Path) -> None:
        """
        Load category definitions from a YAML file.

        Args:
            path: Path to YAML file
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        for cat_name, cat_data in data.items():
            category = self._parse_category_data(cat_name, cat_data)
            self.register(category)

    def load_from_json(self, path: Path) -> None:
        """
        Load category definitions from a JSON file.

        Args:
            path: Path to JSON file
        """
        with open(path) as f:
            data = json.load(f)

        for cat_name, cat_data in data.items():
            category = self._parse_category_data(cat_name, cat_data)
            self.register(category)

    def _parse_category_data(self, name: str, data: dict) -> CategoryDefinition:
        """Parse category data from dictionary."""
        # Parse properties
        properties = []
        for prop_data in data.get("properties", []):
            prop_type = PropertyType[prop_data.get("type", "TYPICAL").upper()]

            # Parse threshold
            threshold = prop_data.get("threshold", 0.5)
            if isinstance(threshold, dict):
                threshold = Range(min=threshold.get("min"), max=threshold.get("max"))

            properties.append(
                CharacteristicProperty(
                    name=prop_data["name"],
                    property_type=prop_type,
                    formal_definition=prop_data.get("definition", ""),
                    measurement_function=lambda c: 0.5,  # Placeholder
                    threshold=threshold,
                    weight=prop_data.get("weight", 1.0),
                )
            )

        # Parse examples
        positive_examples = [
            Example(
                content=ex.get("content", ""),
                category=name,
                properties=ex.get("properties", {}),
                annotations=ex.get("annotations", {}),
                confidence=ex.get("confidence", 1.0),
                source=ex.get("source", ""),
            )
            for ex in data.get("positive_examples", [])
        ]

        negative_examples = [
            Example(
                content=ex.get("content", ""),
                category=name,
                properties=ex.get("properties", {}),
                annotations=ex.get("annotations", {}),
                confidence=ex.get("confidence", 1.0),
                source=ex.get("source", ""),
            )
            for ex in data.get("negative_examples", [])
        ]

        # Parse rubric
        rubric = None
        if "rubric" in data:
            rubric = EvaluationRubric(
                name=data["rubric"].get("name", f"{name}_rubric"),
                description=data["rubric"].get("description", ""),
                scoring_criteria=data["rubric"].get("criteria", {}),
                scale=tuple(data["rubric"].get("scale", [0, 1])),
            )

        return CategoryDefinition(
            name=name,
            description=data.get("description", ""),
            characteristic_properties=properties,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            parent_category=data.get("parent"),
            evaluation_rubric=rubric,
            metadata=data.get("metadata", {}),
        )

    def save_to_yaml(self, path: Path) -> None:
        """
        Save all categories to a YAML file.

        Args:
            path: Path to save YAML file
        """
        data = {}
        for name, category in self.categories.items():
            data[name] = self._category_to_dict(category)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _category_to_dict(self, category: CategoryDefinition) -> dict:
        """Convert category to dictionary for serialization."""
        return {
            "description": category.description,
            "properties": [
                {
                    "name": prop.name,
                    "type": prop.property_type.value,
                    "definition": prop.formal_definition,
                    "threshold": (
                        {"min": prop.threshold.min, "max": prop.threshold.max}
                        if isinstance(prop.threshold, Range)
                        else prop.threshold
                    ),
                    "weight": prop.weight,
                }
                for prop in category.characteristic_properties
            ],
            "positive_examples": [
                {
                    "content": ex.content,
                    "properties": ex.properties,
                    "annotations": ex.annotations,
                    "confidence": ex.confidence,
                    "source": ex.source,
                }
                for ex in category.positive_examples
            ],
            "negative_examples": [
                {
                    "content": ex.content,
                    "properties": ex.properties,
                    "annotations": ex.annotations,
                    "confidence": ex.confidence,
                    "source": ex.source,
                }
                for ex in category.negative_examples
            ],
            "parent": category.parent_category,
            "rubric": (
                {
                    "name": category.evaluation_rubric.name,
                    "description": category.evaluation_rubric.description,
                    "criteria": category.evaluation_rubric.scoring_criteria,
                    "scale": list(category.evaluation_rubric.scale),
                }
                if category.evaluation_rubric
                else None
            ),
            "metadata": category.metadata,
        }

    @classmethod
    def load(cls, path: str) -> "CategoryRegistry":
        """
        Load category registry from file.

        Args:
            path: Path to YAML or JSON file

        Returns:
            CategoryRegistry instance
        """
        registry = cls()
        path_obj = Path(path)

        if path_obj.suffix == ".yaml" or path_obj.suffix == ".yml":
            registry.load_from_yaml(path_obj)
        elif path_obj.suffix == ".json":
            registry.load_from_json(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        return registry
