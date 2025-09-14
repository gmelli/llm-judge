"""Basic tests for LLM-as-Judge library."""

import pytest
from llm_judge import (
    CategoryDefinition,
    CharacteristicProperty,
    PropertyType,
    Judge,
    CategoryRegistry,
)
from llm_judge.core import Range


def test_category_creation():
    """Test creating a basic category definition."""
    category = CategoryDefinition(
        name="test_category",
        description="A test category",
        characteristic_properties=[
            CharacteristicProperty(
                name="length",
                property_type=PropertyType.NECESSARY,
                formal_definition="Content length > 10",
                measurement_function=lambda c: len(c),
                threshold=10,
            )
        ],
    )

    assert category.name == "test_category"
    assert len(category.characteristic_properties) == 1
    assert category.get_necessary_properties()[0].name == "length"


def test_property_measurement():
    """Test property measurement functions."""
    prop = CharacteristicProperty(
        name="word_count",
        property_type=PropertyType.NECESSARY,
        formal_definition="Word count >= 5",
        measurement_function=lambda c: len(c.split()),
        threshold=5,
    )

    assert prop.measure("This is a test sentence") == 5
    assert prop.meets_threshold(5) == True
    assert prop.meets_threshold(4) == False


def test_range_threshold():
    """Test range-based thresholds."""
    range_threshold = Range(min=10, max=100)

    assert range_threshold.contains(50) == True
    assert range_threshold.contains(5) == False
    assert range_threshold.contains(150) == False


def test_category_registry():
    """Test category registry functionality."""
    registry = CategoryRegistry()

    # Check built-in categories
    assert "academic_writing" in registry.list_categories()
    assert "technical_documentation" in registry.list_categories()
    assert "safe_content" in registry.list_categories()

    # Get a specific category
    academic = registry.get("academic_writing")
    assert academic is not None
    assert academic.name == "academic_writing"


@pytest.mark.asyncio
async def test_mock_judge():
    """Test judge with mock provider."""
    registry = CategoryRegistry()
    category = registry.get("academic_writing")

    judge = Judge(provider="mock", always_match=True)

    content = "This is a test academic paper with citations [1] and formal language."
    result = await judge.evaluate(content, category)

    assert result.category == "academic_writing"
    assert result.matches_category == True
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_comparison():
    """Test content comparison functionality."""
    registry = CategoryRegistry()
    category = registry.get("technical_documentation")

    judge = Judge(provider="mock")

    original = "Short tech doc"
    enhanced = "Comprehensive technical documentation with code examples ```python\nprint('hello')\n```"

    comparison = await judge.compare(original, enhanced, category)

    assert comparison.original_score != comparison.modified_score
    assert comparison.recommendation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])