# API Reference

## Core Classes

### CategoryDefinition

Defines a category with formal characteristic properties and examples.

```python
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType

category = CategoryDefinition(
    name="academic_writing",
    description="Formal academic writing with citations",
    characteristic_properties=[...],
    positive_examples=[...],
    negative_examples=[...],
    parent_category=None,
    evaluation_rubric=None,
)
```

**Parameters:**
- `name` (str): Unique identifier for the category
- `description` (str): Human-readable description
- `characteristic_properties` (List[CharacteristicProperty]): Properties that define the category
- `positive_examples` (List[Example]): Examples that belong to this category
- `negative_examples` (List[Example]): Examples that don't belong
- `parent_category` (Optional[str]): Parent in category hierarchy
- `evaluation_rubric` (Optional[EvaluationRubric]): Scoring rubric

**Methods:**
- `get_necessary_properties()`: Returns properties required for membership
- `get_sufficient_properties()`: Returns properties that alone determine membership
- `get_typical_properties()`: Returns properties usually present
- `add_property(property)`: Add a new characteristic property
- `add_positive_example(example)`: Add a positive example
- `add_negative_example(example)`: Add a negative example

---

### CharacteristicProperty

Formal property that must hold for category membership.

```python
from llm_judge import CharacteristicProperty, PropertyType, Range

property = CharacteristicProperty(
    name="citation_density",
    property_type=PropertyType.NECESSARY,
    formal_definition="Citations per 100 words >= 2",
    measurement_function=lambda c: count_citations(c),
    threshold=2.0,
    weight=1.5,
)
```

**Parameters:**
- `name` (str): Property identifier
- `property_type` (PropertyType): NECESSARY, SUFFICIENT, or TYPICAL
- `formal_definition` (str): Mathematical/logical definition
- `measurement_function` (Callable): Function to measure the property
- `threshold` (Union[float, Range, bool]): Acceptable values
- `weight` (float): Importance in category determination (default: 1.0)

**Methods:**
- `measure(content)`: Measure property value for content
- `meets_threshold(value)`: Check if value meets requirement

---

### PropertyType

Enum for types of characteristic properties.

```python
from llm_judge import PropertyType

PropertyType.NECESSARY   # Must be present
PropertyType.SUFFICIENT  # Alone determines membership
PropertyType.TYPICAL     # Usually present but not required
```

---

### Range

Represents a range of acceptable values.

```python
from llm_judge import Range

range = Range(min=10, max=100)
range.contains(50)  # True
range.contains(200)  # False
```

**Parameters:**
- `min` (Optional[float]): Minimum value (inclusive)
- `max` (Optional[float]): Maximum value (inclusive)

---

### Example

Concrete example with annotations.

```python
from llm_judge import Example

example = Example(
    content="Sample academic text with citations [1]...",
    category="academic_writing",
    properties={"citation_density": 3.5},
    annotations={"note": "Good citation structure"},
    confidence=0.95,
    source="journal_article",
)
```

**Parameters:**
- `content` (str): The example text
- `category` (str): Category it belongs to
- `properties` (Dict[str, Any]): Measured property values
- `annotations` (Dict[str, str]): Human explanations
- `confidence` (float): How exemplary (0-1)
- `source` (str): Where example came from

---

## Judge Class

Main interface for LLM-as-Judge evaluations.

```python
from llm_judge import Judge

judge = Judge(
    provider="openai",  # or "anthropic", "gemini", "mock"
    temperature=0.1,
    consensus_mode=None,
    cache_enabled=True,
)
```

**Parameters:**
- `provider` (Union[str, JudgeProvider, List[JudgeProvider]]): LLM provider(s)
- `temperature` (float): Generation temperature (0-1)
- `consensus_mode` (Optional[ConsensusMode]): For multi-provider consensus
- `cache_enabled` (bool): Enable result caching

### Methods

#### evaluate()

Evaluate content against a category.

```python
result = await judge.evaluate(
    content="Your text here",
    category=category_definition,
    return_feedback=True,
    use_cache=True,
)
```

**Returns:** `EvaluationResult`

#### compare()

Compare two versions of content.

```python
comparison = await judge.compare(
    original="Original text",
    enhanced="Improved text",
    target_category=category,
)
```

**Returns:** `ComparisonResult`

#### batch_evaluate()

Evaluate multiple contents efficiently.

```python
results = await judge.batch_evaluate(
    contents=["text1", "text2", "text3"],
    category=category,
    max_concurrent=5,
)
```

**Returns:** `List[EvaluationResult]`

---

## Result Classes

### EvaluationResult

Result from content evaluation.

```python
result.matches_category      # bool: Does content match?
result.membership_score      # float: 0-1 score
result.confidence           # float: Confidence level
result.property_scores      # Dict: Individual property scores
result.similar_examples     # List: Similar examples found
result.feedback            # str: Human-readable feedback
result.failed_properties   # List: Properties that failed
result.passed_properties   # List: Properties that passed
```

**Methods:**
- `generate_improvement_suggestions()`: Get improvement tips

### ComparisonResult

Result from content comparison.

```python
comparison.overall_improvement     # bool: Did it improve?
comparison.progress_percentage     # float: % improvement
comparison.improved_properties     # List: What got better
comparison.regressed_properties    # List: What got worse
comparison.missing_properties      # List: Still failing
comparison.recommendation         # str: Recommendation
```

---

## Providers

### Built-in Providers

#### OpenAIProvider

```python
judge = Judge(provider="openai", model="gpt-4-turbo-preview")
# Requires: OPENAI_API_KEY environment variable
```

**Models:**
- `gpt-4-turbo-preview`
- `gpt-4`
- `gpt-3.5-turbo`

#### AnthropicProvider

```python
judge = Judge(provider="anthropic", model="claude-3-opus-20240229")
# Requires: ANTHROPIC_API_KEY environment variable
```

**Models:**
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

#### GeminiProvider

```python
judge = Judge(provider="gemini", model="gemini-1.5-pro")
# Requires: GOOGLE_API_KEY environment variable
```

**Models:**
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-1.0-pro`

#### MockProvider

```python
judge = Judge(provider="mock")  # For testing, no API key needed
```

### Custom Providers

Create custom providers by inheriting from `JudgeProvider`:

```python
from llm_judge.providers.base import JudgeProvider, ProviderResult

class CustomProvider(JudgeProvider):
    async def evaluate(self, content, category, examples=None):
        # Your evaluation logic
        return ProviderResult(
            category_match=True,
            confidence=0.95,
            scores={},
            reasoning="...",
            model="custom",
        )
```

---

## Consensus

### ConsensusMode

Strategies for multi-provider agreement.

```python
from llm_judge import ConsensusMode

ConsensusMode.UNANIMOUS  # All must agree
ConsensusMode.MAJORITY   # Simple majority
ConsensusMode.WEIGHTED   # Weighted average
ConsensusMode.STRICT     # 75%+ agreement
```

### ConsensusJudge

```python
from llm_judge import Judge, ConsensusMode

judge = Judge(
    providers=["openai", "anthropic", "gemini"],
    consensus_mode=ConsensusMode.MAJORITY,
)
```

---

## Registry

### CategoryRegistry

Manages category definitions.

```python
from llm_judge import CategoryRegistry

# Initialize with built-in categories
registry = CategoryRegistry()

# List categories
categories = registry.list_categories()

# Get specific category
category = registry.get("academic_writing")

# Register custom category
registry.register(custom_category)

# Load from file
registry = CategoryRegistry.load("categories.yaml")

# Save to file
registry.save_to_yaml("categories.yaml")
```

**Methods:**
- `get(name)`: Get category by name
- `register(category)`: Add new category
- `list_categories()`: List all category names
- `load_from_yaml(path)`: Load from YAML
- `load_from_json(path)`: Load from JSON
- `save_to_yaml(path)`: Save to YAML

---

## Utilities

### EvaluationEngine

Low-level evaluation engine used internally.

```python
from llm_judge.core.evaluation import EvaluationEngine

engine = EvaluationEngine()
result = engine.evaluate(content, category)
```

### ComparisonEvaluator

Handles content comparison logic.

```python
from llm_judge.core.evaluation import ComparisonEvaluator

evaluator = ComparisonEvaluator()
result = evaluator.compare_before_after(original, modified, category)
```

---

## Error Handling

The library uses standard Python exceptions:

```python
try:
    result = await judge.evaluate(content, category)
except ValueError as e:
    print(f"Invalid input: {e}")
except ImportError as e:
    print(f"Missing provider package: {e}")
except Exception as e:
    print(f"Evaluation failed: {e}")
```

---

## Type Hints

All functions and methods include type hints for better IDE support:

```python
from typing import List, Optional, Union
from llm_judge import CategoryDefinition, EvaluationResult

async def evaluate(
    content: str,
    category: CategoryDefinition,
    return_feedback: bool = True,
) -> EvaluationResult:
    ...
```