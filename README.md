# LLM-as-Judge Library

[![CI](https://github.com/gmelli/llm-judge/actions/workflows/ci.yml/badge.svg)](https://github.com/gmelli/llm-judge/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/llm-judge.svg)](https://badge.fury.io/py/llm-judge)
[![Python Support](https://img.shields.io/pypi/pyversions/llm-judge.svg)](https://pypi.org/project/llm-judge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/gmelli/llm-judge/branch/main/graph/badge.svg)](https://codecov.io/gh/gmelli/llm-judge)

A robust Python library for evaluating content using Large Language Models as judges, with support for formal category definitions, characteristic properties, and multi-provider consensus.

## Features

- **Formal Category System**: Define categories with characteristic properties (necessary, sufficient, typical)
- **Multi-Provider Support**: OpenAI GPT, Anthropic Claude, Google Gemini
- **Consensus Mechanisms**: Multiple consensus modes (unanimous, majority, weighted, strict)
- **Property-Based Evaluation**: Measure specific properties with custom functions
- **Example-Based Learning**: Use positive and negative examples for better categorization
- **Comparison Evaluation**: Compare before/after content modifications
- **Extensible Architecture**: Easy to add custom categories, providers, and evaluators
- **Analytics & Monitoring**: Track evaluation metrics and costs
- **Caching**: Intelligent caching for improved performance

## Installation

```bash
pip install llm-judge
```

For development installation with all providers:

```bash
pip install "llm-judge[all]"
```

## Quick Start

### Basic Usage

```python
from llm_judge import Judge, CategoryRegistry

# Load built-in categories
registry = CategoryRegistry()

# Get a specific category
academic_category = registry.get("academic_writing")

# Initialize judge with a provider
judge = Judge(provider="openai", temperature=0.1)

# Evaluate content
content = "This study examines the impact of climate change on biodiversity..."
result = await judge.evaluate(content, academic_category)

print(f"Matches category: {result.matches_category}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Feedback: {result.feedback}")
```

### Defining Custom Categories

```python
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType

# Define a custom category for wiki articles
wiki_stub = CategoryDefinition(
    name="wiki_stub",
    description="Short Wikipedia article that needs expansion",
    characteristic_properties=[
        CharacteristicProperty(
            name="word_count",
            property_type=PropertyType.NECESSARY,
            formal_definition="Word count between 20 and 500",
            measurement_function=lambda c: len(c.split()),
            threshold=Range(min=20, max=500),
            weight=2.0,
        ),
        CharacteristicProperty(
            name="has_definition",
            property_type=PropertyType.NECESSARY,
            formal_definition="Contains opening definition sentence",
            measurement_function=lambda c: c.strip().endswith(".") and len(c.split(".")[0].split()) > 5,
            threshold=True,
            weight=1.5,
        ),
    ],
)

# Register and use the custom category
registry.register(wiki_stub)
result = await judge.evaluate(article_text, wiki_stub)
```

### Multi-Provider Consensus

```python
from llm_judge import Judge, ConsensusMode

# Use multiple providers with consensus
judge = Judge(
    providers=["gpt-4", "claude-3", "gemini-pro"],
    consensus_mode=ConsensusMode.MAJORITY
)

# Evaluate with consensus
result = await judge.evaluate(content, category)
print(f"Agreement: {result.metadata['agreement']}")
print(f"Positive votes: {result.metadata['positive_votes']}")
```

### Content Comparison

```python
# Compare original and enhanced content
comparison = await judge.compare(
    original=stub_content,
    enhanced=enhanced_content,
    target_category=featured_article_category
)

print(f"Progress: {comparison.progress_percentage:.1f}%")
print(f"Improved properties: {comparison.improved_properties}")
print(f"Still needed: {comparison.missing_properties}")
```

### Batch Evaluation

```python
# Evaluate multiple contents efficiently
contents = [content1, content2, content3, ...]
results = await judge.batch_evaluate(
    contents,
    category=safe_content_category,
    max_concurrent=5
)

for content, result in zip(contents, results):
    print(f"Content {content[:50]}... -> {result.matches_category}")
```

## Category Types

### Built-in Categories

The library comes with several built-in category types:

- **Academic Writing**: Formal academic content with citations
- **Technical Documentation**: Technical docs with code examples
- **Safe Content**: Content safe for all audiences
- **Creative Writing**: Fiction, poetry, creative content
- **Wiki Articles**: Various Wikipedia article types

### Loading Categories from Files

```python
# Load from YAML
registry = CategoryRegistry.load("categories.yaml")

# Load from JSON
registry = CategoryRegistry.load("categories.json")
```

Example YAML format:

```yaml
academic_writing:
  description: "Formal academic writing"
  properties:
    - name: citation_density
      type: necessary
      definition: "Citations per 100 words >= 2"
      threshold: 2.0
      weight: 1.5
    - name: formality_score
      type: necessary
      definition: "Formal language score >= 0.7"
      threshold: 0.7
      weight: 1.0
  positive_examples:
    - content: "According to Smith et al. (2023)..."
      properties:
        citation_density: 3.5
        formality_score: 0.85
  negative_examples:
    - content: "I think this is cool..."
      properties:
        citation_density: 0
        formality_score: 0.3
```

## Advanced Features

### Custom Providers

```python
from llm_judge import JudgeProvider, ProviderResult

class CustomProvider(JudgeProvider):
    async def evaluate(self, content, category, examples=None):
        # Your custom evaluation logic
        return ProviderResult(
            category_match=True,
            confidence=0.95,
            scores={"property1": 0.8},
            reasoning="Custom evaluation logic",
            model="custom-model",
        )

# Use custom provider
judge = Judge(provider=CustomProvider())
```

### Property Measurement Functions

```python
# Define custom measurement functions
def count_citations(content: str) -> int:
    import re
    citations = re.findall(r'\[\d+\]|\(\w+,?\s+\d{4}\)', content)
    return len(citations)

def measure_readability(content: str) -> float:
    # Implement readability scoring
    return flesch_reading_ease(content)

# Use in property definitions
CharacteristicProperty(
    name="citations",
    property_type=PropertyType.NECESSARY,
    measurement_function=count_citations,
    threshold=5,
)
```

### Caching and Performance

```python
# Enable caching (default)
judge = Judge(cache_enabled=True)

# Clear cache when needed
judge.clear_cache()

# Get cache statistics
stats = judge.get_cache_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

## API Reference

### Core Classes

- `CategoryDefinition`: Define evaluation categories
- `CharacteristicProperty`: Define measurable properties
- `PropertyType`: Enum for property types (NECESSARY, SUFFICIENT, TYPICAL)
- `Example`: Positive/negative examples for categories
- `EvaluationRubric`: Scoring rubrics for categories

### Evaluation Classes

- `Judge`: Main interface for evaluations
- `EvaluationEngine`: Core evaluation logic
- `EvaluationResult`: Results from evaluation
- `ComparisonResult`: Results from comparison

### Provider Classes

- `JudgeProvider`: Base class for LLM providers
- `ConsensusJudge`: Multi-provider consensus
- `ConsensusMode`: Consensus strategies

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{llm_judge,
  title = {LLM-as-Judge: A Python Library for Content Evaluation},
  author = {Melli, Gabor},
  year = {2024},
  url = {https://github.com/gmelli/llm-judge}
}
```

## Support

- Documentation: [https://gmelli.github.io/llm-judge](https://gmelli.github.io/llm-judge)
- Issues: [GitHub Issues](https://github.com/gmelli/llm-judge/issues)
- Discussions: [GitHub Discussions](https://github.com/gmelli/llm-judge/discussions)