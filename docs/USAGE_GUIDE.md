# Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Working with Categories](#working-with-categories)
4. [Evaluation Strategies](#evaluation-strategies)
5. [Provider Configuration](#provider-configuration)
6. [Best Practices](#best-practices)
7. [Common Use Cases](#common-use-cases)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation
```bash
pip install llm-judge
```

### With Provider Support
```bash
pip install "llm-judge[providers]"
```

### Development Installation
```bash
git clone https://github.com/gmelli/llm-judge.git
cd llm-judge
pip install -e ".[dev]"
```

## Quick Start

### Basic Evaluation

```python
import asyncio
from llm_judge import Judge, CategoryRegistry

async def main():
    # Load built-in categories
    registry = CategoryRegistry()
    category = registry.get("academic_writing")

    # Initialize judge
    judge = Judge(provider="mock")  # Use "openai", "anthropic", or "gemini" with API keys

    # Evaluate content
    content = "Your text to evaluate..."
    result = await judge.evaluate(content, category)

    print(f"Matches: {result.matches_category}")
    print(f"Score: {result.membership_score:.2f}")

asyncio.run(main())
```

### For Jupyter Notebooks

```python
import nest_asyncio
nest_asyncio.apply()

# Then use await directly in cells
result = await judge.evaluate(content, category)
```

## Working with Categories

### Using Built-in Categories

The library includes several pre-defined categories:

```python
registry = CategoryRegistry()

# Available categories
print(registry.list_categories())
# Output: ['academic_writing', 'technical_documentation', 'safe_content']

# Get a specific category
academic = registry.get("academic_writing")
```

### Creating Custom Categories

#### Simple Category

```python
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType

blog_post = CategoryDefinition(
    name="blog_post",
    description="Engaging blog post with informal tone",
    characteristic_properties=[
        CharacteristicProperty(
            name="conversational_tone",
            property_type=PropertyType.NECESSARY,
            formal_definition="Uses personal pronouns and casual language",
            measurement_function=lambda c: c.count("you") + c.count("we") > 5,
            threshold=True,
            weight=2.0,
        ),
    ],
)
```

#### Complex Category with Range Thresholds

```python
from llm_judge import Range

research_paper = CategoryDefinition(
    name="research_paper",
    description="Academic research paper",
    characteristic_properties=[
        CharacteristicProperty(
            name="length",
            property_type=PropertyType.NECESSARY,
            formal_definition="Between 2000 and 10000 words",
            measurement_function=lambda c: len(c.split()),
            threshold=Range(min=2000, max=10000),
            weight=1.0,
        ),
        CharacteristicProperty(
            name="abstract_present",
            property_type=PropertyType.NECESSARY,
            formal_definition="Contains an abstract section",
            measurement_function=lambda c: "abstract" in c.lower(),
            threshold=True,
            weight=1.5,
        ),
    ],
)
```

### Loading Categories from Files

#### YAML Format

```yaml
# categories.yaml
legal_document:
  description: "Legal document with formal language"
  properties:
    - name: legal_terms
      type: necessary
      definition: "Contains legal terminology"
      threshold: 5
      weight: 2.0
    - name: numbered_sections
      type: typical
      definition: "Has numbered sections"
      threshold: true
      weight: 1.0
  positive_examples:
    - content: "WHEREAS the party of the first part..."
      properties:
        legal_terms: 10
```

```python
registry = CategoryRegistry.load("categories.yaml")
```

### Adding Examples

Examples improve evaluation accuracy:

```python
from llm_judge import Example

# Add positive example
category.add_positive_example(Example(
    content="This rigorous study, utilizing a randomized controlled trial...",
    category="academic_writing",
    properties={"citation_density": 4.5, "formality_score": 0.9},
    confidence=0.95,
))

# Add negative example
category.add_negative_example(Example(
    content="Hey guys, check out this cool science thing!",
    category="academic_writing",
    properties={"citation_density": 0, "formality_score": 0.2},
    confidence=0.95,
))
```

## Evaluation Strategies

### Single Provider Evaluation

```python
judge = Judge(provider="openai", model="gpt-4-turbo-preview")
result = await judge.evaluate(content, category)
```

### Multi-Provider Consensus

```python
from llm_judge import ConsensusMode

# Majority vote
judge = Judge(
    providers=["openai", "anthropic", "gemini"],
    consensus_mode=ConsensusMode.MAJORITY
)

# Unanimous agreement required
judge = Judge(
    providers=["openai", "anthropic"],
    consensus_mode=ConsensusMode.UNANIMOUS
)

# Weighted consensus
judge = Judge(
    providers=["openai", "anthropic", "gemini"],
    consensus_mode=ConsensusMode.WEIGHTED,
    weights=[0.5, 0.3, 0.2]  # OpenAI gets 50% weight
)
```

### Content Comparison

```python
# Compare before/after versions
comparison = await judge.compare(
    original="Short article about Python.",
    enhanced="Python is a versatile programming language...",
    target_category=category
)

print(f"Improvement: {comparison.progress_percentage:.1f}%")
print(f"Better properties: {comparison.improved_properties}")
```

### Batch Processing

```python
contents = ["text1", "text2", "text3", ...]

# Process with concurrency control
results = await judge.batch_evaluate(
    contents,
    category,
    max_concurrent=5  # Process 5 at a time
)

for content, result in zip(contents, results):
    print(f"{content[:30]}... -> {result.matches_category}")
```

## Provider Configuration

### API Key Setup

```bash
# Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or in Python:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Model Selection

```python
# OpenAI
judge = Judge(provider="openai", model="gpt-4")  # or "gpt-3.5-turbo"

# Anthropic
judge = Judge(provider="anthropic", model="claude-3-opus-20240229")

# Google
judge = Judge(provider="gemini", model="gemini-1.5-pro")
```

### Temperature Control

```python
# Lower = more consistent (0.0-0.3 recommended for evaluation)
judge = Judge(provider="openai", temperature=0.1)

# Higher = more creative (not recommended for evaluation)
judge = Judge(provider="openai", temperature=0.8)
```

## Best Practices

### 1. Property Design

```python
# Good: Measurable and specific
CharacteristicProperty(
    name="citation_count",
    measurement_function=lambda c: len(re.findall(r'\[\d+\]', c)),
    threshold=5,
)

# Bad: Vague and subjective
CharacteristicProperty(
    name="good_writing",
    measurement_function=lambda c: True,  # Too simplistic
    threshold=True,
)
```

### 2. Category Hierarchy

```python
# Parent category
academic = CategoryDefinition(name="academic", ...)

# Specialized subcategories
scientific = CategoryDefinition(
    name="scientific_paper",
    parent_category="academic",
    characteristic_properties=[
        # Inherits academic properties
        # Plus specific scientific properties
    ],
)
```

### 3. Error Handling

```python
try:
    result = await judge.evaluate(content, category)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
except Exception as e:
    logger.error(f"Evaluation failed: {e}")
    # Use fallback logic
```

### 4. Caching Strategy

```python
# Enable caching for repeated evaluations
judge = Judge(provider="openai", cache_enabled=True)

# Check cache stats
stats = judge.get_cache_stats()
print(f"Cache hits: {stats['hits']}")

# Clear when needed
judge.clear_cache()
```

## Common Use Cases

### Content Moderation

```python
safe_category = registry.get("safe_content")

async def moderate_content(text):
    result = await judge.evaluate(text, safe_category)
    if not result.matches_category:
        return "Content flagged for review"
    return "Content approved"
```

### Quality Assessment

```python
async def assess_quality(article):
    categories = ["stub", "good", "featured"]
    scores = {}

    for cat_name in categories:
        category = registry.get(cat_name)
        result = await judge.evaluate(article, category)
        scores[cat_name] = result.membership_score

    best_category = max(scores, key=scores.get)
    return best_category, scores
```

### A/B Testing

```python
async def compare_versions(version_a, version_b, category):
    results_a = await judge.evaluate(version_a, category)
    results_b = await judge.evaluate(version_b, category)

    if results_b.membership_score > results_a.membership_score:
        return "version_b", results_b.membership_score - results_a.membership_score
    return "version_a", 0
```

### Continuous Monitoring

```python
async def monitor_content_quality(contents, category, threshold=0.7):
    results = await judge.batch_evaluate(contents, category)

    below_threshold = [
        (content, result)
        for content, result in zip(contents, results)
        if result.membership_score < threshold
    ]

    if below_threshold:
        alert_quality_team(below_threshold)

    return len(below_threshold) / len(contents)  # Failure rate
```

## Performance Optimization

### 1. Batch Processing

```python
# Instead of:
for content in contents:
    result = await judge.evaluate(content, category)

# Do:
results = await judge.batch_evaluate(contents, category)
```

### 2. Caching

```python
# Enable caching for repeated content
judge = Judge(cache_enabled=True)

# Pre-warm cache with common evaluations
common_contents = load_common_contents()
await judge.batch_evaluate(common_contents, category)
```

### 3. Provider Selection

```python
# For speed: Use Gemini Flash or GPT-3.5
fast_judge = Judge(provider="gemini", model="gemini-1.5-flash")

# For accuracy: Use GPT-4 or Claude Opus
accurate_judge = Judge(provider="openai", model="gpt-4")

# Tiered approach
async def tiered_evaluation(content, category):
    # Quick check with fast model
    fast_result = await fast_judge.evaluate(content, category)

    # If borderline, verify with accurate model
    if 0.4 < fast_result.membership_score < 0.6:
        return await accurate_judge.evaluate(content, category)

    return fast_result
```

### 4. Async Concurrency

```python
import asyncio

async def evaluate_many(contents, categories):
    tasks = []
    for content in contents:
        for category in categories:
            tasks.append(judge.evaluate(content, category))

    # Run all evaluations concurrently
    results = await asyncio.gather(*tasks)
    return results
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Error: No module named 'openai'
# Solution: Install provider dependencies
pip install "llm-judge[providers]"
```

#### API Key Issues
```python
# Error: API key not found
# Solution: Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Or pass directly (not recommended)
judge = Judge(provider="openai", api_key="your-key")
```

#### Async Issues in Jupyter
```python
# Error: RuntimeError: This event loop is already running
# Solution: Use nest_asyncio
import nest_asyncio
nest_asyncio.apply()
```

#### Rate Limiting
```python
# Implement exponential backoff
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
async def evaluate_with_retry(content, category):
    return await judge.evaluate(content, category)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check what's happening
result = await judge.evaluate(content, category)
print(f"Model used: {result.metadata.get('model_used')}")
print(f"Tokens: {result.metadata.get('tokens_used')}")
print(f"Cost: ${result.evaluation_cost:.4f}")
```

### Performance Profiling

```python
import time

start = time.time()
result = await judge.evaluate(content, category)
print(f"Evaluation took: {time.time() - start:.2f} seconds")

# Check cache performance
stats = judge.get_cache_stats()
hit_rate = stats['hits'] / (stats['hits'] + stats['misses'])
print(f"Cache hit rate: {hit_rate:.2%}")
```