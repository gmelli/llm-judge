# Development Guide

## Setting Up Development Environment

### Prerequisites

- Python 3.9 or higher
- Git
- pip and virtualenv

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/gmelli/llm-judge.git
cd llm-judge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
llm-judge/
├── src/llm_judge/          # Source code
│   ├── core/               # Core functionality
│   │   ├── category.py     # Category definitions
│   │   ├── evaluation.py   # Evaluation engine
│   │   └── judge.py        # Main Judge class
│   ├── categories/         # Category management
│   │   └── registry.py     # Category registry
│   ├── providers/          # LLM providers
│   │   ├── base.py         # Base provider interface
│   │   ├── openai.py       # OpenAI implementation
│   │   ├── anthropic.py    # Anthropic implementation
│   │   ├── gemini.py       # Google Gemini implementation
│   │   └── mock.py         # Mock provider for testing
│   └── utils/              # Utilities
│       └── consensus.py    # Consensus mechanisms
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test fixtures
├── examples/              # Example code
├── docs/                  # Documentation
└── pyproject.toml         # Package configuration
```

## Code Style

### Formatting

We use Black for code formatting and Ruff for linting:

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/

# Fix linting issues
ruff check src/ --fix
```

### Type Hints

All functions should have type hints:

```python
from typing import List, Optional, Union

def evaluate(
    content: str,
    category: CategoryDefinition,
    return_feedback: bool = True,
) -> EvaluationResult:
    """
    Evaluate content against a category.

    Args:
        content: The content to evaluate
        category: The category definition
        return_feedback: Whether to include feedback

    Returns:
        EvaluationResult with scores and feedback
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Brief description of function.

    Longer description if needed, explaining the purpose
    and any important details about the implementation.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        "success"
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_judge --cov-report=html

# Run specific test file
pytest tests/unit/test_category.py

# Run with verbose output
pytest -v

# Run only fast tests (skip integration)
pytest -m "not integration"
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_category.py
import pytest
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType


def test_category_creation():
    """Test creating a category definition."""
    category = CategoryDefinition(
        name="test_category",
        description="Test description",
        characteristic_properties=[],
    )

    assert category.name == "test_category"
    assert category.description == "Test description"
    assert len(category.characteristic_properties) == 0


def test_property_measurement():
    """Test property measurement function."""
    prop = CharacteristicProperty(
        name="length",
        property_type=PropertyType.NECESSARY,
        formal_definition="Length > 10",
        measurement_function=lambda c: len(c),
        threshold=10,
    )

    assert prop.measure("short") == 5
    assert prop.meets_threshold(15) == True
    assert prop.meets_threshold(5) == False


@pytest.mark.asyncio
async def test_async_evaluation():
    """Test async evaluation."""
    from llm_judge import Judge

    judge = Judge(provider="mock")
    # ... test async code
```

#### Integration Test Example

```python
# tests/integration/test_providers.py
import pytest
import os
from llm_judge import Judge, CategoryRegistry


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not set"
)
async def test_openai_provider():
    """Test OpenAI provider integration."""
    judge = Judge(provider="openai", model="gpt-3.5-turbo")
    registry = CategoryRegistry()
    category = registry.get("academic_writing")

    result = await judge.evaluate(
        "This is a test academic paper.",
        category
    )

    assert result is not None
    assert 0 <= result.membership_score <= 1
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType


@pytest.fixture
def sample_category():
    """Provide a sample category for testing."""
    return CategoryDefinition(
        name="test_category",
        description="Test category",
        characteristic_properties=[
            CharacteristicProperty(
                name="test_prop",
                property_type=PropertyType.NECESSARY,
                formal_definition="Test property",
                measurement_function=lambda c: len(c),
                threshold=10,
            )
        ],
    )


@pytest.fixture
def mock_judge():
    """Provide a mock judge for testing."""
    from llm_judge import Judge
    return Judge(provider="mock")
```

## Adding New Features

### Adding a New Provider

1. Create provider file in `src/llm_judge/providers/`:

```python
# src/llm_judge/providers/newprovider.py
from typing import List, Optional, Any
from llm_judge.providers.base import JudgeProvider, ProviderResult
from llm_judge.core.category import CategoryDefinition, Example


class NewProvider(JudgeProvider):
    """New provider implementation."""

    def __init__(
        self,
        model: str = "default-model",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key or os.getenv("NEW_PROVIDER_API_KEY")

    async def evaluate(
        self,
        content: str,
        category: CategoryDefinition,
        examples: Optional[List[Example]] = None,
    ) -> ProviderResult:
        """Implement evaluation logic."""
        # Your implementation
        ...

    async def batch_evaluate(
        self,
        contents: List[str],
        category: CategoryDefinition,
    ) -> List[ProviderResult]:
        """Implement batch evaluation."""
        # Your implementation
        ...
```

2. Register in `judge.py`:

```python
# src/llm_judge/core/judge.py
provider_map = {
    # ... existing providers
    "newprovider": "llm_judge.providers.newprovider.NewProvider",
}
```

3. Add tests:

```python
# tests/unit/test_newprovider.py
import pytest
from llm_judge.providers.newprovider import NewProvider


@pytest.mark.asyncio
async def test_newprovider_evaluate():
    provider = NewProvider()
    # Test implementation
    ...
```

### Adding a New Category Type

1. Define the category:

```python
# src/llm_judge/categories/custom.py
from llm_judge import CategoryDefinition, CharacteristicProperty, PropertyType


def create_custom_category() -> CategoryDefinition:
    """Create a custom category definition."""
    return CategoryDefinition(
        name="custom_category",
        description="Custom category description",
        characteristic_properties=[
            CharacteristicProperty(
                name="custom_property",
                property_type=PropertyType.NECESSARY,
                formal_definition="Custom property definition",
                measurement_function=custom_measurement,
                threshold=0.5,
            ),
        ],
    )


def custom_measurement(content: str) -> float:
    """Custom measurement function."""
    # Implement measurement logic
    return 0.0
```

2. Register in registry:

```python
# src/llm_judge/categories/registry.py
def _load_builtin_categories(self):
    # ... existing categories
    self._add_custom_categories()

def _add_custom_categories(self):
    from llm_judge.categories.custom import create_custom_category
    self.register(create_custom_category())
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Using VS Code

`.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "${file}"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

### Using PyCharm

1. Right-click on `tests/` directory
2. Select "Run 'pytest in tests'"
3. For debugging, select "Debug 'pytest in tests'"

## CI/CD

### GitHub Actions

The CI pipeline runs on every push and PR:

```yaml
# .github/workflows/ci.yml
- Linting with Ruff
- Formatting check with Black
- Type checking with MyPy
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Coverage reporting
```

### Pre-commit Hooks

Pre-commit hooks run automatically before commits:

```bash
# Manual run
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### Creating a Release

1. Create and push a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

2. GitHub Actions will automatically:
   - Run tests
   - Build the package
   - Publish to PyPI (if PYPI_API_TOKEN is set)

3. Create GitHub Release:
   - Go to GitHub releases page
   - Click "Create release"
   - Select the tag
   - Add release notes from CHANGELOG

## Benchmarking

### Performance Testing

```python
# benchmarks/test_performance.py
import time
import asyncio
from llm_judge import Judge, CategoryRegistry


async def benchmark_evaluation():
    judge = Judge(provider="mock")
    registry = CategoryRegistry()
    category = registry.get("academic_writing")

    content = "Test content" * 100  # Longer content

    start = time.time()
    for _ in range(100):
        await judge.evaluate(content, category)
    elapsed = time.time() - start

    print(f"100 evaluations took {elapsed:.2f} seconds")
    print(f"Average: {elapsed/100:.4f} seconds per evaluation")


if __name__ == "__main__":
    asyncio.run(benchmark_evaluation())
```

### Memory Profiling

```python
from memory_profiler import profile


@profile
def memory_test():
    from llm_judge import Judge, CategoryRegistry

    judge = Judge(provider="mock")
    registry = CategoryRegistry()

    # Create many categories
    for i in range(1000):
        category = registry.get("academic_writing")
        # ... use category


if __name__ == "__main__":
    memory_test()
```

## Troubleshooting Development Issues

### Common Problems

#### ImportError in Tests
```bash
# Solution: Install in editable mode
pip install -e .
```

#### Type Checking Failures
```bash
# Install type stubs
pip install types-PyYAML types-requests
```

#### Pre-commit Hook Failures
```bash
# Skip hooks temporarily
git commit --no-verify

# Fix and re-run
pre-commit run --all-files
```

## Contributing Checklist

Before submitting a PR:

- [ ] Code follows style guidelines (Black, Ruff)
- [ ] Added/updated tests for new functionality
- [ ] All tests pass (`pytest`)
- [ ] Added/updated documentation
- [ ] Type hints added for new functions
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Pre-commit hooks pass
- [ ] Branch is up-to-date with main

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/gmelli/llm-judge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gmelli/llm-judge/discussions)
- **Documentation**: [API Reference](./API.md)
- **Examples**: [examples/](../examples/)