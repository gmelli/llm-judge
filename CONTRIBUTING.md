# Contributing to LLM-as-Judge

Thank you for your interest in contributing to LLM-as-Judge! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

- Check if the issue already exists in the [issue tracker](https://github.com/gmelli/llm-judge/issues)
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Share relevant code snippets or error messages

### Suggesting Features

- Open a discussion in [GitHub Discussions](https://github.com/gmelli/llm-judge/discussions)
- Describe the feature and its use case
- Explain how it would benefit users

### Submitting Pull Requests

1. **Fork the repository** and create a new branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up development environment**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests and checks**
   ```bash
   # Run tests
   pytest tests/

   # Check formatting
   black src/

   # Run linter
   ruff check src/

   # Type checking
   mypy src/llm_judge
   ```

5. **Commit your changes**
   ```bash
   git commit -m "feat: add new feature"
   ```

   Follow conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for refactoring

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Use Black for formatting (line length: 88)
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for all public functions and classes

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external API calls in tests

### Documentation

- Update README for user-facing changes
- Add docstrings following Google style
- Include examples for new features
- Update type hints

## Adding New Providers

To add a new LLM provider:

1. Create a new file in `src/llm_judge/providers/`
2. Inherit from `JudgeProvider` base class
3. Implement required methods:
   - `evaluate()`
   - `batch_evaluate()`
4. Add tests in `tests/providers/`
5. Update documentation

Example structure:
```python
from llm_judge.providers.base import JudgeProvider, ProviderResult

class YourProvider(JudgeProvider):
    async def evaluate(self, content, category, examples=None):
        # Your implementation
        return ProviderResult(...)
```

## Adding New Categories

To add new category definitions:

1. Use the `CategoryDefinition` class
2. Define characteristic properties
3. Add positive and negative examples
4. Register with `CategoryRegistry`

## Release Process

Releases are automated via GitHub Actions when a new tag is pushed:

```bash
git tag v0.2.0
git push origin v0.2.0
```

This triggers:
1. Running all tests
2. Building the package
3. Publishing to PyPI

## Questions?

Feel free to:
- Open an issue for bugs
- Start a discussion for questions
- Contact the maintainers

Thank you for contributing!