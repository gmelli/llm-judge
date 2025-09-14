# Changelog

All notable changes to the LLM-as-Judge library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API documentation
- Usage guide with best practices
- Development guide for contributors

### Validated
- 82% accuracy on binary classification tasks (+24% over baseline)
- $0.0001 per evaluation cost with Gemini 1.5 Flash
- 100% precision on hallucination detection
- Production validation across 5 phases with <$0.02 total cost
- Superior performance over ROUGE/BLEU for semantic tasks

## [0.1.0] - 2024-12-14

### Added
- Initial release of LLM-as-Judge library
- Core category system with formal properties
- Three LLM providers: OpenAI, Anthropic, Google Gemini
- Mock provider for testing
- Consensus mechanisms for multi-provider evaluation
- Category registry with built-in categories
- Content comparison functionality
- Batch evaluation support
- Result caching system
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Jupyter notebook demo with 10 examples
- Contributing guidelines

### Categories
- Academic Writing
- Technical Documentation
- Safe Content
- Custom category support

### Providers
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini 1.5 Pro, Flash)
- Mock provider for testing

### Features
- Property-based evaluation with custom measurement functions
- Example-based learning with positive/negative examples
- Before/after content comparison
- Multi-provider consensus (unanimous, majority, weighted, strict)
- Async/await support for concurrent operations
- Type hints throughout
- Comprehensive error handling
- Result caching for performance

### Documentation
- README with installation and usage
- API reference documentation
- Contributing guidelines
- MIT License
- Example notebooks

[0.1.0]: https://github.com/gmelli/llm-judge/releases/tag/v0.1.0