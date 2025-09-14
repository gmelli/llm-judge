"""
Basic usage example for LLM-as-Judge library.
"""

import asyncio
from llm_judge import (
    CategoryDefinition,
    CharacteristicProperty,
    PropertyType,
    Judge,
    CategoryRegistry,
)
from llm_judge.core import Range


async def main():
    """Demonstrate basic usage of the LLM-as-Judge library."""

    # 1. Use built-in categories
    print("=== Using Built-in Categories ===")
    registry = CategoryRegistry()

    # Get academic writing category
    academic_category = registry.get("academic_writing")
    print(f"Category: {academic_category.name}")
    print(f"Description: {academic_category.description}")
    print(f"Properties: {[p.name for p in academic_category.characteristic_properties]}")

    # 2. Create a custom category
    print("\n=== Creating Custom Category ===")
    wiki_stub = CategoryDefinition(
        name="wiki_stub",
        description="Short Wikipedia article that needs expansion",
        characteristic_properties=[
            CharacteristicProperty(
                name="word_count",
                property_type=PropertyType.NECESSARY,
                formal_definition="Word count between 50 and 500",
                measurement_function=lambda c: len(c.split()),
                threshold=Range(min=50, max=500),
                weight=2.0,
            ),
            CharacteristicProperty(
                name="has_definition",
                property_type=PropertyType.NECESSARY,
                formal_definition="Contains opening definition",
                measurement_function=lambda c: "is" in c.lower() or "was" in c.lower(),
                threshold=True,
                weight=1.5,
            ),
        ],
    )

    # Register the custom category
    registry.register(wiki_stub)
    print(f"Registered category: {wiki_stub.name}")

    # 3. Initialize judge with mock provider
    print("\n=== Initializing Judge ===")
    judge = Judge(provider="mock", temperature=0.1)
    print("Judge initialized with mock provider")

    # 4. Evaluate content
    print("\n=== Evaluating Content ===")

    # Academic content
    academic_content = """
    This study examines the impact of climate change on biodiversity.
    According to Smith et al. (2023), global warming has led to significant
    habitat loss. The research methodology involved systematic review of
    peer-reviewed literature from 2000-2023. Results indicate a 30% decline
    in species diversity in affected regions [1].
    """

    result = await judge.evaluate(academic_content, academic_category)
    print(f"\nAcademic content evaluation:")
    print(f"  Matches category: {result.matches_category}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Feedback: {result.feedback}")

    # Wiki stub content
    stub_content = """
    Python is a high-level programming language. It was created by
    Guido van Rossum and first released in 1991. Python emphasizes
    code readability and uses significant indentation.
    """

    result = await judge.evaluate(stub_content, wiki_stub)
    print(f"\nWiki stub evaluation:")
    print(f"  Matches category: {result.matches_category}")
    print(f"  Word count: {len(stub_content.split())}")
    print(f"  Confidence: {result.confidence:.2f}")

    # 5. Compare content versions
    print("\n=== Comparing Content Versions ===")

    original = "Python is a programming language."

    enhanced = """
    Python is a high-level, interpreted programming language known for
    its simplicity and versatility. Created by Guido van Rossum and first
    released in 1991, Python has become one of the most popular programming
    languages worldwide. It features dynamic typing, automatic memory
    management, and supports multiple programming paradigms including
    object-oriented, functional, and procedural programming.
    """

    comparison = await judge.compare(original, enhanced, wiki_stub)
    print(f"Comparison results:")
    print(f"  Original score: {comparison.original_score:.2f}")
    print(f"  Enhanced score: {comparison.modified_score:.2f}")
    print(f"  Progress: {comparison.progress_percentage:.1f}%")
    print(f"  Recommendation: {comparison.recommendation}")

    # 6. Batch evaluation
    print("\n=== Batch Evaluation ===")

    contents = [
        "Short technical note about APIs.",
        "Comprehensive research paper with multiple citations [1] [2] [3].",
        "Creative story about a dragon and a knight.",
    ]

    results = await judge.batch_evaluate(contents, academic_category)

    for i, (content, result) in enumerate(zip(contents, results), 1):
        print(f"  Content {i}: {content[:50]}...")
        print(f"    Matches: {result.matches_category}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())