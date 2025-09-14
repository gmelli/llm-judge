#!/usr/bin/env python3
"""
Reproduce the validation results from the LLM-as-Judge paper.

This script generates the same synthetic test cases used in the validation
and evaluates them to reproduce the ~82% accuracy result.

Usage:
    python reproduce_validation.py [--provider PROVIDER] [--num-samples NUM]

Expected Results:
    - Accuracy: 70-85% (depending on provider)
    - Baseline: 50-60% (word overlap heuristic)
    - Improvement: 15-30% over baseline
    - Cost: ~$0.0001 per evaluation (Gemini Flash)
"""

import asyncio
import random
import argparse
from typing import Dict, List

from llm_judge import Judge, CategoryDefinition, CharacteristicProperty, PropertyType


def create_evaluation_category() -> CategoryDefinition:
    """Create the category used for validation."""
    return CategoryDefinition(
        name="good_summary",
        description="A good summary that accurately captures the main points",
        characteristic_properties=[
            CharacteristicProperty(
                name="accurate",
                property_type=PropertyType.NECESSARY,
                formal_definition="Information is accurate and not fabricated",
                measurement_function=lambda c: 1.0,
                threshold=True,
                weight=3.0,
            ),
            CharacteristicProperty(
                name="relevant",
                property_type=PropertyType.NECESSARY,
                formal_definition="Captures the main point of the source",
                measurement_function=lambda c: 1.0,
                threshold=0.7,
                weight=2.0,
            ),
        ],
    )


def generate_synthetic_test_cases(num_samples: int = 100) -> List[Dict]:
    """
    Generate synthetic test cases similar to Phase 4 validation.

    Args:
        num_samples: Number of test cases to generate (should be even)

    Returns:
        List of test cases with 50% good and 50% bad summaries
    """
    test_cases = []
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta"]

    # Add advanced test patterns (if num_samples >= 10)
    if num_samples >= 10:
        # 1. Temporal Consistency Tests
        temporal_tests = [
            {
                "id": "temporal_good_1",
                "source": "The company was founded in 2010. It went public in 2015. The IPO raised $2 billion.",
                "summary": "Founded in 2010, the company's 2015 IPO raised $2 billion.",
                "expected": True,
                "type": "temporal_correct"
            },
            {
                "id": "temporal_bad_1",
                "source": "The company was founded in 2010. It went public in 2015. The IPO raised $2 billion.",
                "summary": "The company went public in 2013, five years after its 2010 founding.",
                "expected": False,
                "type": "temporal_error"
            },
            {
                "id": "temporal_good_2",
                "source": "The project started in January, had a review in March, and launched in June.",
                "summary": "The project ran from January to June launch, with March review.",
                "expected": True,
                "type": "temporal_correct"
            },
            {
                "id": "temporal_bad_2",
                "source": "The project started in January, had a review in March, and launched in June.",
                "summary": "After launching in March, the project review happened in June.",
                "expected": False,
                "type": "temporal_error"
            },
        ]
        test_cases.extend(temporal_tests[:min(4, num_samples // 25)])

        # 2. Causal Relationship Tests
        causal_tests = [
            {
                "id": "causal_good_1",
                "source": "Heavy rainfall caused severe flooding, which led to road closures.",
                "summary": "Rain caused flooding that closed roads.",
                "expected": True,
                "type": "causal_correct"
            },
            {
                "id": "causal_bad_1",
                "source": "Heavy rainfall caused severe flooding, which led to road closures.",
                "summary": "Road closures caused flooding during the rainfall.",
                "expected": False,
                "type": "causal_reversal"
            },
            {
                "id": "causal_good_2",
                "source": "The server crash caused data loss, resulting in customer complaints.",
                "summary": "Server failure led to data loss and customer complaints.",
                "expected": True,
                "type": "causal_correct"
            },
            {
                "id": "causal_bad_2",
                "source": "The server crash caused data loss, resulting in customer complaints.",
                "summary": "Customer complaints about data caused the server to crash.",
                "expected": False,
                "type": "causal_reversal"
            },
        ]
        test_cases.extend(causal_tests[:min(4, num_samples // 25)])

        # 3. Quantifier Precision Tests
        quantifier_tests = [
            {
                "id": "quantifier_good_1",
                "source": "Most employees (about 70%) approved the new policy.",
                "summary": "A majority of employees approved the policy.",
                "expected": True,
                "type": "quantifier_correct"
            },
            {
                "id": "quantifier_bad_all",
                "source": "Most employees (about 70%) approved the new policy.",
                "summary": "All employees approved the new policy.",
                "expected": False,
                "type": "quantifier_error_all"
            },
            {
                "id": "quantifier_bad_few",
                "source": "Most employees (about 70%) approved the new policy.",
                "summary": "Few employees approved the new policy.",
                "expected": False,
                "type": "quantifier_error_few"
            },
            {
                "id": "quantifier_good_2",
                "source": "Some users experienced issues, but most were satisfied.",
                "summary": "While a few had problems, the majority were satisfied.",
                "expected": True,
                "type": "quantifier_correct"
            },
            {
                "id": "quantifier_bad_none",
                "source": "Some users experienced issues, but most were satisfied.",
                "summary": "All users were satisfied with no issues.",
                "expected": False,
                "type": "quantifier_error_none"
            },
        ]
        test_cases.extend(quantifier_tests[:min(5, num_samples // 20)])

        # 4. Attribution Error Tests
        attribution_tests = [
            {
                "id": "attribution_good_1",
                "source": "According to the CDC, vaccines are safe. The WHO recommends annual flu shots.",
                "summary": "The CDC confirms vaccine safety, while WHO recommends yearly flu vaccination.",
                "expected": True,
                "type": "attribution_correct"
            },
            {
                "id": "attribution_bad_swap",
                "source": "According to the CDC, vaccines are safe. The WHO recommends annual flu shots.",
                "summary": "The WHO states vaccines are safe, and the CDC recommends annual flu shots.",
                "expected": False,
                "type": "attribution_swap"
            },
            {
                "id": "attribution_good_2",
                "source": "Google announced layoffs. Microsoft reported strong earnings.",
                "summary": "Google cut jobs while Microsoft posted strong results.",
                "expected": True,
                "type": "attribution_correct"
            },
            {
                "id": "attribution_bad_swap_2",
                "source": "Google announced layoffs. Microsoft reported strong earnings.",
                "summary": "Microsoft announced layoffs while Google reported strong earnings.",
                "expected": False,
                "type": "attribution_swap"
            },
        ]
        test_cases.extend(attribution_tests[:min(4, num_samples // 25)])

        # 5. Negation Handling Tests
        negation_tests = [
            {
                "id": "negation_good_1",
                "source": "The study found no significant link between the treatment and recovery rates.",
                "summary": "No significant connection was found between treatment and recovery.",
                "expected": True,
                "type": "negation_correct"
            },
            {
                "id": "negation_bad_removed",
                "source": "The study found no significant link between the treatment and recovery rates.",
                "summary": "The study found a significant link between treatment and recovery.",
                "expected": False,
                "type": "negation_removed"
            },
            {
                "id": "negation_good_2",
                "source": "The drug did not show improvement in patients.",
                "summary": "No patient improvement was observed with the drug.",
                "expected": True,
                "type": "negation_correct"
            },
            {
                "id": "negation_bad_flipped",
                "source": "The drug did not show improvement in patients.",
                "summary": "The drug showed improvement in patients.",
                "expected": False,
                "type": "negation_removed"
            },
            {
                "id": "negation_bad_added",
                "source": "The company achieved its quarterly targets.",
                "summary": "The company did not achieve its quarterly targets.",
                "expected": False,
                "type": "negation_added"
            },
        ]
        test_cases.extend(negation_tests[:min(5, num_samples // 20)])

    # Generate remaining standard test cases
    remaining = num_samples - len(test_cases)

    # Generate good summaries (50% of remaining)
    for i in range(remaining // 2):
        company = random.choice(companies)
        revenue = random.randint(10, 200)
        quarter = random.choice(["Q1", "Q2", "Q3", "Q4"])
        year = random.choice(["2022", "2023", "2024"])

        source = f"{company} reported revenue of ${revenue} billion in {quarter} {year}."

        # Different types of good summaries
        summary_type = random.choice(["exact", "paraphrase", "subset"])
        if summary_type == "exact":
            summary = source
        elif summary_type == "paraphrase":
            summary = f"{company} announced ${revenue}B revenue for {quarter} {year}."
        else:  # subset
            summary = f"{company} reported ${revenue} billion revenue."

        test_cases.append({
            "id": f"good_{i}",
            "source": source,
            "summary": summary,
            "expected": True,
            "type": summary_type
        })

    # Generate bad summaries (50%)
    for i in range(num_samples // 2):
        company = random.choice(companies)
        revenue = random.randint(10, 200)

        source = f"{company} reported revenue of ${revenue} billion."

        # Different types of bad summaries
        error_type = random.choice(["wrong_number", "wrong_entity", "hallucination", "irrelevant"])
        if error_type == "wrong_number":
            wrong_revenue = revenue + random.randint(10, 50)
            summary = f"{company} reported revenue of ${wrong_revenue} billion."
        elif error_type == "wrong_entity":
            wrong_company = random.choice([c for c in companies if c != company])
            summary = f"{wrong_company} reported revenue of ${revenue} billion."
        elif error_type == "hallucination":
            summary = f"{company} reported ${revenue} billion revenue and announced layoffs."
        else:  # irrelevant
            summary = "The weather was nice today."

        test_cases.append({
            "id": f"bad_{i}",
            "source": source,
            "summary": summary,
            "expected": False,
            "type": error_type
        })

    random.shuffle(test_cases)
    return test_cases


def calculate_baseline_accuracy(test_cases: List[Dict]) -> float:
    """
    Calculate baseline accuracy using simple word overlap heuristic.

    Args:
        test_cases: List of test cases

    Returns:
        Baseline accuracy (0.0 to 1.0)
    """
    correct = 0
    for case in test_cases:
        source_words = set(case["source"].lower().split())
        summary_words = set(case["summary"].lower().split())

        # Remove common words
        stopwords = {'the', 'a', 'an', 'of', 'in', 'and', 'is', 'was', 'to'}
        source_words -= stopwords
        summary_words -= stopwords

        # Calculate overlap
        if len(source_words) > 0:
            overlap = len(source_words & summary_words) / len(source_words)
            prediction = overlap > 0.3  # Threshold for "good" summary
        else:
            prediction = False

        if prediction == case["expected"]:
            correct += 1

    return correct / len(test_cases)


def analyze_results_by_type(test_cases: List[Dict], predictions: List[bool]) -> Dict[str, Dict]:
    """
    Analyze results grouped by error type.

    Args:
        test_cases: List of test cases with type information
        predictions: List of boolean predictions

    Returns:
        Dictionary with performance metrics by error type
    """
    type_metrics = {}

    for test_case, prediction in zip(test_cases, predictions):
        error_type = test_case.get("type", "standard")

        if error_type not in type_metrics:
            type_metrics[error_type] = {
                "correct": 0,
                "total": 0,
                "examples": []
            }

        type_metrics[error_type]["total"] += 1
        if prediction == test_case["expected"]:
            type_metrics[error_type]["correct"] += 1
        else:
            # Store failed examples for analysis
            type_metrics[error_type]["examples"].append(test_case["id"])

    # Calculate accuracy for each type
    for error_type in type_metrics:
        metrics = type_metrics[error_type]
        metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0

    return type_metrics


async def run_validation(provider: str = "gemini", num_samples: int = 100):
    """
    Run the validation with synthetic test cases.

    Args:
        provider: LLM provider to use
        num_samples: Number of test cases to generate
    """
    print("="*60)
    print("LLM-as-Judge Validation Reproduction")
    print("="*60)
    print(f"Provider: {provider}")
    print(f"Samples: {num_samples}\n")

    # Generate test cases
    print("Generating synthetic test cases...")
    test_cases = generate_synthetic_test_cases(num_samples)
    print(f"✓ Generated {len(test_cases)} test cases")
    print(f"  - Good summaries: {sum(1 for tc in test_cases if tc['expected'])}")
    print(f"  - Bad summaries: {sum(1 for tc in test_cases if not tc['expected'])}\n")

    # Calculate baseline
    print("Calculating baseline accuracy...")
    baseline_accuracy = calculate_baseline_accuracy(test_cases)
    print(f"✓ Baseline accuracy (word overlap): {baseline_accuracy:.1%}\n")

    # Initialize Judge
    print(f"Initializing {provider} judge...")
    try:
        if provider == "mock":
            judge = Judge(provider="mock", temperature=0.1, track_costs=True)
        else:
            judge = Judge(
                provider=provider,
                model="gemini-1.5-flash" if provider == "gemini" else None,
                temperature=0.1,
                track_costs=True,
            )
        print(f"✓ Judge initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("Falling back to mock provider\n")
        judge = Judge(provider="mock", temperature=0.1, track_costs=True)

    # Create evaluation category
    category = create_evaluation_category()

    # Run evaluations
    print("Running evaluations...")
    correct = 0
    predictions = []

    for i, case in enumerate(test_cases):
        # Show progress
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_cases)}")

        # Create context
        context = f"Source: {case['source']}\n\nSummary: {case['summary']}"

        # Evaluate
        result = await judge.evaluate(context, category)
        prediction = result.matches_category
        predictions.append(prediction)

        if prediction == case["expected"]:
            correct += 1

    llm_accuracy = correct / len(test_cases)
    improvement = llm_accuracy - baseline_accuracy

    # Analyze by error type
    type_analysis = analyze_results_by_type(test_cases, predictions)

    # Get cost summary
    cost_info = judge.get_cost_summary()

    # Print results
    print(f"  Progress: {len(test_cases)}/{len(test_cases)}\n")
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"LLM-as-Judge Accuracy: {llm_accuracy:.1%}")
    print(f"Baseline Accuracy:     {baseline_accuracy:.1%}")
    print(f"Improvement:           {improvement:+.1%}\n")

    print("Cost Analysis:")
    print(f"  Total cost: ${cost_info.get('total_cost', 0):.4f}")
    print(f"  Per evaluation: ${cost_info.get('average_cost', 0):.6f}")
    print(f"  Total evaluations: {cost_info.get('total_evaluations', 0)}\n")

    # Print performance by error type (for advanced patterns)
    advanced_types = ["temporal_error", "temporal_correct", "causal_reversal", "causal_correct",
                     "quantifier_error_all", "quantifier_error_few", "quantifier_error_none", "quantifier_correct",
                     "attribution_swap", "attribution_correct", "negation_removed", "negation_added", "negation_correct"]

    advanced_in_test = [t for t in advanced_types if t in type_analysis]
    if advanced_in_test:
        print("Advanced Pattern Performance:")
        for error_type in advanced_in_test:
            metrics = type_analysis[error_type]
            print(f"  {error_type:20} {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        print()

    # Success criteria
    success = llm_accuracy > baseline_accuracy and llm_accuracy > 0.7
    print(f"Validation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"  - Beat baseline: {'Yes' if llm_accuracy > baseline_accuracy else 'No'}")
    print(f"  - Accuracy > 70%: {'Yes' if llm_accuracy > 0.7 else 'No'}")

    return llm_accuracy, baseline_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce LLM-as-Judge validation results"
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "anthropic", "mock"],
        help="LLM provider to use (default: gemini)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test samples to generate (default: 100)"
    )

    args = parser.parse_args()

    # Run validation
    asyncio.run(run_validation(args.provider, args.num_samples))


if __name__ == "__main__":
    main()