"""
Evaluate summarization models using LLM-as-Judge alongside ROUGE scores.

This script shows how to use LLM-as-Judge with real summarization datasets
like CNN/DailyMail or XSum, demonstrating value beyond ROUGE scores.
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Dataset loading (install with: pip install datasets)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Install datasets: pip install datasets")
    DATASETS_AVAILABLE = False

# ROUGE scoring
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Install rouge-score: pip install rouge-score")
    ROUGE_AVAILABLE = False

from llm_judge import (
    Judge,
    CategoryDefinition,
    CharacteristicProperty,
    PropertyType,
    Range,
    Example,
)


class SummarizationEvaluator:
    """Evaluator for summarization quality using LLM-as-Judge."""

    @staticmethod
    def create_summary_quality_category() -> CategoryDefinition:
        """Comprehensive category for high-quality summaries."""
        return CategoryDefinition(
            name="high_quality_summary",
            description="A summary that accurately captures key information with good coherence",
            characteristic_properties=[
                # Factual accuracy
                CharacteristicProperty(
                    name="factual_accuracy",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="All facts in summary are present in or directly inferable from source",
                    measurement_function=lambda c: 1.0,  # LLM evaluates
                    threshold=0.9,
                    weight=3.0,
                ),
                # Coverage
                CharacteristicProperty(
                    name="key_point_coverage",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Covers all major points from the source article",
                    measurement_function=lambda c: 1.0,
                    threshold=0.75,
                    weight=2.0,
                ),
                # Conciseness
                CharacteristicProperty(
                    name="conciseness",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Summary is appropriately brief without losing essential information",
                    measurement_function=lambda c: len(c.split()) < 150,  # Most summaries should be <150 words
                    threshold=True,
                    weight=1.0,
                ),
                # Coherence
                CharacteristicProperty(
                    name="coherence",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Summary flows logically and is easy to understand",
                    measurement_function=lambda c: '.' in c and len(c.split('.')) > 1,
                    threshold=True,
                    weight=1.5,
                ),
                # No redundancy
                CharacteristicProperty(
                    name="non_redundancy",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Avoids repeating the same information",
                    measurement_function=lambda c: 1.0,
                    threshold=0.8,
                    weight=1.0,
                ),
            ],
            positive_examples=[
                Example(
                    content="Researchers at MIT developed a new AI system that can detect diseases early. The system analyzes medical images with 95% accuracy, outperforming human doctors in trials.",
                    category="high_quality_summary",
                    properties={"factual_accuracy": 1.0, "conciseness": 1.0},
                    confidence=0.95,
                ),
            ],
            negative_examples=[
                Example(
                    content="There was something about AI. Doctors were mentioned. MIT did research. The accuracy was good. AI is important for medicine.",
                    category="high_quality_summary",
                    properties={"coherence": 0.3, "key_point_coverage": 0.4},
                    confidence=0.95,
                ),
            ],
        )

    @staticmethod
    def create_hallucination_detection_category() -> CategoryDefinition:
        """Category specifically for detecting hallucinations."""
        return CategoryDefinition(
            name="hallucination_free",
            description="Summary contains only information from the source without fabrication",
            characteristic_properties=[
                CharacteristicProperty(
                    name="no_fabricated_facts",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Contains no facts, names, numbers, or claims not in the source",
                    measurement_function=lambda c: 1.0,
                    threshold=True,
                    weight=5.0,  # Heavily weighted
                ),
                CharacteristicProperty(
                    name="no_unsupported_inferences",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Makes no conclusions that aren't directly supported by source",
                    measurement_function=lambda c: 1.0,
                    threshold=True,
                    weight=3.0,
                ),
            ],
        )


async def evaluate_summary_batch(
    judge: Judge,
    articles: List[str],
    summaries: List[str],
    references: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Evaluate a batch of summaries using LLM-as-Judge.

    Args:
        judge: LLM-as-Judge instance
        articles: Source articles
        summaries: Generated summaries to evaluate
        references: Optional reference summaries for comparison

    Returns:
        DataFrame with evaluation results
    """
    evaluator = SummarizationEvaluator()
    quality_category = evaluator.create_summary_quality_category()
    hallucination_category = evaluator.create_hallucination_detection_category()

    results = []

    for i, (article, summary) in enumerate(zip(articles, summaries)):
        # Create evaluation context
        context = f"Article:\n{article}\n\nSummary:\n{summary}"

        # Evaluate quality
        quality_result = await judge.evaluate(context, quality_category)

        # Check for hallucinations
        hallucination_result = await judge.evaluate(context, hallucination_category)

        # Calculate ROUGE if reference available
        rouge_scores = {}
        if ROUGE_AVAILABLE and references and i < len(references):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(references[i], summary)
            rouge_scores = {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }

        result = {
            'summary_id': i,
            'quality_score': quality_result.membership_score,
            'quality_match': quality_result.matches_category,
            'factual_accuracy': quality_result.property_scores.get('factual_accuracy', {}).get('raw_score', 0),
            'coherence': quality_result.property_scores.get('coherence', {}).get('raw_score', 0),
            'coverage': quality_result.property_scores.get('key_point_coverage', {}).get('raw_score', 0),
            'hallucination_free': hallucination_result.matches_category,
            'hallucination_score': hallucination_result.membership_score,
            'word_count': len(summary.split()),
            **rouge_scores,
            'feedback': quality_result.feedback,
        }

        results.append(result)

    return pd.DataFrame(results)


async def evaluate_model_outputs(
    model_name: str,
    dataset_name: str = "cnn_dailymail",
    num_samples: int = 100,
    use_cached: bool = True,
) -> Dict:
    """Evaluate model outputs on a standard dataset.

    Args:
        model_name: Name of the model being evaluated
        dataset_name: Dataset to use (cnn_dailymail, xsum, etc.)
        num_samples: Number of samples to evaluate
        use_cached: Whether to use cached model outputs

    Returns:
        Dictionary with evaluation results and analysis
    """
    # Load dataset
    if DATASETS_AVAILABLE:
        if dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            article_key = "article"
            summary_key = "highlights"
        elif dataset_name == "xsum":
            dataset = load_dataset("xsum", split="test")
            article_key = "document"
            summary_key = "summary"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Sample data
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        articles = dataset[article_key]
        reference_summaries = dataset[summary_key]
    else:
        # Use dummy data for demonstration
        articles = [
            "Scientists at Stanford University have developed a new artificial intelligence system that can predict earthquakes with unprecedented accuracy. The system uses machine learning to analyze seismic data patterns."
        ] * num_samples
        reference_summaries = [
            "Stanford scientists created an AI system that predicts earthquakes by analyzing seismic patterns."
        ] * num_samples

    # Generate or load model summaries
    if use_cached and Path(f"{model_name}_summaries.json").exists():
        with open(f"{model_name}_summaries.json", "r") as f:
            model_summaries = json.load(f)
    else:
        # In practice, you would generate these using the model
        # For demo, we'll create variations
        model_summaries = [
            f"AI system at Stanford predicts earthquakes using ML on seismic data."
            for _ in range(num_samples)
        ]

    # Initialize judge (use real provider for production)
    judge = Judge(
        provider="gemini",
        model="gemini-1.5-flash",
        temperature=0.1,
        track_costs=True,
    )

    # Evaluate summaries
    print(f"Evaluating {num_samples} summaries from {model_name}...")
    df_results = await evaluate_summary_batch(
        judge,
        articles,
        model_summaries,
        reference_summaries,
    )

    # Analyze results
    analysis = analyze_results(df_results, model_name)

    # Get cost report
    cost_report = judge.get_cost_report() if judge.track_costs else None

    return {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "results": df_results.to_dict(),
        "analysis": analysis,
        "cost_report": cost_report,
    }


def analyze_results(df: pd.DataFrame, model_name: str) -> Dict:
    """Analyze evaluation results to show LLM-as-Judge value."""
    analysis = {}

    # Basic statistics
    analysis["quality_stats"] = {
        "mean_quality_score": df["quality_score"].mean(),
        "std_quality_score": df["quality_score"].std(),
        "high_quality_pct": (df["quality_match"] == True).mean() * 100,
    }

    # Hallucination analysis
    analysis["hallucination_stats"] = {
        "hallucination_free_pct": (df["hallucination_free"] == True).mean() * 100,
        "mean_hallucination_score": df["hallucination_score"].mean(),
    }

    # Compare with ROUGE if available
    if "rouge1" in df.columns:
        # Find discrepancies
        high_rouge_low_quality = df[
            (df["rouge1"] > df["rouge1"].median()) &
            (df["quality_score"] < 0.5)
        ]
        low_rouge_high_quality = df[
            (df["rouge1"] < df["rouge1"].median()) &
            (df["quality_score"] > 0.7)
        ]

        analysis["rouge_comparison"] = {
            "rouge1_mean": df["rouge1"].mean(),
            "rouge2_mean": df["rouge2"].mean(),
            "rougeL_mean": df["rougeL"].mean(),
            "high_rouge_low_quality_count": len(high_rouge_low_quality),
            "low_rouge_high_quality_count": len(low_rouge_high_quality),
        }

        # Correlation analysis
        from scipy import stats
        if len(df) > 3:
            corr_rouge_quality = stats.pearsonr(df["rouge1"], df["quality_score"])[0]
            corr_rouge_hallucination = stats.pearsonr(df["rouge1"], df["hallucination_score"])[0]

            analysis["correlations"] = {
                "rouge1_vs_quality": corr_rouge_quality,
                "rouge1_vs_hallucination": corr_rouge_hallucination,
            }

    # Key insights
    insights = []

    if analysis["hallucination_stats"]["hallucination_free_pct"] < 80:
        insights.append(f"⚠️ {model_name} shows hallucination issues in {100 - analysis['hallucination_stats']['hallucination_free_pct']:.1f}% of summaries")

    if "rouge_comparison" in analysis:
        if analysis["rouge_comparison"]["high_rouge_low_quality_count"] > 0:
            insights.append(f"📊 Found {analysis['rouge_comparison']['high_rouge_low_quality_count']} summaries with high ROUGE but low quality")

        if analysis["rouge_comparison"]["low_rouge_high_quality_count"] > 0:
            insights.append(f"✨ Found {analysis['rouge_comparison']['low_rouge_high_quality_count']} high-quality summaries with low ROUGE (good paraphrasing)")

    analysis["key_insights"] = insights

    return analysis


def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {results['model']} on {results['dataset']}")
    print("=" * 70)

    # Quality metrics
    quality = results["analysis"]["quality_stats"]
    print(f"\nQUALITY METRICS:")
    print(f"  Mean Quality Score: {quality['mean_quality_score']:.3f}")
    print(f"  High Quality: {quality['high_quality_pct']:.1f}%")

    # Hallucination detection
    hallucination = results["analysis"]["hallucination_stats"]
    print(f"\nHALLUCINATION DETECTION:")
    print(f"  Hallucination-Free: {hallucination['hallucination_free_pct']:.1f}%")

    # ROUGE comparison if available
    if "rouge_comparison" in results["analysis"]:
        rouge = results["analysis"]["rouge_comparison"]
        print(f"\nROUGE SCORES:")
        print(f"  ROUGE-1: {rouge['rouge1_mean']:.3f}")
        print(f"  ROUGE-2: {rouge['rouge2_mean']:.3f}")
        print(f"  ROUGE-L: {rouge['rougeL_mean']:.3f}")

        print(f"\nVALUE BEYOND ROUGE:")
        print(f"  High ROUGE but Low Quality: {rouge['high_rouge_low_quality_count']} samples")
        print(f"  Low ROUGE but High Quality: {rouge['low_rouge_high_quality_count']} samples")

    # Correlations
    if "correlations" in results["analysis"]:
        corr = results["analysis"]["correlations"]
        print(f"\nCORRELATIONS:")
        print(f"  ROUGE-1 vs Quality: {corr['rouge1_vs_quality']:.3f}")
        print(f"  ROUGE-1 vs Hallucination: {corr['rouge1_vs_hallucination']:.3f}")

    # Key insights
    if results["analysis"]["key_insights"]:
        print(f"\nKEY INSIGHTS:")
        for insight in results["analysis"]["key_insights"]:
            print(f"  {insight}")

    # Cost report
    if results.get("cost_report"):
        print(f"\n{results['cost_report']}")

    print("\n" + "=" * 70)


async def main():
    """Run evaluation on summarization models."""
    # Example: Evaluate different models
    models_to_evaluate = [
        "gpt-3.5-turbo",
        "llama-2-7b",
        "t5-base",
    ]

    for model_name in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")

        try:
            results = await evaluate_model_outputs(
                model_name=model_name,
                dataset_name="cnn_dailymail",
                num_samples=50,  # Use more for real evaluation
            )

            # Print report
            print_evaluation_report(results)

            # Save detailed results
            with open(f"{model_name}_evaluation.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")


if __name__ == "__main__":
    # For demo purposes, show how it would work
    print("Summarization Evaluation with LLM-as-Judge")
    print("This demonstrates evaluation beyond ROUGE scores")

    # Run async main
    asyncio.run(main())