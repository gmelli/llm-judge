"""
Automated benchmark runner for demonstrating LLM-as-Judge value.

This script runs evaluations on standard NLG benchmarks and compares
LLM-as-Judge results with traditional ROUGE/BLEU metrics.
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Install datasets: pip install datasets")
    DATASETS_AVAILABLE = False

# Traditional metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    METRICS_AVAILABLE = True
except ImportError:
    print("Install metrics: pip install rouge-score nltk")
    METRICS_AVAILABLE = False

from llm_judge import Judge, CategoryDefinition, CharacteristicProperty, PropertyType


class BenchmarkRunner:
    """Runs LLM-as-Judge on standard benchmarks with comparison to traditional metrics."""

    # Benchmark configurations
    BENCHMARK_CONFIGS = {
        "summeval": {
            "dataset": "mteb/summeval",
            "source_key": "text",
            "reference_key": "human_summaries",
            "generated_key": "machine_summaries",
            "limit": 100,
            "description": "Expert-annotated CNN/DailyMail summaries",
        },
        "xsum": {
            "dataset": "xsum",
            "source_key": "document",
            "reference_key": "summary",
            "generated_key": None,  # Need to generate
            "limit": 1000,
            "description": "BBC extreme summarization",
        },
        "reddit_tifu": {
            "dataset": "reddit_tifu",
            "source_key": "documents",
            "reference_key": "tldr",
            "generated_key": None,
            "limit": 1000,
            "description": "Reddit TL;DR summaries",
        },
        "qags": {
            "dataset": "xsum",  # QAGS is based on XSum
            "source_key": "document",
            "reference_key": "summary",
            "generated_key": None,
            "limit": 500,
            "description": "Hallucination detection benchmark",
            "special": "hallucination_focus",
        },
        "cnn_dailymail": {
            "dataset": "cnn_dailymail",
            "version": "3.0.0",
            "source_key": "article",
            "reference_key": "highlights",
            "generated_key": None,
            "limit": 1000,
            "description": "News article summarization",
        },
        "multinews": {
            "dataset": "multi_news",
            "source_key": "document",
            "reference_key": "summary",
            "generated_key": None,
            "limit": 500,
            "description": "Multi-document summarization",
        },
        "samsum": {
            "dataset": "samsum",
            "source_key": "dialogue",
            "reference_key": "summary",
            "generated_key": None,
            "limit": 500,
            "description": "Dialogue summarization",
        },
    }

    def __init__(self, judge: Optional[Judge] = None, use_cache: bool = True):
        """Initialize benchmark runner.

        Args:
            judge: LLM-as-Judge instance (defaults to Gemini Flash)
            use_cache: Whether to cache evaluations
        """
        self.judge = judge or Judge(
            provider="gemini",
            model="gemini-1.5-flash",
            temperature=0.1,
            track_costs=True,
        )
        self.use_cache = use_cache
        self.cache_dir = Path("benchmark_cache")
        self.cache_dir.mkdir(exist_ok=True)

        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
            self.smoothing = SmoothingFunction()

    def create_evaluation_category(self, benchmark_name: str) -> CategoryDefinition:
        """Create appropriate evaluation category for benchmark.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            CategoryDefinition tailored to the benchmark
        """
        if "qags" in benchmark_name.lower() or "hallucination" in benchmark_name.lower():
            # Focus on factual consistency
            return CategoryDefinition(
                name="factually_consistent",
                description="Content that is factually consistent with source",
                characteristic_properties=[
                    CharacteristicProperty(
                        name="no_hallucination",
                        property_type=PropertyType.NECESSARY,
                        formal_definition="Contains no fabricated facts",
                        measurement_function=lambda c: 1.0,
                        threshold=True,
                        weight=5.0,
                    ),
                    CharacteristicProperty(
                        name="source_grounded",
                        property_type=PropertyType.NECESSARY,
                        formal_definition="All claims traceable to source",
                        measurement_function=lambda c: 1.0,
                        threshold=0.95,
                        weight=3.0,
                    ),
                ],
            )
        elif "dialogue" in benchmark_name.lower() or "samsum" in benchmark_name.lower():
            # Focus on conversational coherence
            return CategoryDefinition(
                name="coherent_dialogue_summary",
                description="Summary capturing conversational flow",
                characteristic_properties=[
                    CharacteristicProperty(
                        name="speaker_attribution",
                        property_type=PropertyType.TYPICAL,
                        formal_definition="Correctly attributes statements to speakers",
                        measurement_function=lambda c: 1.0,
                        threshold=0.8,
                        weight=2.0,
                    ),
                    CharacteristicProperty(
                        name="conversation_flow",
                        property_type=PropertyType.NECESSARY,
                        formal_definition="Maintains logical conversation progression",
                        measurement_function=lambda c: 1.0,
                        threshold=0.7,
                        weight=2.0,
                    ),
                ],
            )
        else:
            # General high-quality summary
            return CategoryDefinition(
                name="high_quality_summary",
                description="Well-formed, accurate, and coherent summary",
                characteristic_properties=[
                    CharacteristicProperty(
                        name="factual_accuracy",
                        property_type=PropertyType.NECESSARY,
                        formal_definition="Facts match source document",
                        measurement_function=lambda c: 1.0,
                        threshold=0.9,
                        weight=3.0,
                    ),
                    CharacteristicProperty(
                        name="coherence",
                        property_type=PropertyType.NECESSARY,
                        formal_definition="Logical flow and structure",
                        measurement_function=lambda c: '.' in c,
                        threshold=True,
                        weight=2.0,
                    ),
                    CharacteristicProperty(
                        name="completeness",
                        property_type=PropertyType.TYPICAL,
                        formal_definition="Covers key points",
                        measurement_function=lambda c: 1.0,
                        threshold=0.7,
                        weight=1.5,
                    ),
                ],
            )

    async def load_benchmark_data(
        self,
        benchmark_name: str,
        limit: Optional[int] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Load benchmark dataset.

        Args:
            benchmark_name: Name of benchmark to load
            limit: Maximum number of samples

        Returns:
            Tuple of (sources, references, generated summaries)
        """
        if not DATASETS_AVAILABLE:
            # Return dummy data for testing
            return (
                ["Source text"] * 10,
                ["Reference summary"] * 10,
                ["Generated summary"] * 10,
            )

        config = self.BENCHMARK_CONFIGS[benchmark_name]

        # Load dataset
        if "version" in config:
            dataset = load_dataset(config["dataset"], config["version"], split="test")
        else:
            dataset = load_dataset(config["dataset"], split="test")

        # Limit samples
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        elif config.get("limit"):
            dataset = dataset.select(range(min(config["limit"], len(dataset))))

        # Extract texts
        sources = dataset[config["source_key"]]
        references = dataset[config["reference_key"]]

        # Handle generated summaries
        if config["generated_key"]:
            generated = dataset[config["generated_key"]]
        else:
            # For demo, use references with slight modification
            # In production, you'd load actual model outputs
            generated = [ref + " [Generated]" for ref in references]

        return sources, references, generated

    def calculate_traditional_metrics(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """Calculate ROUGE and BLEU scores.

        Args:
            generated: Generated summary
            reference: Reference summary

        Returns:
            Dictionary of metric scores
        """
        if not METRICS_AVAILABLE:
            return {"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4, "bleu": 0.35}

        # ROUGE scores
        scores = self.rouge_scorer.score(reference, generated)
        rouge_scores = {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }

        # BLEU score
        ref_tokens = reference.split()
        gen_tokens = generated.split()
        bleu = sentence_bleu(
            [ref_tokens],
            gen_tokens,
            smoothing_function=self.smoothing.method1
        )

        return {**rouge_scores, "bleu": bleu}

    async def evaluate_sample(
        self,
        source: str,
        generated: str,
        reference: str,
        category: CategoryDefinition,
    ) -> Dict:
        """Evaluate a single sample.

        Args:
            source: Source text
            generated: Generated summary
            reference: Reference summary
            category: Evaluation category

        Returns:
            Dictionary with all evaluation results
        """
        # Traditional metrics
        traditional = self.calculate_traditional_metrics(generated, reference)

        # LLM-as-Judge evaluation
        context = f"Source:\n{source}\n\nSummary:\n{generated}"
        llm_result = await self.judge.evaluate(context, category)

        return {
            "traditional_metrics": traditional,
            "llm_judge": {
                "score": llm_result.membership_score,
                "matches": llm_result.matches_category,
                "confidence": llm_result.confidence,
                "property_scores": {
                    name: score.raw_score
                    for name, score in llm_result.property_scores.items()
                },
                "feedback": llm_result.feedback,
            },
            "cost": llm_result.evaluation_cost,
        }

    async def run_benchmark(
        self,
        benchmark_name: str,
        samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """Run evaluation on a benchmark.

        Args:
            benchmark_name: Name of benchmark to run
            samples: Number of samples to evaluate
            verbose: Whether to print progress

        Returns:
            Dictionary with benchmark results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {benchmark_name.upper()}")
            print(f"Description: {self.BENCHMARK_CONFIGS[benchmark_name]['description']}")
            print(f"{'='*60}")

        # Load data
        sources, references, generated = await self.load_benchmark_data(
            benchmark_name, samples
        )

        # Create evaluation category
        category = self.create_evaluation_category(benchmark_name)

        # Evaluate samples
        results = []
        for i, (src, ref, gen) in enumerate(zip(sources, references, generated)):
            if verbose and i % 10 == 0:
                print(f"Evaluating sample {i+1}/{len(sources)}...")

            result = await self.evaluate_sample(src, gen, ref, category)
            results.append(result)

        # Analyze results
        analysis = self.analyze_results(results, benchmark_name)

        # Get cost summary
        total_cost = sum(r.get("cost", 0) for r in results)

        return {
            "benchmark": benchmark_name,
            "num_samples": len(results),
            "results": results,
            "analysis": analysis,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        }

    def analyze_results(self, results: List[Dict], benchmark_name: str) -> Dict:
        """Analyze benchmark results.

        Args:
            results: List of evaluation results
            benchmark_name: Name of the benchmark

        Returns:
            Dictionary with analysis
        """
        # Extract scores
        rouge1_scores = [r["traditional_metrics"]["rouge1"] for r in results]
        rouge2_scores = [r["traditional_metrics"]["rouge2"] for r in results]
        rougeL_scores = [r["traditional_metrics"]["rougeL"] for r in results]
        bleu_scores = [r["traditional_metrics"]["bleu"] for r in results]
        llm_scores = [r["llm_judge"]["score"] for r in results]
        llm_matches = [r["llm_judge"]["matches"] for r in results]

        # Basic statistics
        analysis = {
            "traditional_metrics": {
                "rouge1": {"mean": np.mean(rouge1_scores), "std": np.std(rouge1_scores)},
                "rouge2": {"mean": np.mean(rouge2_scores), "std": np.std(rouge2_scores)},
                "rougeL": {"mean": np.mean(rougeL_scores), "std": np.std(rougeL_scores)},
                "bleu": {"mean": np.mean(bleu_scores), "std": np.std(bleu_scores)},
            },
            "llm_judge": {
                "mean_score": np.mean(llm_scores),
                "std_score": np.std(llm_scores),
                "match_rate": np.mean(llm_matches),
            },
        }

        # Find interesting cases
        high_rouge_low_llm = []
        low_rouge_high_llm = []

        for i, r in enumerate(results):
            rouge_avg = np.mean([
                r["traditional_metrics"]["rouge1"],
                r["traditional_metrics"]["rouge2"],
                r["traditional_metrics"]["rougeL"],
            ])
            llm_score = r["llm_judge"]["score"]

            if rouge_avg > 0.5 and llm_score < 0.5:
                high_rouge_low_llm.append(i)
            elif rouge_avg < 0.3 and llm_score > 0.7:
                low_rouge_high_llm.append(i)

        analysis["interesting_cases"] = {
            "high_rouge_low_quality": len(high_rouge_low_llm),
            "low_rouge_high_quality": len(low_rouge_high_llm),
            "high_rouge_low_quality_indices": high_rouge_low_llm[:5],  # First 5
            "low_rouge_high_quality_indices": low_rouge_high_llm[:5],
        }

        # Correlation analysis
        if len(results) > 3:
            from scipy import stats

            correlations = {}
            correlations["rouge1_vs_llm"] = stats.pearsonr(rouge1_scores, llm_scores)[0]
            correlations["rouge2_vs_llm"] = stats.pearsonr(rouge2_scores, llm_scores)[0]
            correlations["rougeL_vs_llm"] = stats.pearsonr(rougeL_scores, llm_scores)[0]
            correlations["bleu_vs_llm"] = stats.pearsonr(bleu_scores, llm_scores)[0]

            analysis["correlations"] = correlations

        return analysis

    def print_analysis(self, benchmark_result: Dict):
        """Print formatted analysis of benchmark results.

        Args:
            benchmark_result: Result dictionary from run_benchmark
        """
        print(f"\n{'='*60}")
        print(f"RESULTS: {benchmark_result['benchmark'].upper()}")
        print(f"Samples: {benchmark_result['num_samples']}")
        print(f"{'='*60}")

        analysis = benchmark_result["analysis"]

        # Traditional metrics
        print("\nTRADITIONAL METRICS:")
        for metric, stats in analysis["traditional_metrics"].items():
            print(f"  {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")

        # LLM-as-Judge
        print("\nLLM-AS-JUDGE:")
        print(f"  Mean Score: {analysis['llm_judge']['mean_score']:.3f}")
        print(f"  Match Rate: {analysis['llm_judge']['match_rate']:.1%}")

        # Interesting cases
        cases = analysis["interesting_cases"]
        print("\nINTERESTING FINDINGS:")
        print(f"  High ROUGE but Low Quality: {cases['high_rouge_low_quality']} samples")
        print(f"  Low ROUGE but High Quality: {cases['low_rouge_high_quality']} samples")

        # Correlations
        if "correlations" in analysis:
            print("\nCORRELATIONS (Traditional vs LLM-as-Judge):")
            for metric, corr in analysis["correlations"].items():
                print(f"  {metric}: {corr:.3f}")

        # Cost
        print(f"\nEVALUATION COST: ${benchmark_result['total_cost']:.4f}")
        print(f"  Per sample: ${benchmark_result['total_cost']/benchmark_result['num_samples']:.6f}")

    async def run_all_benchmarks(
        self,
        benchmarks: Optional[List[str]] = None,
        samples_per_benchmark: int = 100,
    ) -> Dict:
        """Run multiple benchmarks and generate comparison.

        Args:
            benchmarks: List of benchmark names (None = all)
            samples_per_benchmark: Samples to evaluate per benchmark

        Returns:
            Dictionary with all results
        """
        if benchmarks is None:
            benchmarks = ["summeval", "xsum", "reddit_tifu", "qags"]

        all_results = {}

        for benchmark in benchmarks:
            if benchmark not in self.BENCHMARK_CONFIGS:
                print(f"Unknown benchmark: {benchmark}")
                continue

            result = await self.run_benchmark(
                benchmark,
                samples=samples_per_benchmark
            )
            all_results[benchmark] = result
            self.print_analysis(result)

        # Generate comparison report
        comparison = self.generate_comparison(all_results)

        return {
            "benchmarks": all_results,
            "comparison": comparison,
            "cost_report": self.judge.get_cost_report() if self.judge.track_costs else None,
        }

    def generate_comparison(self, all_results: Dict) -> Dict:
        """Generate comparison across benchmarks.

        Args:
            all_results: Results from all benchmarks

        Returns:
            Comparison analysis
        """
        comparison = {
            "summary": [],
            "best_llm_value": None,
            "total_cost": 0,
        }

        max_gap = 0
        best_benchmark = None

        for name, result in all_results.items():
            analysis = result["analysis"]

            # Calculate value gap
            if "correlations" in analysis:
                avg_correlation = np.mean(list(analysis["correlations"].values()))
                value_gap = 1 - avg_correlation  # Higher gap = more value

                if value_gap > max_gap:
                    max_gap = value_gap
                    best_benchmark = name

            # Add to summary
            interesting = analysis["interesting_cases"]
            comparison["summary"].append({
                "benchmark": name,
                "samples": result["num_samples"],
                "high_rouge_low_quality": interesting["high_rouge_low_quality"],
                "low_rouge_high_quality": interesting["low_rouge_high_quality"],
                "cost": result["total_cost"],
            })

            comparison["total_cost"] += result["total_cost"]

        comparison["best_llm_value"] = best_benchmark
        comparison["max_value_gap"] = max_gap

        return comparison


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge benchmarks"
    )
    parser.add_argument(
        "--dataset",
        choices=list(BenchmarkRunner.BENCHMARK_CONFIGS.keys()),
        help="Specific dataset to evaluate"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-flash",
        help="LLM model to use for evaluation"
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        help="LLM provider (openai, anthropic, gemini)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Initialize judge
    judge = Judge(
        provider=args.provider,
        model=args.model,
        temperature=0.1,
        track_costs=True,
    )

    # Initialize runner
    runner = BenchmarkRunner(judge=judge)

    # Run benchmarks
    async def run():
        if args.all:
            results = await runner.run_all_benchmarks(samples_per_benchmark=args.samples)
        elif args.dataset:
            result = await runner.run_benchmark(args.dataset, samples=args.samples)
            runner.print_analysis(result)
            results = {args.dataset: result}
        else:
            # Default: run top 4 benchmarks
            results = await runner.run_all_benchmarks(
                benchmarks=["summeval", "qags", "xsum", "reddit_tifu"],
                samples_per_benchmark=args.samples
            )

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

        # Print cost report
        if judge.track_costs:
            print("\n" + judge.get_cost_report())

        return results

    # Run async
    results = asyncio.run(run())

    # Print final summary
    if "comparison" in results:
        print(f"\n{'='*60}")
        print("BENCHMARK COMPARISON SUMMARY")
        print(f"{'='*60}")

        comparison = results["comparison"]
        print(f"\nBest value from LLM-as-Judge: {comparison['best_llm_value']}")
        print(f"Maximum value gap: {comparison['max_value_gap']:.2f}")
        print(f"Total evaluation cost: ${comparison['total_cost']:.4f}")


if __name__ == "__main__":
    main()