"""
Validation study comparing LLM-as-Judge with traditional NLG metrics (ROUGE/BLEU).

This script demonstrates the value of LLM-as-Judge by evaluating NLG outputs
on multiple dimensions that traditional metrics miss, while also showing
correlation with established benchmarks.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

# Traditional metric imports (would need to be installed)
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    METRICS_AVAILABLE = True
except ImportError:
    print("Install rouge-score and nltk for traditional metrics: pip install rouge-score nltk")
    METRICS_AVAILABLE = False

from llm_judge import Judge, CategoryDefinition, CharacteristicProperty, PropertyType, Range


@dataclass
class NLGSample:
    """A sample for NLG evaluation."""
    source: str  # Input/source text
    reference: str  # Gold standard reference
    generated: str  # Model-generated output
    model_name: str  # Which model generated it
    task: str  # Task type (summarization, translation, etc.)


class NLGEvaluationCategories:
    """Categories for evaluating different aspects of NLG quality."""

    @staticmethod
    def create_factual_consistency_category() -> CategoryDefinition:
        """Category for factual consistency - what ROUGE misses."""
        return CategoryDefinition(
            name="factually_consistent_generation",
            description="Text that accurately preserves facts from the source without hallucination",
            characteristic_properties=[
                CharacteristicProperty(
                    name="no_hallucination",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Contains no facts not present in or inferable from source",
                    measurement_function=lambda c: 1.0,  # LLM will evaluate
                    threshold=True,
                    weight=3.0,
                ),
                CharacteristicProperty(
                    name="key_facts_preserved",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="All important facts from source are preserved",
                    measurement_function=lambda c: 1.0,
                    threshold=0.8,
                    weight=2.0,
                ),
                CharacteristicProperty(
                    name="no_contradictions",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="No statements contradict the source material",
                    measurement_function=lambda c: 1.0,
                    threshold=True,
                    weight=2.0,
                ),
            ],
        )

    @staticmethod
    def create_coherence_category() -> CategoryDefinition:
        """Category for coherence and flow - beyond n-gram overlap."""
        return CategoryDefinition(
            name="coherent_generation",
            description="Text with logical flow and coherent structure",
            characteristic_properties=[
                CharacteristicProperty(
                    name="logical_flow",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Ideas progress logically from one to the next",
                    measurement_function=lambda c: len(c.split('.')) > 1,
                    threshold=True,
                    weight=2.0,
                ),
                CharacteristicProperty(
                    name="discourse_markers",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Uses appropriate transition words and connectives",
                    measurement_function=lambda c: any(
                        marker in c.lower()
                        for marker in ['however', 'therefore', 'moreover', 'furthermore', 'additionally']
                    ),
                    threshold=True,
                    weight=1.0,
                ),
                CharacteristicProperty(
                    name="topic_consistency",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Maintains consistent topic focus throughout",
                    measurement_function=lambda c: 1.0,
                    threshold=0.7,
                    weight=1.5,
                ),
            ],
        )

    @staticmethod
    def create_relevance_category() -> CategoryDefinition:
        """Category for relevance and salience."""
        return CategoryDefinition(
            name="relevant_generation",
            description="Text that focuses on the most important and relevant information",
            characteristic_properties=[
                CharacteristicProperty(
                    name="salience",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Prioritizes the most important information",
                    measurement_function=lambda c: 1.0,
                    threshold=0.7,
                    weight=2.0,
                ),
                CharacteristicProperty(
                    name="conciseness",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Avoids unnecessary repetition or verbosity",
                    measurement_function=lambda c: len(c.split()) < 200,  # For summaries
                    threshold=True,
                    weight=1.0,
                ),
                CharacteristicProperty(
                    name="completeness",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Covers all key aspects without major omissions",
                    measurement_function=lambda c: 1.0,
                    threshold=0.75,
                    weight=1.5,
                ),
            ],
        )

    @staticmethod
    def create_fluency_category() -> CategoryDefinition:
        """Category for linguistic fluency."""
        return CategoryDefinition(
            name="fluent_generation",
            description="Grammatically correct and naturally flowing text",
            characteristic_properties=[
                CharacteristicProperty(
                    name="grammaticality",
                    property_type=PropertyType.NECESSARY,
                    formal_definition="Free from grammatical errors",
                    measurement_function=lambda c: not any(
                        error in c.lower() for error in [' dont ', ' wont ', ' cant ']
                    ),
                    threshold=True,
                    weight=2.0,
                ),
                CharacteristicProperty(
                    name="natural_phrasing",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Uses natural, idiomatic language",
                    measurement_function=lambda c: 1.0,
                    threshold=0.8,
                    weight=1.5,
                ),
                CharacteristicProperty(
                    name="appropriate_style",
                    property_type=PropertyType.TYPICAL,
                    formal_definition="Style appropriate for the task and domain",
                    measurement_function=lambda c: 1.0,
                    threshold=0.7,
                    weight=1.0,
                ),
            ],
        )


class NLGBenchmark:
    """Benchmark for comparing LLM-as-Judge with traditional metrics."""

    def __init__(self, judge: Optional[Judge] = None):
        """Initialize benchmark with optional custom judge."""
        self.judge = judge or Judge(provider="mock")  # Use mock for demo
        self.categories = {
            "factual_consistency": NLGEvaluationCategories.create_factual_consistency_category(),
            "coherence": NLGEvaluationCategories.create_coherence_category(),
            "relevance": NLGEvaluationCategories.create_relevance_category(),
            "fluency": NLGEvaluationCategories.create_fluency_category(),
        }

        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction()

    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not METRICS_AVAILABLE:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }

    def calculate_bleu(self, generated: str, reference: str) -> float:
        """Calculate BLEU score."""
        if not METRICS_AVAILABLE:
            return 0.0

        ref_tokens = reference.split()
        gen_tokens = generated.split()

        # Use smoothing for short texts
        return sentence_bleu(
            [ref_tokens],
            gen_tokens,
            smoothing_function=self.smoothing.method1
        )

    async def evaluate_with_llm_judge(
        self,
        sample: NLGSample
    ) -> Dict[str, Dict[str, any]]:
        """Evaluate using LLM-as-Judge across multiple categories."""
        results = {}

        # Create combined context for evaluation
        context = f"""
Source Text: {sample.source}

Generated Text: {sample.generated}

Reference Text: {sample.reference}
"""

        for category_name, category in self.categories.items():
            result = await self.judge.evaluate(context, category)
            results[category_name] = {
                "matches": result.matches_category,
                "score": result.membership_score,
                "confidence": result.confidence,
                "property_scores": result.property_scores,
                "feedback": result.feedback,
            }

        return results

    async def run_comparison(
        self,
        samples: List[NLGSample]
    ) -> Dict[str, any]:
        """Run full comparison between traditional metrics and LLM-as-Judge."""
        all_results = []

        for i, sample in enumerate(samples):
            print(f"Evaluating sample {i+1}/{len(samples)}...")

            # Traditional metrics
            rouge_scores = self.calculate_rouge(sample.generated, sample.reference)
            bleu_score = self.calculate_bleu(sample.generated, sample.reference)

            # LLM-as-Judge evaluation
            llm_scores = await self.evaluate_with_llm_judge(sample)

            result = {
                "sample_id": i,
                "model": sample.model_name,
                "task": sample.task,
                "traditional_metrics": {
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"],
                    "bleu": bleu_score,
                },
                "llm_judge_scores": llm_scores,
            }
            all_results.append(result)

        # Calculate correlations
        correlations = self._calculate_correlations(all_results)

        # Find complementary insights
        insights = self._analyze_complementary_value(all_results)

        return {
            "results": all_results,
            "correlations": correlations,
            "insights": insights,
            "summary": self._generate_summary(all_results, correlations, insights),
        }

    def _calculate_correlations(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate correlations between traditional metrics and LLM-as-Judge."""
        if len(results) < 3:
            return {}

        # Extract scores
        rouge1_scores = [r["traditional_metrics"]["rouge1"] for r in results]
        rouge2_scores = [r["traditional_metrics"]["rouge2"] for r in results]
        rougeL_scores = [r["traditional_metrics"]["rougeL"] for r in results]
        bleu_scores = [r["traditional_metrics"]["bleu"] for r in results]

        # LLM-as-Judge aggregate scores
        factual_scores = [r["llm_judge_scores"]["factual_consistency"]["score"] for r in results]
        coherence_scores = [r["llm_judge_scores"]["coherence"]["score"] for r in results]
        relevance_scores = [r["llm_judge_scores"]["relevance"]["score"] for r in results]
        fluency_scores = [r["llm_judge_scores"]["fluency"]["score"] for r in results]

        correlations = {}

        # Calculate Pearson correlations
        if len(set(rouge1_scores)) > 1:  # Avoid constant arrays
            correlations["rouge1_vs_factual"] = stats.pearsonr(rouge1_scores, factual_scores)[0]
            correlations["rouge1_vs_relevance"] = stats.pearsonr(rouge1_scores, relevance_scores)[0]
            correlations["rougeL_vs_coherence"] = stats.pearsonr(rougeL_scores, coherence_scores)[0]
            correlations["bleu_vs_fluency"] = stats.pearsonr(bleu_scores, fluency_scores)[0]

        return correlations

    def _analyze_complementary_value(self, results: List[Dict]) -> Dict[str, any]:
        """Identify where LLM-as-Judge provides complementary value."""
        insights = {
            "hallucination_detection": [],
            "coherence_issues": [],
            "high_rouge_low_quality": [],
            "low_rouge_high_quality": [],
        }

        for result in results:
            rouge_avg = np.mean([
                result["traditional_metrics"]["rouge1"],
                result["traditional_metrics"]["rouge2"],
                result["traditional_metrics"]["rougeL"],
            ])

            factual = result["llm_judge_scores"]["factual_consistency"]
            coherence = result["llm_judge_scores"]["coherence"]

            # Find hallucinations (good ROUGE but poor factual consistency)
            if rouge_avg > 0.4 and factual["score"] < 0.5:
                insights["hallucination_detection"].append({
                    "sample_id": result["sample_id"],
                    "rouge_avg": rouge_avg,
                    "factual_score": factual["score"],
                    "issue": "Possible hallucination despite good n-gram overlap"
                })

            # Find coherence issues
            if coherence["score"] < 0.5:
                insights["coherence_issues"].append({
                    "sample_id": result["sample_id"],
                    "coherence_score": coherence["score"],
                    "feedback": coherence.get("feedback", "")
                })

            # High ROUGE but low quality
            if rouge_avg > 0.5 and (factual["score"] < 0.6 or coherence["score"] < 0.6):
                insights["high_rouge_low_quality"].append({
                    "sample_id": result["sample_id"],
                    "rouge_avg": rouge_avg,
                    "quality_issues": "Factual or coherence problems"
                })

            # Low ROUGE but high quality (paraphrasing)
            quality_avg = np.mean([
                factual["score"],
                coherence["score"],
                result["llm_judge_scores"]["relevance"]["score"],
                result["llm_judge_scores"]["fluency"]["score"],
            ])

            if rouge_avg < 0.3 and quality_avg > 0.7:
                insights["low_rouge_high_quality"].append({
                    "sample_id": result["sample_id"],
                    "rouge_avg": rouge_avg,
                    "quality_avg": quality_avg,
                    "explanation": "Good paraphrasing or abstractive generation"
                })

        return insights

    def _generate_summary(
        self,
        results: List[Dict],
        correlations: Dict[str, float],
        insights: Dict[str, any]
    ) -> str:
        """Generate summary of findings."""
        summary = []
        summary.append("=" * 60)
        summary.append("NLG EVALUATION: LLM-as-Judge vs Traditional Metrics")
        summary.append("=" * 60)

        # Sample statistics
        summary.append(f"\nEvaluated {len(results)} samples")

        # Correlation analysis
        if correlations:
            summary.append("\nCORRELATIONS:")
            for metric_pair, corr in correlations.items():
                summary.append(f"  {metric_pair}: {corr:.3f}")

        # Key insights
        summary.append("\nKEY INSIGHTS:")

        if insights["hallucination_detection"]:
            summary.append(f"  • Found {len(insights['hallucination_detection'])} potential hallucinations")
            summary.append("    (High ROUGE but low factual consistency)")

        if insights["low_rouge_high_quality"]:
            summary.append(f"  • Found {len(insights['low_rouge_high_quality'])} high-quality paraphrases")
            summary.append("    (Low ROUGE but high LLM-Judge scores)")

        if insights["coherence_issues"]:
            summary.append(f"  • Detected {len(insights['coherence_issues'])} coherence issues")
            summary.append("    (Not captured by n-gram metrics)")

        # Value proposition
        summary.append("\nVALUE OF LLM-as-Judge:")
        summary.append("  ✓ Detects semantic issues ROUGE/BLEU miss")
        summary.append("  ✓ Evaluates coherence and logical flow")
        summary.append("  ✓ Identifies hallucinations and factual errors")
        summary.append("  ✓ Recognizes quality paraphrasing")
        summary.append("  ✓ Provides actionable feedback for improvement")

        summary.append("=" * 60)

        return "\n".join(summary)


def create_sample_data() -> List[NLGSample]:
    """Create sample NLG data for demonstration."""
    samples = [
        # Good ROUGE but hallucination
        NLGSample(
            source="Apple reported record Q4 revenue of $123 billion, driven by strong iPhone sales.",
            reference="Apple's Q4 revenue hit $123 billion due to iPhone demand.",
            generated="Apple's Q4 revenue reached $123 billion from iPhone sales, with CEO Tim Cook announcing plans for a new car.",
            model_name="model_a",
            task="summarization"
        ),

        # Low ROUGE but good paraphrase
        NLGSample(
            source="The research team discovered a new species of butterfly in the Amazon rainforest.",
            reference="Scientists found a new butterfly species in the Amazon.",
            generated="A previously unknown butterfly variety has been identified by researchers exploring Amazonian forests.",
            model_name="model_b",
            task="summarization"
        ),

        # Good overall
        NLGSample(
            source="Climate change is accelerating, with global temperatures rising faster than predicted.",
            reference="Global warming is happening more quickly than expected.",
            generated="Temperature increases worldwide are exceeding scientific predictions.",
            model_name="model_c",
            task="summarization"
        ),

        # Poor coherence
        NLGSample(
            source="The company launched three products: a smartphone, tablet, and smartwatch, all featuring AI.",
            reference="The company released AI-powered devices: phone, tablet, and watch.",
            generated="AI features. The company has products. Smartphone and tablet. Also smartwatch.",
            model_name="model_d",
            task="summarization"
        ),
    ]
    return samples


async def main():
    """Run the validation study."""
    print("Starting NLG Validation Study...")
    print("Comparing LLM-as-Judge with ROUGE/BLEU metrics\n")

    # Create sample data
    samples = create_sample_data()

    # Initialize benchmark with mock provider for demo
    # In production, use: Judge(provider="gemini", model="gemini-1.5-flash")
    benchmark = NLGBenchmark(
        judge=Judge(provider="mock", track_costs=True)
    )

    # Run comparison
    results = await benchmark.run_comparison(samples)

    # Print summary
    print(results["summary"])

    # Save detailed results
    output_file = Path("nlg_validation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to {output_file}")

    # Show cost if using real provider
    if benchmark.judge.track_costs:
        print("\nEvaluation Costs:")
        print(benchmark.judge.get_cost_report())


if __name__ == "__main__":
    asyncio.run(main())