# Reproducing the Validation Results

This guide explains how to reproduce the validation results reported in this repository.

## Important Note

The validation results (82% accuracy, 24% improvement over baseline) were obtained using **synthetic test cases** designed to test specific evaluation capabilities. This approach was chosen to:
- Keep validation costs minimal (<$0.02 total)
- Test specific failure modes (hallucinations, contradictions, etc.)
- Avoid dataset licensing issues
- Enable rapid iteration

## Validation Phases Explained

### Phase 1-3: Basic Functionality (10-40 test cases)
- Used hand-crafted examples
- Tested single category and multi-dimensional evaluation
- Total cost: ~$0.006

### Phase 4: Statistical Validation (100 test cases)
- **Synthetic dataset** with programmatically generated summaries
- 50 good summaries (exact match, paraphrase, subset, reordered)
- 50 bad summaries (wrong numbers, wrong entities, hallucinations, contradictions)
- Compared against word-overlap baseline
- Total cost: ~$0.01

### Phase 5: Real-World Patterns (76 test cases)
- **Synthetic news-style** articles following CNN/DailyMail patterns
- Generated using templates with variations
- Not actual CNN/DailyMail data
- Total cost: ~$0.003

## Reproducing with Real Datasets

### Option 1: Use Our Synthetic Test Generator

```python
# Install the library
pip install llm-judge

# Run the Phase 4 validation script
python validation/phase4_hundred_cases.py

# This generates 100 synthetic test cases and evaluates them
# Expected output: ~82% accuracy, ~$0.01 cost
```

### Option 2: Evaluate on Real Benchmark Datasets

#### Step 1: Install Dependencies
```bash
pip install llm-judge datasets rouge-score
```

#### Step 2: Load a Real Dataset
```python
from datasets import load_dataset
from llm_judge import Judge, CategoryDefinition, CharacteristicProperty, PropertyType

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")

# Create evaluation category
good_summary_category = CategoryDefinition(
    name="good_summary",
    description="A factually accurate and relevant summary",
    characteristic_properties=[
        CharacteristicProperty(
            name="factual_accuracy",
            property_type=PropertyType.NECESSARY,
            formal_definition="All facts in summary match the source",
            measurement_function=lambda c: 1.0,
            threshold=True,
            weight=3.0,
        ),
        CharacteristicProperty(
            name="relevance",
            property_type=PropertyType.NECESSARY,
            formal_definition="Captures key information",
            measurement_function=lambda c: 1.0,
            threshold=0.7,
            weight=2.0,
        ),
    ],
)
```

#### Step 3: Run Evaluation
```python
import asyncio

async def evaluate_dataset():
    # Initialize Judge
    judge = Judge(
        provider="gemini",
        model="gemini-1.5-flash",
        temperature=0.1,
        track_costs=True,
    )

    results = []
    for example in dataset:
        # Create context
        context = f"Article: {example['article']}\n\nSummary: {example['highlights']}"

        # Evaluate
        result = await judge.evaluate(context, good_summary_category)
        results.append({
            'id': example['id'],
            'score': result.membership_score,
            'matches': result.matches_category,
            'confidence': result.confidence,
        })

    # Calculate accuracy (you'll need ground truth labels)
    # For CNN/DailyMail, the 'highlights' are considered good summaries

    # Get cost summary
    print(judge.get_cost_summary())

    return results

# Run evaluation
results = asyncio.run(evaluate_dataset())
```

### Option 3: Compare with ROUGE Baseline

```python
from rouge_score import rouge_scorer

def compare_with_rouge(dataset):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []
    llm_scores = []

    for example in dataset:
        # Calculate ROUGE
        rouge = scorer.score(example['article'], example['highlights'])
        rouge_scores.append(rouge['rougeL'].fmeasure)

        # Get LLM evaluation (from previous step)
        # llm_scores.append(result.membership_score)

    # Compare correlation with human judgments
    # (Requires human evaluation data)

    return rouge_scores, llm_scores
```

## Recommended Benchmark Datasets

### For Summarization
1. **CNN/DailyMail** - News summarization
   ```python
   dataset = load_dataset("cnn_dailymail", "3.0.0")
   ```

2. **XSum** - Abstractive summarization
   ```python
   dataset = load_dataset("xsum")
   ```

3. **Reddit TIFU** - Informal summaries
   ```python
   dataset = load_dataset("reddit_tifu", "long")
   ```

### For Factual Consistency
1. **QAGS** - Hallucination detection
   ```python
   # Available from: https://github.com/W4ngatang/qags
   ```

2. **SummEval** - Human-evaluated summaries
   ```python
   # Available from: https://github.com/Yale-LILY/SummEval
   ```

## Creating Your Own Test Cases

### Pattern 1: Hallucination Detection
```python
test_cases = [
    {
        "source": "Company X reported $50M revenue",
        "summary": "Company X reported $100M revenue",  # Wrong number
        "expected": False,
        "error_type": "number_hallucination"
    },
    {
        "source": "CEO John Smith announced...",
        "summary": "CEO Jane Doe announced...",  # Wrong entity
        "expected": False,
        "error_type": "entity_hallucination"
    }
]
```

### Pattern 2: Paraphrase Recognition
```python
test_cases = [
    {
        "source": "Revenue increased significantly",
        "summary": "Sales grew substantially",  # Valid paraphrase
        "expected": True,
        "error_type": None
    }
]
```

## Expected Results

When running on synthetic test cases similar to ours:
- **Binary Classification**: 70-85% accuracy expected
- **Hallucination Detection**: 90-100% accuracy on clear cases
- **Baseline Comparison**: 15-30% improvement over word-overlap
- **Cost**: ~$0.0001 per evaluation with Gemini Flash

## Cost Optimization Tips

1. **Use Caching**: Enable cache for repeated evaluations
   ```python
   judge = Judge(provider="gemini", cache_enabled=True)
   ```

2. **Batch Processing**: Process multiple examples together
   ```python
   results = await judge.batch_evaluate(contents, category)
   ```

3. **Start Small**: Test with 100 examples before scaling
   ```python
   dataset = load_dataset("cnn_dailymail", split="test[:100]")
   ```

## Validation Transparency

Our reported metrics come from:
- **100 synthetic test cases** in Phase 4
- **76 synthetic news examples** in Phase 5
- **Not** from standard benchmark datasets

To validate on real benchmarks:
1. Choose a dataset from the list above
2. Create appropriate ground truth labels
3. Run the evaluation code
4. Compare with ROUGE/BLEU baselines

The synthetic approach allowed us to:
- Test specific capabilities (hallucination, paraphrase, etc.)
- Keep costs minimal for development
- Avoid dataset licensing complexity
- Share reproducible test generation code

For production use, we recommend evaluating on your specific domain data.