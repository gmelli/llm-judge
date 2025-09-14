"""Cost tracking for LLM evaluations.

This module provides utilities to track and analyze the costs associated with
LLM-based evaluations across different providers and models.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationCost:
    """Represents the cost of a single evaluation.

    Attributes:
        provider: LLM provider name (openai, anthropic, gemini)
        model: Specific model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens used
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        total_cost: Total cost for the evaluation
        timestamp: When the evaluation occurred
        category: Category evaluated (if tracked)
        metadata: Additional metadata
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCost":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class CostTracker:
    """Tracks costs for LLM evaluations.

    Provides functionality to:
    - Track individual evaluation costs
    - Aggregate costs by provider, model, or time period
    - Save/load cost history
    - Generate cost reports
    - Recommend cost optimization strategies
    """

    # Model pricing per 1M tokens (input, output)
    MODEL_PRICING = {
        # OpenAI models
        "gpt-4-turbo-preview": (10.0, 30.0),
        "gpt-4": (30.0, 60.0),
        "gpt-3.5-turbo": (0.5, 1.5),
        "gpt-4o": (5.0, 15.0),
        "gpt-4o-mini": (0.15, 0.6),
        # Anthropic models
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        # Google Gemini models
        "gemini-2.0-flash-exp": (0.0, 0.0),  # Free experimental
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
        "gemini-1.5-pro": (3.50, 10.50),
        "gemini-1.0-pro": (0.50, 1.50),
    }

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize cost tracker.

        Args:
            history_file: Optional file to persist cost history
        """
        self.history: List[EvaluationCost] = []
        self.history_file = history_file
        self.current_session_costs: List[EvaluationCost] = []

        # Load existing history if file provided
        if self.history_file and self.history_file.exists():
            self.load_history()

    def track_evaluation(
        self,
        provider: str,
        model: str,
        usage: Dict[str, Any],
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationCost:
        """Track the cost of an evaluation.

        Args:
            provider: Provider name
            model: Model name
            usage: Usage dictionary with token counts and costs
            category: Optional category name
            metadata: Optional additional metadata

        Returns:
            EvaluationCost object with calculated costs
        """
        # Extract token counts
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Use provided costs or calculate from pricing
        if "total_cost" in usage:
            # Costs already calculated by provider
            input_cost = usage.get("input_cost", 0.0)
            output_cost = usage.get("output_cost", 0.0)
            total_cost = usage.get("total_cost", input_cost + output_cost)
        else:
            # Calculate costs from pricing table
            input_cost, output_cost, total_cost = self.calculate_cost(
                model, input_tokens, output_tokens
            )

        # Create cost record
        cost_record = EvaluationCost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata,
        )

        # Add to history
        self.history.append(cost_record)
        self.current_session_costs.append(cost_record)

        # Save if configured
        if self.history_file:
            self.save_history()

        return cost_record

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate cost based on model pricing.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        # Get pricing for model (default to GPT-3.5 pricing if unknown)
        pricing = self.MODEL_PRICING.get(model, (0.5, 1.5))
        input_price_per_million = pricing[0]
        output_price_per_million = pricing[1]

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_price_per_million
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session costs.

        Returns:
            Dictionary with session statistics
        """
        if not self.current_session_costs:
            return {
                "total_evaluations": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "by_provider": {},
                "by_model": {},
            }

        total_cost = sum(c.total_cost for c in self.current_session_costs)
        total_tokens = sum(c.total_tokens for c in self.current_session_costs)

        # Group by provider
        by_provider: Any = defaultdict(lambda: {"count": 0, "cost": 0.0, "tokens": 0})  # type: ignore[var-annotated]
        for cost in self.current_session_costs:
            by_provider[cost.provider]["count"] += 1
            by_provider[cost.provider]["cost"] += cost.total_cost
            by_provider[cost.provider]["tokens"] += cost.total_tokens

        # Group by model
        by_model: Any = defaultdict(lambda: {"count": 0, "cost": 0.0, "tokens": 0})  # type: ignore[var-annotated]
        for cost in self.current_session_costs:
            by_model[cost.model]["count"] += 1
            by_model[cost.model]["cost"] += cost.total_cost
            by_model[cost.model]["tokens"] += cost.total_tokens

        return {
            "total_evaluations": len(self.current_session_costs),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_cost": total_cost / len(self.current_session_costs),
            "by_provider": dict(by_provider),
            "by_model": dict(by_model),
        }

    def get_history_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get summary of historical costs.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary with historical statistics
        """
        # Filter history by date if specified
        filtered_history = self.history
        if start_date:
            filtered_history = [
                c for c in filtered_history if c.timestamp >= start_date
            ]
        if end_date:
            filtered_history = [c for c in filtered_history if c.timestamp <= end_date]

        if not filtered_history:
            return {
                "total_evaluations": 0,
                "total_cost": 0.0,
                "date_range": None,
            }

        total_cost = sum(c.total_cost for c in filtered_history)
        total_tokens = sum(c.total_tokens for c in filtered_history)

        # Find date range
        dates = [c.timestamp for c in filtered_history]
        date_range = {
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat(),
        }

        # Group by provider
        by_provider: Any = defaultdict(lambda: {"count": 0, "cost": 0.0})  # type: ignore[var-annotated]
        for cost in filtered_history:
            by_provider[cost.provider]["count"] += 1
            by_provider[cost.provider]["cost"] += cost.total_cost

        # Group by model
        by_model: Any = defaultdict(lambda: {"count": 0, "cost": 0.0})  # type: ignore[var-annotated]
        for cost in filtered_history:
            by_model[cost.model]["count"] += 1
            by_model[cost.model]["cost"] += cost.total_cost

        return {
            "total_evaluations": len(filtered_history),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_cost": total_cost / len(filtered_history),
            "date_range": date_range,
            "by_provider": dict(by_provider),
            "by_model": dict(by_model),
        }

    def recommend_optimization(self) -> Dict[str, Any]:
        """Recommend cost optimization strategies.

        Returns:
            Dictionary with optimization recommendations
        """
        if not self.history:
            return {"message": "No evaluation history to analyze"}

        recommendations = []

        # Analyze model usage
        model_costs: Any = defaultdict(lambda: {"count": 0, "total_cost": 0.0})  # type: ignore[var-annotated]
        for cost in self.history[-100:]:  # Last 100 evaluations
            model_costs[cost.model]["count"] += 1
            model_costs[cost.model]["total_cost"] += cost.total_cost

        # Find most expensive model
        most_expensive = max(model_costs.items(), key=lambda x: x[1]["total_cost"])
        model_name = most_expensive[0]
        model_data = most_expensive[1]

        # Recommend alternatives
        if "gpt-4" in model_name and "turbo" not in model_name:
            recommendations.append(
                {
                    "current": model_name,
                    "alternative": "gpt-4-turbo-preview",
                    "potential_savings": "~66%",
                    "reason": "GPT-4 Turbo offers similar quality at 1/3 the cost",
                }
            )
        elif "claude-3-opus" in model_name:
            recommendations.append(
                {
                    "current": model_name,
                    "alternative": "claude-3-sonnet-20240229",
                    "potential_savings": "~80%",
                    "reason": "Claude Sonnet offers good quality at much lower cost",
                }
            )
        elif "gemini-1.5-pro" in model_name:
            recommendations.append(
                {
                    "current": model_name,
                    "alternative": "gemini-1.5-flash",
                    "potential_savings": "~97%",
                    "reason": "Gemini Flash is extremely cost-effective for most evaluations",
                }
            )

        # Recommend batching
        avg_tokens = sum(c.total_tokens for c in self.history[-20:]) / min(
            20, len(self.history)
        )
        if avg_tokens < 500:
            recommendations.append(
                {
                    "strategy": "Batch evaluations",
                    "reason": f"Your average evaluation uses only {avg_tokens:.0f} tokens. "
                    "Batching multiple evaluations could improve efficiency.",
                }
            )

        return {
            "recommendations": recommendations,
            "current_most_used": model_name,
            "current_cost": model_data["total_cost"],
            "evaluation_count": model_data["count"],
        }

    def save_history(self):
        """Save cost history to file."""
        if not self.history_file:
            return

        # Convert to JSON-serializable format
        data = [cost.to_dict() for cost in self.history]

        # Save to file
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.history)} cost records to {self.history_file}")

    def load_history(self):
        """Load cost history from file."""
        if not self.history_file or not self.history_file.exists():
            return

        try:
            with open(self.history_file) as f:
                data = json.load(f)

            # Convert from dictionaries
            self.history = [EvaluationCost.from_dict(item) for item in data]

            logger.debug(
                f"Loaded {len(self.history)} cost records from {self.history_file}"
            )
        except Exception as e:
            logger.error(f"Failed to load cost history: {e}")

    def clear_session(self):
        """Clear current session costs."""
        self.current_session_costs = []

    def get_cost_report(self) -> str:
        """Generate a formatted cost report.

        Returns:
            Formatted string report of costs
        """
        session_summary = self.get_session_summary()
        history_summary = self.get_history_summary()
        recommendations = self.recommend_optimization()

        report = []
        report.append("=" * 60)
        report.append("LLM EVALUATION COST REPORT")
        report.append("=" * 60)

        # Session summary
        report.append("\nCURRENT SESSION:")
        report.append(f"  Evaluations: {session_summary['total_evaluations']}")
        report.append(f"  Total Cost: ${session_summary['total_cost']:.4f}")
        if session_summary["total_evaluations"] > 0:
            report.append(f"  Average Cost: ${session_summary['average_cost']:.4f}")

        if session_summary["by_model"]:
            report.append("\n  By Model:")
            for model, stats in session_summary["by_model"].items():
                report.append(
                    f"    {model}: ${stats['cost']:.4f} ({stats['count']} calls)"
                )

        # Historical summary
        report.append("\nALL TIME:")
        report.append(f"  Total Evaluations: {history_summary['total_evaluations']}")
        report.append(f"  Total Cost: ${history_summary['total_cost']:.4f}")
        if history_summary["total_evaluations"] > 0:
            report.append(f"  Average Cost: ${history_summary['average_cost']:.4f}")

        # Recommendations
        if recommendations.get("recommendations"):
            report.append("\nCOST OPTIMIZATION RECOMMENDATIONS:")
            for rec in recommendations["recommendations"]:
                if "alternative" in rec:
                    report.append(
                        f"  • Switch from {rec['current']} to {rec['alternative']}"
                    )
                    report.append(f"    Potential savings: {rec['potential_savings']}")
                    report.append(f"    Reason: {rec['reason']}")
                else:
                    report.append(f"  • {rec['strategy']}")
                    report.append(f"    {rec['reason']}")

        report.append("=" * 60)

        return "\n".join(report)
