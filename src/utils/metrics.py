"""Metrics utilities."""
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from loguru import logger


class MetricsTracker:
    """Track and aggregate experiment metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.results = []
        
    def add_result(
        self,
        model_name: str,
        dataset_name: str,
        task: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a result to the tracker.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            task: Task name
            metrics: Dictionary of metrics
            metadata: Additional metadata
        """
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "task": task,
            **metrics,
        }
        
        if metadata:
            result["metadata"] = metadata
            
        self.results.append(result)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all results as DataFrame.

        Only includes numeric metrics (excludes examples and other non-numeric data).

        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()

        # Filter out non-numeric metrics (like examples) for summary display
        numeric_results = []
        for result in self.results:
            numeric_result = {}
            for key, value in result.items():
                # Keep metadata as-is, filter others to numeric only
                if key == "metadata" or isinstance(value, (int, float, bool, type(None), str)):
                    numeric_result[key] = value
                # Skip lists/dicts (examples)
            numeric_results.append(numeric_result)

        return pd.DataFrame(numeric_results)
    
    def get_comparison(self, metric: str = "accuracy") -> pd.DataFrame:
        """
        Get comparison table for a specific metric across models and datasets.
        
        Args:
            metric: Metric to compare (e.g., 'accuracy', 'spearman', 'mrr')
            
        Returns:
            Pivot table DataFrame
        """
        df = self.get_summary()
        
        if df.empty:
            return pd.DataFrame()
        
        # Check if metric exists
        if metric not in df.columns:
            available = [c for c in df.columns if c not in ["model", "dataset", "task", "metadata"]]
            logger.warning(f"Metric '{metric}' not found. Available: {available}")
            return pd.DataFrame()
        
        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index=["dataset", "task"],
            columns="model",
            aggfunc="first"
        )
        
        return pivot
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {json_path}")
        
        # Save as CSV
        df = self.get_summary()
        if not df.empty:
            csv_path = output_dir / "results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        # Save comparison tables
        self._save_comparisons(output_dir)
    
    def _save_comparisons(self, output_dir: Path) -> None:
        """Save comparison tables for common metrics."""
        df = self.get_summary()
        
        if df.empty:
            return
        
        # Determine which metrics to compare based on what's available
        metric_candidates = [
            "accuracy", "f1", "precision", "recall",  # Classification
            "spearman", "pearson",  # Similarity
            "mrr", "recall@10",  # Retrieval
            "silhouette", "adjusted_rand_index",  # Clustering
        ]
        
        available_metrics = [m for m in metric_candidates if m in df.columns]
        
        for metric in available_metrics:
            comparison = self.get_comparison(metric)
            if not comparison.empty:
                comparison_path = output_dir / f"comparison_{metric}.csv"
                comparison.to_csv(comparison_path)
                logger.debug(f"Comparison table saved: {comparison_path}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Dictionary with formatted strings
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    return formatted


def aggregate_metrics(
    results: List[Dict[str, Any]],
    group_by: str = "model"
) -> pd.DataFrame:
    """
    Aggregate metrics across multiple results.
    
    Args:
        results: List of result dictionaries
        group_by: Column to group by ('model', 'dataset', or 'task')
        
    Returns:
        DataFrame with aggregated metrics
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        return pd.DataFrame()
    
    # Get numeric columns (metrics)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Group and aggregate
    aggregated = df.groupby(group_by)[numeric_cols].agg(["mean", "std"]).round(4)
    
    return aggregated
