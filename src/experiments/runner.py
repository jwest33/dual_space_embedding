"""Experiment runner for orchestrating evaluations."""
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime
from loguru import logger
import mlflow

from embeddings import SingleEmbedder, HierarchicalEmbedder
from data_loaders import get_benchmark_dataset, load_custom_dataset, load_temporal_dataset
from evaluation import (
    SimilarityEvaluator,
    RetrievalEvaluator,
    ClassificationEvaluator,
    ClusteringEvaluator,
    TemporalRetrievalEvaluator,
)
from evaluation.temporal_examples import save_examples, generate_examples_summary
from utils import ExperimentConfig, ModelConfig, DatasetConfig, MetricsTracker


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.metrics_tracker = MetricsTracker()

        # Setup output directory (create subdirectory for experiment name)
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup MLflow if enabled
        if config.mlflow_tracking:
            mlflow.set_tracking_uri(config.mlflow_uri)
            mlflow.set_experiment(config.name)
        
        logger.info(f"Experiment runner initialized: {config.name}")
    
    def run(self) -> MetricsTracker:
        """
        Run the full experiment.
        
        Returns:
            MetricsTracker with all results
        """
        logger.info("=" * 80)
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run experiments for each model
        for model_config in self.config.models:
            self._run_model_experiments(model_config)
        
        # Save results
        self._save_results()
        
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Experiment complete! Total time: {elapsed:.2f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return self.metrics_tracker
    
    def _run_model_experiments(self, model_config: ModelConfig) -> None:
        """
        Run experiments for a single model.
        
        Args:
            model_config: Model configuration
        """
        logger.info("-" * 80)
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info("-" * 80)
        
        # Load embedder
        embedder = self._load_embedder(model_config)
        
        # Start MLflow run
        if self.config.mlflow_tracking:
            mlflow.start_run(run_name=model_config.name)
            mlflow.log_params(embedder.get_model_info())
        
        try:
            # Run on each dataset
            for dataset_config in self.config.datasets:
                self._run_dataset_experiments(embedder, model_config, dataset_config)
        finally:
            if self.config.mlflow_tracking:
                mlflow.end_run()
    
    def _run_dataset_experiments(
        self,
        embedder,
        model_config: ModelConfig,
        dataset_config: DatasetConfig
    ) -> None:
        """
        Run experiments for a model on a single dataset.
        
        Args:
            embedder: Embedding model
            model_config: Model configuration
            dataset_config: Dataset configuration
        """
        logger.info(f"Dataset: {dataset_config.name}")
        
        # Load dataset(s)
        datasets = self._load_datasets(dataset_config)
        
        # Run each task
        for task in self.config.tasks:
            if task not in datasets:
                logger.warning(f"Skipping task '{task}' - no suitable dataset loaded")
                continue
            
            logger.info(f"  Task: {task}")
            
            try:
                metrics = self._run_task(
                    embedder,
                    task,
                    datasets[task]
                )
                
                # Track metrics
                self.metrics_tracker.add_result(
                    model_name=model_config.name,
                    dataset_name=dataset_config.name,
                    task=task,
                    metrics=metrics,
                    metadata={
                        "model_info": embedder.get_model_info(),
                        "dataset_info": datasets[task]["dataset"].get_info() if "dataset" in datasets[task] else {}
                    }
                )
                
                # Save temporal examples if available
                if task == "temporal" and self.config.temporal_save_examples:
                    if "within_group_examples" in metrics or "cross_group_examples" in metrics:
                        save_examples(
                            model_name=model_config.name,
                            dataset_name=dataset_config.name,
                            metrics=metrics,
                            output_dir=self.output_dir,
                        )
                        logger.info(f"    Saved temporal examples to {self.output_dir}/examples/")

                # Log to MLflow
                if self.config.mlflow_tracking:
                    for metric_name, metric_value in metrics.items():
                        # Skip non-numeric values (like examples)
                        if isinstance(metric_value, (int, float)):
                            # Sanitize metric name for MLflow (replace @ with _at_)
                            sanitized_name = metric_name.replace("@", "_at_")
                            mlflow.log_metric(
                                f"{dataset_config.name}_{task}_{sanitized_name}",
                                metric_value
                            )

                # Print key metrics
                self._print_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error in task '{task}': {e}", exc_info=True)
    
    def _load_embedder(self, model_config: ModelConfig):
        """Load embedding model from configuration."""
        if model_config.type == "single":
            return SingleEmbedder(
                model_name=model_config.model_name,
                device=model_config.device,
                normalize=model_config.normalize,
            )
        elif model_config.type == "hierarchical":
            return HierarchicalEmbedder(
                coarse_model=model_config.coarse_model,
                fine_model=model_config.fine_model,
                device=model_config.device,
                normalize=model_config.normalize,
                combination_method=model_config.combination_method,
                coarse_weight=model_config.coarse_weight,
                fine_weight=model_config.fine_weight,
                hyperbolic_curvature=model_config.hyperbolic_curvature,
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")
    
    def _load_datasets(self, dataset_config: DatasetConfig) -> Dict[str, Any]:
        """
        Load dataset(s) for different tasks.

        Returns:
            Dictionary mapping task names to dataset configurations
        """
        datasets = {}

        if dataset_config.type == "benchmark":
            # Load benchmark dataset
            dataset = get_benchmark_dataset(
                dataset_config.name,
                split=dataset_config.split,
                num_samples=dataset_config.num_samples
            )

            # Determine which tasks this dataset supports
            info = dataset.get_info()

            if info["has_pairs"] and info["has_labels"]:
                # Supports similarity
                datasets["similarity"] = {"dataset": dataset}
                datasets["retrieval"] = {"dataset": dataset}

            if not info["has_pairs"] and info["has_labels"]:
                # Supports classification and clustering
                datasets["classification"] = {
                    "train": dataset,
                    "test": dataset  # Will split or use separate split
                }
                datasets["clustering"] = {"dataset": dataset}

        elif dataset_config.type == "custom":
            # Load custom dataset
            dataset = load_custom_dataset(
                file_path=dataset_config.file_path,
                text1_column=dataset_config.text1_column,
                text2_column=dataset_config.text2_column,
                label_column=dataset_config.label_column,
                format=dataset_config.format,
            )

            # Determine supported tasks
            info = dataset.get_info()

            if info["has_pairs"] and info["has_labels"]:
                datasets["similarity"] = {"dataset": dataset}
                datasets["retrieval"] = {"dataset": dataset}

            if not info["has_pairs"] and info["has_labels"]:
                datasets["classification"] = {
                    "train": dataset,
                    "test": dataset
                }
                datasets["clustering"] = {"dataset": dataset}

            if not info["has_labels"]:
                # Unsupervised clustering only
                datasets["clustering"] = {"dataset": dataset}

        elif dataset_config.type == "temporal":
            # Load temporal dataset
            dataset = load_temporal_dataset(
                file_path=dataset_config.file_path,
                text_column=dataset_config.text_column,
                timestamp_column=dataset_config.timestamp_column or "metadata.timestamp",
                group_id_column=dataset_config.group_id_column or "metadata.group_id",
                append_timestamp_to_text=dataset_config.append_timestamp_to_text or self.config.temporal_append_timestamp,
                timestamp_format=dataset_config.timestamp_format or self.config.temporal_timestamp_format,
                num_samples=dataset_config.num_samples
            )

            # Temporal datasets support temporal task
            datasets["temporal"] = {"dataset": dataset}

        return datasets
    
    def _run_task(
        self,
        embedder,
        task: str,
        dataset_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Run a specific task.

        Args:
            embedder: Embedding model
            task: Task name
            dataset_config: Dataset configuration for this task

        Returns:
            Dictionary of metrics
        """
        if task == "similarity":
            evaluator = SimilarityEvaluator(embedder)
            return evaluator.evaluate(
                dataset_config["dataset"],
                batch_size=self.config.batch_size
            )

        elif task == "retrieval":
            evaluator = RetrievalEvaluator(embedder)
            return evaluator.evaluate(
                dataset_config["dataset"],
                batch_size=self.config.batch_size,
                k_values=self.config.retrieval_k_values
            )

        elif task == "classification":
            evaluator = ClassificationEvaluator(
                embedder,
                classifier=self.config.classification_classifier
            )
            return evaluator.evaluate(
                dataset_config["train"],
                dataset_config["test"],
                batch_size=self.config.batch_size
            )

        elif task == "clustering":
            evaluator = ClusteringEvaluator(
                embedder,
                algorithm=self.config.clustering_algorithm,
                n_clusters=self.config.clustering_n_clusters
            )
            return evaluator.evaluate(
                dataset_config["dataset"],
                batch_size=self.config.batch_size
            )

        elif task == "temporal":
            evaluator = TemporalRetrievalEvaluator(
                embedder,
                evaluate_within_group=self.config.temporal_evaluate_within_group,
                evaluate_cross_group=self.config.temporal_evaluate_cross_group,
                rank_correlation_method=self.config.temporal_rank_correlation_method,
                temporal_drift_analysis=self.config.temporal_drift_analysis,
                save_examples=self.config.temporal_save_examples,
                num_examples=self.config.temporal_num_examples,
            )
            return evaluator.evaluate(
                dataset_config["dataset"],
                batch_size=self.config.batch_size,
                k_values=self.config.retrieval_k_values
            )

        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print key metrics."""
        # Select most important metrics based on what's available
        key_metrics = {}

        priority_metrics = [
            "accuracy", "f1", "spearman", "pearson",
            "mrr", "recall@10", "silhouette",
            "within_group_temporal_order_correlation", "within_group_nearest_fact_mrr",
            "cross_group_mrr", "cross_group_purity@5",
            "temporal_drift_correlation"
        ]

        for metric in priority_metrics:
            if metric in metrics:
                key_metrics[metric] = metrics[metric]

        # Print
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in key_metrics.items()])
        logger.info(f"    Metrics: {metrics_str}")
    
    def _save_results(self) -> None:
        """Save all results."""
        self.metrics_tracker.save_results(self.output_dir)
        
        # Generate summary report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate experiment report."""
        report_path = self.output_dir / "report.txt"
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Experiment Report: {self.config.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Description: {self.config.description}\n\n")
            
            f.write(f"Models evaluated: {len(self.config.models)}\n")
            for model in self.config.models:
                f.write(f"  - {model.name} ({model.type})\n")
            f.write("\n")
            
            f.write(f"Datasets used: {len(self.config.datasets)}\n")
            for dataset in self.config.datasets:
                f.write(f"  - {dataset.name} ({dataset.type})\n")
            f.write("\n")
            
            f.write(f"Tasks evaluated: {', '.join(self.config.tasks)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Results Summary\n")
            f.write("=" * 80 + "\n\n")
            
            # Write summary table
            summary = self.metrics_tracker.get_summary()
            if not summary.empty:
                f.write(summary.to_string(index=False))
            else:
                f.write("No results available.\n")

            # Add examples summary if temporal task with examples
            examples_dir = self.output_dir / "examples"
            if examples_dir.exists() and any(examples_dir.glob("*.json")):
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write("Temporal Retrieval Examples\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Example retrievals have been saved to: {examples_dir.relative_to(self.output_dir)}/\n")
                f.write("See *_examples.txt files for human-readable examples\n")
                f.write("See *_examples.json files for machine-readable data\n")

                # Generate and save examples summary
                summary_md = generate_examples_summary(self.output_dir)
                summary_path = examples_dir / "README.md"
                with open(summary_path, "w") as sf:
                    sf.write(summary_md)
                f.write(f"\nSee {examples_dir.relative_to(self.output_dir)}/README.md for full examples summary\n")

        logger.info(f"Report saved to {report_path}")
