"""Temporal retrieval evaluation for time-varying facts."""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, kendalltau
from loguru import logger
from datetime import datetime

from embeddings.base import BaseEmbedder
from data_loaders.temporal import TemporalDataset


class TemporalRetrievalEvaluator:
    """
    Evaluator for temporal fact retrieval tasks.

    Supports:
    - Within-group temporal ordering
    - Cross-group retrieval and discrimination
    - Temporal drift analysis
    - Multiple ranking metrics
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        evaluate_within_group: bool = True,
        evaluate_cross_group: bool = True,
        rank_correlation_method: str = "spearman",
        temporal_drift_analysis: bool = True,
        save_examples: bool = True,
        num_examples: int = 5,
    ):
        """
        Initialize temporal retrieval evaluator.

        Args:
            embedder: Embedding model to evaluate
            evaluate_within_group: Evaluate temporal ordering within groups
            evaluate_cross_group: Evaluate cross-corpus retrieval
            rank_correlation_method: 'spearman' or 'kendall'
            temporal_drift_analysis: Compute temporal drift metrics
            save_examples: Collect example retrievals for analysis
            num_examples: Number of examples to collect
        """
        self.embedder = embedder
        self.evaluate_within_group = evaluate_within_group
        self.evaluate_cross_group = evaluate_cross_group
        self.rank_correlation_method = rank_correlation_method
        self.temporal_drift_analysis = temporal_drift_analysis
        self.save_examples = save_examples
        self.num_examples = num_examples

    def evaluate(
        self,
        dataset: TemporalDataset,
        batch_size: int = 32,
        k_values: List[int] = [1, 5, 10, 20],
        temporal_timestamp_target: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate embedder on temporal retrieval task.

        Args:
            dataset: Temporal dataset
            batch_size: Batch size for encoding
            k_values: K values for recall@k and precision@k
            temporal_timestamp_target: Which model gets timestamps ("coarse" or "fine")
                                      for hierarchical models. None for single models.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating temporal retrieval on {dataset.name} ({len(dataset)} samples)")

        # Encode all facts
        logger.debug("Encoding all facts...")

        # Check if using hierarchical model with temporal timestamp targeting
        from embeddings.hierarchical import HierarchicalEmbedder
        if isinstance(self.embedder, HierarchicalEmbedder) and temporal_timestamp_target:
            # Get separate texts for coarse and fine models
            coarse_texts, fine_texts = dataset.get_texts_for_hierarchical(temporal_timestamp_target)
            logger.debug(f"Using hierarchical encoding with temporal_timestamp_target='{temporal_timestamp_target}'")
            embeddings = self.embedder.encode(
                texts=coarse_texts,  # Provide default texts
                coarse_texts=coarse_texts,
                fine_texts=fine_texts,
                batch_size=batch_size
            )
        else:
            # Standard encoding for single models or hierarchical without targeting
            texts = [s.text1 for s in dataset]
            embeddings = self.embedder.encode(texts, batch_size=batch_size)

        # Compute similarity matrix (all vs all)
        logger.debug("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        metrics = {}

        # Within-group evaluation
        if self.evaluate_within_group:
            logger.debug("Evaluating within-group temporal ordering...")
            within_metrics = self._evaluate_within_group(
                dataset, similarity_matrix, k_values
            )
            metrics.update(within_metrics)

        # Cross-group evaluation
        if self.evaluate_cross_group:
            logger.debug("Evaluating cross-group retrieval...")
            cross_metrics = self._evaluate_cross_group(
                dataset, similarity_matrix, k_values
            )
            metrics.update(cross_metrics)

        # Temporal drift analysis
        if self.temporal_drift_analysis:
            logger.debug("Analyzing temporal drift...")
            drift_metrics = self._evaluate_temporal_drift(dataset, similarity_matrix)
            metrics.update(drift_metrics)

        logger.info(f"Temporal retrieval evaluation complete")

        return metrics

    def _evaluate_within_group(
        self,
        dataset: TemporalDataset,
        similarity_matrix: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate temporal ordering within each group.

        For each fact, retrieve other facts from the same group
        and measure if they are retrieved in temporal order.

        Metrics:
        - Rank correlation: Negative values indicate good temporal ordering
          (earlier facts rank higher in similarity)
        - MRR: How quickly we find the temporally nearest fact
        """
        rank_correlations = []
        mrr_scores = []
        examples = []

        # Select random groups for examples
        import random
        example_groups = random.sample(list(dataset.groups.keys()), min(self.num_examples, len(dataset.groups))) if self.save_examples else []

        for group_id, sample_indices in dataset.groups.items():
            if len(sample_indices) < 2:
                continue  # Skip groups with only 1 fact

            # Get ground truth temporal order
            group_samples = [(idx, dataset.get_timestamp(idx)) for idx in sample_indices]
            group_samples.sort(key=lambda x: x[1])  # Sort by timestamp
            true_temporal_order = [idx for idx, _ in group_samples]

            # For each fact in the group, retrieve others
            for query_idx in sample_indices:
                # Get similarities to other facts in group (exclude self)
                other_indices = [idx for idx in sample_indices if idx != query_idx]
                similarities = similarity_matrix[query_idx, other_indices]

                # Rank by similarity (descending)
                ranked_indices_local = np.argsort(-similarities)
                retrieved_indices = [other_indices[i] for i in ranked_indices_local]

                # Compute rank correlation with temporal order
                # Create temporal ranks (excluding query)
                temporal_ranks = []
                retrieved_ranks = []
                for rank, idx in enumerate(retrieved_indices):
                    if idx in true_temporal_order:
                        temporal_ranks.append(true_temporal_order.index(idx))
                        retrieved_ranks.append(rank)

                corr = None
                if len(temporal_ranks) > 1:
                    if self.rank_correlation_method == "spearman":
                        corr, _ = spearmanr(temporal_ranks, retrieved_ranks)
                    else:  # kendall
                        corr, _ = kendalltau(temporal_ranks, retrieved_ranks)

                    if not np.isnan(corr):
                        rank_correlations.append(corr)

                # MRR: Find the rank of the temporally closest fact
                query_timestamp = dataset.get_timestamp(query_idx)
                time_diffs = [
                    abs((dataset.get_timestamp(idx) - query_timestamp).total_seconds())
                    for idx in other_indices
                ]
                nearest_temporal_idx = other_indices[np.argmin(time_diffs)]
                rank_of_nearest = retrieved_indices.index(nearest_temporal_idx) + 1
                mrr_scores.append(1.0 / rank_of_nearest)

                # Collect example if this is an example group (use first query from each)
                if self.save_examples and group_id in example_groups and query_idx == sample_indices[0]:
                    retrieved_facts = []
                    for rank, idx in enumerate(retrieved_indices[:10]):  # Top 10
                        retrieved_facts.append({
                            "text": dataset.samples[idx].text1[:100] + "...",  # Truncate
                            "timestamp": str(dataset.get_timestamp(idx)),
                            "rank": rank + 1,
                            "similarity": float(similarities[ranked_indices_local[rank]]),
                            "temporal_position": true_temporal_order.index(idx) + 1 if idx in true_temporal_order else None,
                        })

                    examples.append({
                        "query_text": dataset.samples[query_idx].text1[:100] + "...",
                        "query_timestamp": str(query_timestamp),
                        "query_group": group_id,
                        "query_temporal_position": true_temporal_order.index(query_idx) + 1,
                        "retrieved_facts": retrieved_facts,
                        "temporal_order_correlation": float(corr) if corr is not None and not np.isnan(corr) else None,
                        "nearest_fact_rank": rank_of_nearest,
                    })

        metrics = {
            "within_group_temporal_order_correlation": float(np.mean(rank_correlations)) if rank_correlations else 0.0,
            "within_group_nearest_fact_mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "within_group_num_queries": len(mrr_scores),
        }

        if self.save_examples:
            metrics["within_group_examples"] = examples

        return metrics

    def _evaluate_cross_group(
        self,
        dataset: TemporalDataset,
        similarity_matrix: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate cross-corpus retrieval.

        For each fact, retrieve from all facts and measure
        if same-group facts rank higher.
        """
        mrr_scores = []
        recall_at_k = {k: [] for k in k_values}
        precision_at_k = {k: [] for k in k_values}
        group_purity_at_k = {k: [] for k in k_values}
        examples = []

        num_queries = len(dataset)

        # Select random queries for examples
        import random
        example_indices = random.sample(range(num_queries), min(self.num_examples, num_queries)) if self.save_examples else []

        for query_idx in range(num_queries):
            query_group = dataset.get_group_id(query_idx)
            same_group_indices = [
                idx for idx in dataset.groups[query_group] if idx != query_idx
            ]

            if not same_group_indices:
                continue  # Skip if no other facts in group

            # Get similarities to all other facts (exclude self)
            similarities = similarity_matrix[query_idx].copy()
            similarities[query_idx] = -np.inf  # Exclude self

            # Rank by similarity (descending)
            ranking = np.argsort(-similarities)

            # Find position of first same-group fact
            positions_of_same_group = [
                np.where(ranking == idx)[0][0] + 1 for idx in same_group_indices
            ]
            first_same_group_rank = int(min(positions_of_same_group))
            mrr_scores.append(1.0 / first_same_group_rank)

            # Recall@k and Precision@k
            for k in k_values:
                top_k = ranking[:k]
                num_same_group_in_top_k = sum(1 for idx in top_k if idx in same_group_indices)
                total_same_group = len(same_group_indices)

                recall = num_same_group_in_top_k / total_same_group
                precision = num_same_group_in_top_k / k
                purity = num_same_group_in_top_k / k  # Same as precision

                recall_at_k[k].append(recall)
                precision_at_k[k].append(precision)
                group_purity_at_k[k].append(purity)

            # Collect example
            if self.save_examples and query_idx in example_indices:
                top_k_examples = ranking[:10]
                retrieved_facts = []
                for rank, idx in enumerate(top_k_examples):
                    retrieved_facts.append({
                        "text": dataset.samples[idx].text1[:100] + "...",
                        "group": dataset.get_group_id(idx),
                        "rank": rank + 1,
                        "similarity": float(similarities[idx]),
                        "is_same_group": idx in same_group_indices,
                    })

                examples.append({
                    "query_text": dataset.samples[query_idx].text1[:100] + "...",
                    "query_group": query_group,
                    "retrieved_facts": retrieved_facts,
                    "same_group_in_top_5": sum(1 for f in retrieved_facts[:5] if f["is_same_group"]),
                    "same_group_in_top_10": sum(1 for f in retrieved_facts if f["is_same_group"]),
                    "first_same_group_rank": first_same_group_rank,
                })

        metrics = {
            "cross_group_mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "cross_group_num_queries": len(mrr_scores),
        }

        for k in k_values:
            if recall_at_k[k]:
                metrics[f"cross_group_recall@{k}"] = float(np.mean(recall_at_k[k]))
                metrics[f"cross_group_precision@{k}"] = float(np.mean(precision_at_k[k]))
                metrics[f"cross_group_purity@{k}"] = float(np.mean(group_purity_at_k[k]))

        if self.save_examples:
            metrics["cross_group_examples"] = examples

        return metrics

    def _evaluate_temporal_drift(
        self,
        dataset: TemporalDataset,
        similarity_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """
        Analyze temporal drift: how similarity changes over time.

        Measures correlation between temporal distance and semantic similarity
        within groups.
        """
        temporal_distances = []  # in seconds
        semantic_similarities = []

        for group_id, sample_indices in dataset.groups.items():
            if len(sample_indices) < 2:
                continue

            # For all pairs in the group
            for i, idx1 in enumerate(sample_indices):
                for idx2 in sample_indices[i + 1 :]:
                    # Temporal distance
                    t1 = dataset.get_timestamp(idx1)
                    t2 = dataset.get_timestamp(idx2)
                    time_diff_seconds = abs((t2 - t1).total_seconds())

                    # Semantic similarity
                    similarity = similarity_matrix[idx1, idx2]

                    temporal_distances.append(time_diff_seconds)
                    semantic_similarities.append(similarity)

        metrics = {}

        if temporal_distances:
            # Convert to numpy arrays
            temporal_distances = np.array(temporal_distances)
            semantic_similarities = np.array(semantic_similarities)

            # Correlation between temporal distance and similarity
            # Negative correlation means similarity decreases with time (drift)
            corr, _ = spearmanr(temporal_distances, semantic_similarities)
            metrics["temporal_drift_correlation"] = float(corr) if not np.isnan(corr) else 0.0

            # Average similarity at different time ranges
            # Bin temporal distances
            max_time_diff = np.max(temporal_distances)
            if max_time_diff > 0:
                num_bins = min(5, len(temporal_distances) // 10)  # At least 10 samples per bin
                if num_bins > 1:
                    bins = np.linspace(0, max_time_diff, num_bins + 1)
                    bin_indices = np.digitize(temporal_distances, bins)

                    for bin_idx in range(1, num_bins + 1):
                        mask = bin_indices == bin_idx
                        if np.sum(mask) > 0:
                            avg_sim = np.mean(semantic_similarities[mask])
                            metrics[f"avg_similarity_time_bin_{bin_idx}"] = float(avg_sim)

            # Overall average within-group similarity
            metrics["avg_within_group_similarity"] = float(np.mean(semantic_similarities))

        return metrics
