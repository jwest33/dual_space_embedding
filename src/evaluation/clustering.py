"""Clustering evaluation."""
from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from loguru import logger

from embeddings.base import BaseEmbedder
from data_loaders.base import BaseDataset


class ClusteringEvaluator:
    """Evaluator for clustering tasks."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        algorithm: str = "kmeans",
        n_clusters: int = None
    ):
        """
        Initialize clustering evaluator.
        
        Args:
            embedder: Embedding model to evaluate
            algorithm: Clustering algorithm ('kmeans', 'agglomerative')
            n_clusters: Number of clusters (None to infer from labels)
        """
        self.embedder = embedder
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        
    def evaluate(
        self,
        dataset: BaseDataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate embedder on clustering task.
        
        Args:
            dataset: Dataset with texts and optional labels
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating clustering on {dataset.name} ({len(dataset)} samples)")
        
        # Get texts and labels
        texts = [s.text1 for s in dataset]
        true_labels = [s.label for s in dataset if s.label is not None]
        
        has_labels = len(true_labels) > 0
        
        # Infer number of clusters from labels if not specified
        if self.n_clusters is None:
            if has_labels:
                self.n_clusters = len(set(true_labels))
                logger.debug(f"Inferred {self.n_clusters} clusters from labels")
            else:
                # Default heuristic: sqrt(n)
                self.n_clusters = int(np.sqrt(len(texts)))
                logger.debug(f"Using heuristic: {self.n_clusters} clusters")
        
        # Encode texts
        logger.debug("Encoding texts...")
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        
        # Cluster
        logger.debug(f"Clustering with {self.algorithm}...")
        predicted_labels = self._cluster(embeddings)
        
        # Compute metrics
        metrics = self._compute_metrics(
            embeddings,
            predicted_labels,
            true_labels if has_labels else None
        )
        
        logger.info(
            f"Clustering evaluation complete. "
            f"Silhouette: {metrics['silhouette']:.4f}"
        )
        
        return metrics
    
    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering.
        
        Args:
            embeddings: Text embeddings
            
        Returns:
            Cluster labels
        """
        if self.algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.algorithm == "agglomerative":
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
        
        labels = clusterer.fit_predict(embeddings)
        return labels
    
    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: List[Any] = None
    ) -> Dict[str, float]:
        """
        Compute clustering metrics.
        
        Args:
            embeddings: Text embeddings
            predicted_labels: Predicted cluster labels
            true_labels: True labels (if available)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "n_clusters": self.n_clusters,
            "num_samples": len(embeddings),
        }
        
        # Intrinsic metrics (no ground truth needed)
        try:
            # Silhouette score (-1 to 1, higher is better)
            silhouette = silhouette_score(embeddings, predicted_labels)
            metrics["silhouette"] = float(silhouette)
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            metrics["silhouette"] = 0.0
        
        try:
            # Davies-Bouldin index (lower is better)
            davies_bouldin = davies_bouldin_score(embeddings, predicted_labels)
            metrics["davies_bouldin"] = float(davies_bouldin)
        except Exception as e:
            logger.warning(f"Could not compute Davies-Bouldin score: {e}")
            metrics["davies_bouldin"] = 0.0
        
        try:
            # Calinski-Harabasz index (higher is better)
            calinski_harabasz = calinski_harabasz_score(embeddings, predicted_labels)
            metrics["calinski_harabasz"] = float(calinski_harabasz)
        except Exception as e:
            logger.warning(f"Could not compute Calinski-Harabasz score: {e}")
            metrics["calinski_harabasz"] = 0.0
        
        # Extrinsic metrics (require ground truth)
        if true_labels is not None:
            try:
                # Adjusted Rand Index (0 to 1, higher is better)
                ari = adjusted_rand_score(true_labels, predicted_labels)
                metrics["adjusted_rand_index"] = float(ari)
                
                # Normalized Mutual Information (0 to 1, higher is better)
                nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                metrics["normalized_mutual_info"] = float(nmi)
            except Exception as e:
                logger.warning(f"Could not compute extrinsic metrics: {e}")
        
        return metrics
