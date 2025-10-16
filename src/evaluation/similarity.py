"""Semantic similarity evaluation."""
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from loguru import logger

from embeddings.base import BaseEmbedder
from data_loaders.base import BaseDataset


class SimilarityEvaluator:
    """Evaluator for semantic textual similarity tasks."""
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize similarity evaluator.
        
        Args:
            embedder: Embedding model to evaluate
        """
        self.embedder = embedder
        
    def evaluate(
        self,
        dataset: BaseDataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate embedder on similarity task.
        
        Args:
            dataset: Dataset with text pairs and similarity scores
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating similarity on {dataset.name} ({len(dataset)} samples)")
        
        # Get pairs and labels
        texts1 = [s.text1 for s in dataset]
        texts2 = [s.text2 for s in dataset]
        labels = [s.label for s in dataset]

        if None in labels:
            raise ValueError("Dataset must have labels for similarity evaluation")
        
        # Encode texts
        logger.debug("Encoding text1...")
        embeddings1 = self.embedder.encode(texts1, batch_size=batch_size)

        logger.debug("Encoding text2...")
        embeddings2 = self.embedder.encode(texts2, batch_size=batch_size)

        # Compute cosine similarities
        similarities = self._compute_similarities(embeddings1, embeddings2)
        
        # Compute metrics
        metrics = self._compute_metrics(similarities, labels)
        
        logger.info(f"Similarity evaluation complete. Spearman: {metrics['spearman']:.4f}")
        
        return metrics
    
    def _compute_similarities(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarities.

        Args:
            embeddings1: First set of embeddings (shape: n_samples x embedding_dim)
            embeddings2: Second set of embeddings (shape: n_samples x embedding_dim)

        Returns:
            Array of similarity scores (shape: n_samples)
        """
        # Vectorized computation: element-wise dot product along dimension axis
        # For normalized embeddings, this is equivalent to cosine similarity
        # For non-normalized: cosine = dot(a,b) / (norm(a) * norm(b))

        # Compute dot product for each pair
        similarities = np.sum(embeddings1 * embeddings2, axis=1)

        # Normalize by magnitudes (in case embeddings aren't already normalized)
        norms1 = np.linalg.norm(embeddings1, axis=1)
        norms2 = np.linalg.norm(embeddings2, axis=1)

        # Avoid division by zero
        norms1 = np.where(norms1 == 0, 1, norms1)
        norms2 = np.where(norms2 == 0, 1, norms2)

        similarities = similarities / (norms1 * norms2)

        return similarities
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: List[float]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted similarity scores
            labels: True similarity scores

        Returns:
            Dictionary with metrics
        """
        labels = np.array(labels)

        # Check for constant predictions (zero variance)
        pred_std = np.std(predictions)
        if pred_std < 1e-8:
            logger.warning(
                f"Predictions are constant (all ~{predictions[0]:.6f})! "
                f"This suggests embeddings may be collapsing. "
                f"Correlation will be set to 0.0."
            )
            logger.debug(
                f"Prediction stats - min: {np.min(predictions):.6f}, "
                f"max: {np.max(predictions):.6f}, std: {pred_std:.6e}"
            )
            spearman_corr = 0.0
            spearman_p = 1.0
            pearson_corr = 0.0
            pearson_p = 1.0
        else:
            # Spearman correlation (rank-based)
            spearman_corr, spearman_p = spearmanr(predictions, labels)

            # Pearson correlation (linear)
            pearson_corr, pearson_p = pearsonr(predictions, labels)

        # Mean absolute error
        mae = np.mean(np.abs(predictions - labels))

        # Root mean squared error
        rmse = np.sqrt(np.mean((predictions - labels) ** 2))

        return {
            "spearman": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "pearson": float(pearson_corr),
            "pearson_p": float(pearson_p),
            "mae": float(mae),
            "rmse": float(rmse),
            "num_samples": len(labels),
            "pred_std": float(pred_std),
        }
