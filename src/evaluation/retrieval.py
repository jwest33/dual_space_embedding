"""Information retrieval evaluation."""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from embeddings.base import BaseEmbedder
from datasets.base import BaseDataset


class RetrievalEvaluator:
    """Evaluator for information retrieval tasks."""
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize retrieval evaluator.
        
        Args:
            embedder: Embedding model to evaluate
        """
        self.embedder = embedder
        
    def evaluate(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate embedder on retrieval task.
        
        For each query, ranks all passages and computes ranking metrics.
        
        Args:
            dataset: Dataset with query-passage pairs
            batch_size: Batch size for encoding
            k_values: K values for recall@k and precision@k
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating retrieval on {dataset.name} ({len(dataset)} samples)")
        
        # Separate queries and passages
        queries = [s.text1 for s in dataset]
        passages = [s.text2 for s in dataset]
        labels = [s.label for s in dataset]
        
        # Encode
        logger.debug("Encoding queries...")
        query_embeddings = self.embedder.encode(queries, batch_size=batch_size)
        
        logger.debug("Encoding passages...")
        passage_embeddings = self.embedder.encode(passages, batch_size=batch_size)
        
        # Compute similarity matrix (queries x passages)
        logger.debug("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(query_embeddings, passage_embeddings)
        
        # For each query, the relevant passage is at the same index
        # (assuming 1:1 query-passage pairs)
        relevant_indices = np.arange(len(queries))
        
        # Compute metrics
        metrics = self._compute_metrics(
            similarity_matrix,
            relevant_indices,
            k_values
        )
        
        logger.info(f"Retrieval evaluation complete. MRR: {metrics['mrr']:.4f}")
        
        return metrics
    
    def _compute_metrics(
        self,
        similarity_matrix: np.ndarray,
        relevant_indices: np.ndarray,
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            similarity_matrix: Similarity scores (queries x passages)
            relevant_indices: Index of relevant passage for each query
            k_values: K values for top-k metrics
            
        Returns:
            Dictionary with metrics
        """
        num_queries = similarity_matrix.shape[0]
        
        # Get ranking for each query (descending order)
        rankings = np.argsort(-similarity_matrix, axis=1)
        
        # Find position of relevant passage for each query
        reciprocal_ranks = []
        recalls_at_k = {k: [] for k in k_values}
        precisions_at_k = {k: [] for k in k_values}
        
        for query_idx in range(num_queries):
            relevant_idx = relevant_indices[query_idx]
            ranking = rankings[query_idx]
            
            # Find position (1-indexed) of relevant passage
            position = np.where(ranking == relevant_idx)[0][0] + 1
            
            # Reciprocal rank
            reciprocal_ranks.append(1.0 / position)
            
            # Recall@k and Precision@k
            for k in k_values:
                top_k = ranking[:k]
                if relevant_idx in top_k:
                    recalls_at_k[k].append(1.0)
                    precisions_at_k[k].append(1.0 / k)
                else:
                    recalls_at_k[k].append(0.0)
                    precisions_at_k[k].append(0.0)
        
        # Mean Reciprocal Rank
        mrr = float(np.mean(reciprocal_ranks))
        
        # Recall@k and Precision@k
        metrics = {
            "mrr": mrr,
            "num_queries": num_queries,
        }
        
        for k in k_values:
            metrics[f"recall@{k}"] = float(np.mean(recalls_at_k[k]))
            metrics[f"precision@{k}"] = float(np.mean(precisions_at_k[k]))
        
        return metrics
