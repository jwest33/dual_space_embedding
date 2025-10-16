"""Hierarchical dual-layer embedding implementation."""
from typing import List, Union, Dict, Any, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import BaseEmbedder


class HierarchicalEmbedder(BaseEmbedder):
    """
    Hierarchical dual-layer embedder.
    
    Uses a coarse-grained model for initial broad semantic understanding,
    then refines with a fine-grained model for detailed semantic nuances.
    """
    
    def __init__(
        self,
        coarse_model: str = "all-MiniLM-L6-v2",
        fine_model: str = "all-mpnet-base-v2",
        device: str = None,
        normalize: bool = True,
        combination_method: Literal["concat", "weighted_sum", "learned"] = "concat",
        coarse_weight: float = 0.5,
        fine_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize hierarchical embedding model.
        
        Args:
            coarse_model: Name of coarse-grained sentence-transformer model
            fine_model: Name of fine-grained sentence-transformer model
            device: Device to run models on ('cpu', 'cuda', or None for auto)
            normalize: Whether to normalize final embeddings
            combination_method: How to combine embeddings ('concat', 'weighted_sum', 'learned')
            coarse_weight: Weight for coarse embeddings (used in weighted_sum)
            fine_weight: Weight for fine embeddings (used in weighted_sum)
            **kwargs: Additional arguments
        """
        super().__init__(f"hierarchical:{coarse_model}+{fine_model}", **kwargs)
        
        self.coarse_model_name = coarse_model
        self.fine_model_name = fine_model
        self.normalize = normalize
        self.combination_method = combination_method
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        
        # Load models
        logger.info(f"Loading coarse model: {coarse_model}")
        self.coarse_model = SentenceTransformer(coarse_model, device=device)
        
        logger.info(f"Loading fine model: {fine_model}")
        self.fine_model = SentenceTransformer(fine_model, device=device)
        
        self.coarse_dim = self.coarse_model.get_sentence_embedding_dimension()
        self.fine_dim = self.fine_model.get_sentence_embedding_dimension()
        
        logger.info(
            f"Hierarchical model loaded. Coarse dim: {self.coarse_dim}, "
            f"Fine dim: {self.fine_dim}, Combined dim: {self.get_embedding_dim()}"
        )
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) using hierarchical approach.
        
        First encodes with coarse model for broad semantics,
        then with fine model for detailed semantics,
        finally combines according to combination_method.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of combined embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode with coarse model (broad semantic understanding)
        logger.debug(f"Encoding {len(texts)} texts with coarse model")
        coarse_embeddings = self.coarse_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,  # We'll normalize at the end
            convert_to_numpy=True,
            **kwargs
        )
        
        # Encode with fine model (detailed semantic understanding)
        logger.debug(f"Encoding {len(texts)} texts with fine model")
        fine_embeddings = self.fine_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
            convert_to_numpy=True,
            **kwargs
        )
        
        # Combine embeddings
        combined = self._combine_embeddings(coarse_embeddings, fine_embeddings)

        # Normalize if requested (only for non-concat methods, as concat normalizes components)
        if self.normalize and self.combination_method != "concat":
            combined = self._normalize(combined)

        return combined
    
    def _combine_embeddings(
        self,
        coarse: np.ndarray,
        fine: np.ndarray
    ) -> np.ndarray:
        """
        Combine coarse and fine embeddings.

        Args:
            coarse: Coarse-grained embeddings
            fine: Fine-grained embeddings

        Returns:
            Combined embeddings
        """
        if self.combination_method == "concat":
            # Normalize each component before concatenation to preserve variance
            # This prevents the embeddings from collapsing to similar vectors
            coarse_normalized = self._normalize(coarse)
            fine_normalized = self._normalize(fine)
            return np.concatenate([coarse_normalized, fine_normalized], axis=1)
        
        elif self.combination_method == "weighted_sum":
            # Weighted sum (requires same dimensions or projection)
            if coarse.shape[1] != fine.shape[1]:
                raise ValueError(
                    f"weighted_sum requires same dimensions. "
                    f"Got coarse: {coarse.shape[1]}, fine: {fine.shape[1]}"
                )
            return self.coarse_weight * coarse + self.fine_weight * fine
        
        elif self.combination_method == "learned":
            # For now, learned is the same as concat
            # In production, this would use a learned projection layer
            logger.warning(
                "Learned combination not implemented yet, falling back to concat"
            )
            return np.concatenate([coarse, fine], axis=1)
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def get_embedding_dim(self) -> int:
        """Get final embedding dimensionality."""
        if self.combination_method == "concat":
            return self.coarse_dim + self.fine_dim
        elif self.combination_method == "weighted_sum":
            return self.coarse_dim  # Assumes same dimensions
        elif self.combination_method == "learned":
            return self.coarse_dim + self.fine_dim  # For now
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "hierarchical",
            "coarse_model": self.coarse_model_name,
            "fine_model": self.fine_model_name,
            "coarse_dim": self.coarse_dim,
            "fine_dim": self.fine_dim,
            "combined_dim": self.get_embedding_dim(),
            "normalize": self.normalize,
            "combination_method": self.combination_method,
            "coarse_weight": self.coarse_weight,
            "fine_weight": self.fine_weight,
        }
