"""Hierarchical dual-layer embedding implementation."""
from typing import List, Union, Dict, Any, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import BaseEmbedder


# Hyperbolic space operations (Poincaré ball model)
class HyperbolicOps:
    """Operations in hyperbolic space using Poincaré ball model."""

    @staticmethod
    def _clip_norm(x: np.ndarray, max_norm: float = 1.0 - 1e-5) -> np.ndarray:
        """Clip vector norms to avoid numerical issues at boundary."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        scale = np.minimum(1.0, max_norm / norms)
        return x * scale

    @staticmethod
    def to_poincare(x: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Project Euclidean embeddings to Poincaré ball.

        Args:
            x: Euclidean embeddings
            c: Curvature (positive, related to radius)

        Returns:
            Embeddings in Poincaré ball
        """
        # Normalize to unit ball, then scale by sqrt(c)
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        # Use tanh to map to (-1, 1), then scale
        scale = np.tanh(norm / 2)
        result = (x / norm) * scale / np.sqrt(c)
        return HyperbolicOps._clip_norm(result)

    @staticmethod
    def from_poincare(x: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Project from Poincaré ball back to Euclidean space.

        Args:
            x: Embeddings in Poincaré ball
            c: Curvature

        Returns:
            Euclidean embeddings
        """
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        # Inverse of tanh mapping
        scale = 2 * np.arctanh(norm * np.sqrt(c))
        return (x / norm) * scale

    @staticmethod
    def mobius_add(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Möbius addition in Poincaré ball.

        Args:
            x, y: Vectors in Poincaré ball
            c: Curvature

        Returns:
            x ⊕_c y in Poincaré ball
        """
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        y2 = np.sum(y * y, axis=-1, keepdims=True)
        xy = np.sum(x * y, axis=-1, keepdims=True)

        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c * c * x2 * y2

        result = numerator / (denominator + 1e-8)
        return HyperbolicOps._clip_norm(result)


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
        combination_method: Literal["concat", "weighted_sum", "learned", "hyperbolic"] = "concat",
        coarse_weight: float = 0.5,
        fine_weight: float = 0.5,
        hyperbolic_curvature: float = 1.0,
        **kwargs
    ):
        """
        Initialize hierarchical embedding model.

        Args:
            coarse_model: Name of coarse-grained sentence-transformer model
            fine_model: Name of fine-grained sentence-transformer model
            device: Device to run models on ('cpu', 'cuda', or None for auto)
            normalize: Whether to normalize final embeddings
            combination_method: How to combine embeddings ('concat', 'weighted_sum', 'learned', 'hyperbolic')
            coarse_weight: Weight for coarse embeddings (used in weighted_sum)
            fine_weight: Weight for fine embeddings (used in weighted_sum)
            hyperbolic_curvature: Curvature for hyperbolic space (positive value, default: 1.0)
            **kwargs: Additional arguments
        """
        super().__init__(f"hierarchical:{coarse_model}+{fine_model}", **kwargs)
        
        self.coarse_model_name = coarse_model
        self.fine_model_name = fine_model
        self.normalize = normalize
        self.combination_method = combination_method
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.hyperbolic_curvature = hyperbolic_curvature
        
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

        elif self.combination_method == "hyperbolic":
            # Hierarchical combination: project coarse into fine using hyperbolic geometry
            # This refines the fine embedding with coarse semantic information

            # Handle dimension mismatch by projecting coarse to fine dimension
            if coarse.shape[1] != fine.shape[1]:
                # Project coarse to match fine dimension
                if coarse.shape[1] < fine.shape[1]:
                    # Pad with zeros
                    padding = np.zeros((coarse.shape[0], fine.shape[1] - coarse.shape[1]))
                    coarse_projected = np.concatenate([coarse, padding], axis=1)
                else:
                    # Truncate (though this loses information)
                    logger.warning(
                        f"Coarse dim ({coarse.shape[1]}) > fine dim ({fine.shape[1]}). "
                        "Truncating coarse embedding. Consider using models with fine_dim >= coarse_dim."
                    )
                    coarse_projected = coarse[:, :fine.shape[1]]
            else:
                coarse_projected = coarse

            # Project both to hyperbolic space
            coarse_hyp = HyperbolicOps.to_poincare(coarse_projected, c=self.hyperbolic_curvature)
            fine_hyp = HyperbolicOps.to_poincare(fine, c=self.hyperbolic_curvature)

            # Combine using Möbius addition: coarse ⊕ fine
            # This hierarchically refines fine with coarse information
            combined_hyp = HyperbolicOps.mobius_add(
                coarse_hyp, fine_hyp, c=self.hyperbolic_curvature
            )

            # Project back to Euclidean space
            return HyperbolicOps.from_poincare(combined_hyp, c=self.hyperbolic_curvature)

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
        elif self.combination_method == "hyperbolic":
            # Hyperbolic uses Möbius addition, output dimension is fine_dim
            # (coarse is projected to fine dimension if needed)
            return self.fine_dim
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
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

        # Add hyperbolic-specific info if using hyperbolic combination
        if self.combination_method == "hyperbolic":
            info["hyperbolic_curvature"] = self.hyperbolic_curvature

        return info
