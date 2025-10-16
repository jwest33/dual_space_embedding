"""Single embedding model implementation."""
from typing import List, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import BaseEmbedder


class SingleEmbedder(BaseEmbedder):
    """Single sentence-transformer model embedder."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize single embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments passed to SentenceTransformer
        """
        super().__init__(model_name, **kwargs)
        self.normalize = normalize
        
        logger.info(f"Loading single model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded. Embedding dimension: {self.get_embedding_dim()}")
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to model.encode()
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            **kwargs
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "type": "single",
            "model_name": self.model_name,
            "embedding_dim": self.get_embedding_dim(),
            "normalize": self.normalize,
            "max_seq_length": self.model.max_seq_length,
        }
