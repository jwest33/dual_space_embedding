"""Base classes for embedding models."""
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name or path of the model
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional encoding arguments
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Integer dimension of embeddings
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model metadata
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
