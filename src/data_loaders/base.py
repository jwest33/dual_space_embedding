"""Base dataset classes."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DatasetSample:
    """Single dataset sample."""
    text1: str
    text2: str = None  # For pair tasks
    label: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDataset(ABC):
    """Abstract base class for datasets."""
    
    def __init__(self, name: str):
        """
        Initialize dataset.
        
        Args:
            name: Name of the dataset
        """
        self.name = name
        self.samples = []
        
    @abstractmethod
    def load(self) -> None:
        """Load the dataset."""
        pass
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DatasetSample:
        """Get a sample by index."""
        return self.samples[idx]
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)
    
    def get_texts(self) -> List[str]:
        """Get all texts (text1 and text2 if exists)."""
        texts = [s.text1 for s in self.samples]
        texts.extend([s.text2 for s in self.samples if s.text2 is not None])
        return texts
    
    def get_pairs(self) -> List[Tuple[str, str]]:
        """Get all text pairs."""
        return [(s.text1, s.text2) for s in self.samples if s.text2 is not None]
    
    def get_labels(self) -> List[Any]:
        """Get all labels."""
        return [s.label for s in self.samples if s.label is not None]
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.name,
            "size": len(self),
            "has_pairs": any(s.text2 is not None for s in self.samples),
            "has_labels": any(s.label is not None for s in self.samples),
        }
