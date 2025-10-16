"""Embedding models package."""
from .base import BaseDataset
from .benchmarks import STSBenchmark, MSMARCODataset, AGNewsDataset, TRECDataset
from .custom import CustomDataset

__all__ = [
    "BaseDataset",
    "STSBenchmark",
    "MSMARCODataset",
    "AGNewsDataset",
    "TRECDataset",
    "CustomDataset",
]
