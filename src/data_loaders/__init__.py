"""Data loaders package for benchmark and custom datasets."""
from .base import BaseDataset
from .benchmarks import STSBenchmark, MSMARCODataset, AGNewsDataset, TRECDataset, get_benchmark_dataset
from .custom import CustomDataset, load_custom_dataset

__all__ = [
    "BaseDataset",
    "STSBenchmark",
    "MSMARCODataset",
    "AGNewsDataset",
    "TRECDataset",
    "CustomDataset",
    "get_benchmark_dataset",
    "load_custom_dataset",
]
