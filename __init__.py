"""Datasets package."""
from .src.datasets.base import BaseDataset, DatasetSample
from .src.embeddings.benchmarks import (
    STSBenchmark,
    MSMARCODataset,
    AGNewsDataset,
    TRECDataset,
    get_benchmark_dataset,
)
from .src.embeddings.custom import CustomDataset, load_custom_dataset

__all__ = [
    "BaseDataset",
    "DatasetSample",
    "STSBenchmark",
    "MSMARCODataset",
    "AGNewsDataset",
    "TRECDataset",
    "get_benchmark_dataset",
    "CustomDataset",
    "load_custom_dataset",
]
