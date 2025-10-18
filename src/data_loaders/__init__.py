"""Data loaders package for benchmark, custom, and temporal datasets."""
from .base import BaseDataset, DatasetSample
from .benchmarks import STSBenchmark, MSMARCODataset, AGNewsDataset, TRECDataset, get_benchmark_dataset
from .custom import CustomDataset, load_custom_dataset
from .temporal import TemporalDataset, load_temporal_dataset

__all__ = [
    "BaseDataset",
    "DatasetSample",
    "STSBenchmark",
    "MSMARCODataset",
    "AGNewsDataset",
    "TRECDataset",
    "CustomDataset",
    "TemporalDataset",
    "get_benchmark_dataset",
    "load_custom_dataset",
    "load_temporal_dataset",
]
