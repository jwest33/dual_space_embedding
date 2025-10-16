"""Custom dataset loader for user-provided data."""
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger

from .base import BaseDataset, DatasetSample


class CustomDataset(BaseDataset):
    """
    Custom dataset loader supporting multiple formats.
    
    Supported formats:
    - JSON/JSONL: {"text1": "...", "text2": "...", "label": ...}
    - CSV: columns for text1, text2 (optional), label (optional)
    - TSV: same as CSV
    """
    
    def __init__(
        self,
        file_path: str,
        text1_column: str = "text1",
        text2_column: Optional[str] = "text2",
        label_column: Optional[str] = "label",
        format: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize custom dataset.
        
        Args:
            file_path: Path to dataset file
            text1_column: Name of text1 column
            text2_column: Name of text2 column (None if not present)
            label_column: Name of label column (None if not present)
            format: File format ('json', 'jsonl', 'csv', 'tsv', or None for auto-detect)
            **kwargs: Additional arguments for pandas readers
        """
        self.file_path = Path(file_path)
        super().__init__(f"custom-{self.file_path.stem}")
        
        self.text1_column = text1_column
        self.text2_column = text2_column
        self.label_column = label_column
        self.format = format or self._detect_format()
        self.kwargs = kwargs
        
        self.load()
        
    def _detect_format(self) -> str:
        """Auto-detect file format from extension."""
        suffix = self.file_path.suffix.lower()
        
        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".tsv": "tsv",
            ".txt": "csv",  # Assume CSV for .txt
        }
        
        return format_map.get(suffix, "csv")
    
    def load(self) -> None:
        """Load custom dataset from file."""
        logger.info(f"Loading custom dataset from {self.file_path} (format: {self.format})")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        if self.format == "json":
            self._load_json()
        elif self.format == "jsonl":
            self._load_jsonl()
        elif self.format in ["csv", "tsv"]:
            self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        logger.info(f"Loaded {len(self.samples)} samples from custom dataset")
    
    def _load_json(self) -> None:
        """Load from JSON file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        for item in data:
            self._add_sample(item)
    
    def _load_jsonl(self) -> None:
        """Load from JSONL file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self._add_sample(item)
    
    def _load_csv(self) -> None:
        """Load from CSV/TSV file."""
        separator = "\t" if self.format == "tsv" else ","
        
        df = pd.read_csv(
            self.file_path,
            sep=separator,
            encoding="utf-8",
            **self.kwargs
        )
        
        for _, row in df.iterrows():
            self._add_sample(row.to_dict())
    
    def _add_sample(self, item: Dict[str, Any]) -> None:
        """Add a sample from dictionary."""
        text1 = item.get(self.text1_column)
        if text1 is None:
            logger.warning(f"Missing {self.text1_column} in item, skipping")
            return
        
        text2 = item.get(self.text2_column) if self.text2_column else None
        label = item.get(self.label_column) if self.label_column else None
        
        # Metadata is everything else
        metadata = {
            k: v for k, v in item.items()
            if k not in [self.text1_column, self.text2_column, self.label_column]
        }
        
        sample = DatasetSample(
            text1=str(text1),
            text2=str(text2) if text2 is not None else None,
            label=label,
            metadata=metadata
        )
        self.samples.append(sample)


def load_custom_dataset(file_path: str, **kwargs) -> CustomDataset:
    """
    Convenience function to load custom dataset.
    
    Args:
        file_path: Path to dataset file
        **kwargs: Arguments passed to CustomDataset
        
    Returns:
        CustomDataset instance
    """
    return CustomDataset(file_path, **kwargs)
