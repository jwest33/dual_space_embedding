"""Temporal dataset loader for time-varying facts."""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger

from .base import BaseDataset, DatasetSample


class TemporalDataset(BaseDataset):
    """
    Dataset loader for temporal facts with timestamp and group information.

    Expected JSONL format:
    {
        "instruction": "fact text",
        "response": "",
        "metadata": {
            "timestamp": "2025-01-01T00:00:00",
            "group_id": "group_001",
            "variation_id": 1,
            ...
        }
    }
    """

    def __init__(
        self,
        file_path: str,
        text_column: str = "instruction",
        timestamp_column: str = "metadata.timestamp",
        group_id_column: str = "metadata.group_id",
        append_timestamp_to_text: bool = False,
        timestamp_format: str = "iso",
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize temporal dataset.

        Args:
            file_path: Path to JSONL file
            text_column: Name of column containing fact text
            timestamp_column: Path to timestamp field (supports nested, e.g., "metadata.timestamp")
            group_id_column: Path to group_id field (supports nested, e.g., "metadata.group_id")
            append_timestamp_to_text: If True, append formatted timestamp to text before embedding
            timestamp_format: Format for appended timestamp (iso, human, unix, relative)
            num_samples: Optional limit on number of samples to load
            **kwargs: Additional arguments
        """
        self.file_path = Path(file_path)
        super().__init__(f"temporal-{self.file_path.stem}")

        self.text_column = text_column
        self.timestamp_column = timestamp_column
        self.group_id_column = group_id_column
        self.append_timestamp_to_text = append_timestamp_to_text
        self.timestamp_format = timestamp_format
        self.num_samples = num_samples
        self.kwargs = kwargs

        # Temporal-specific attributes
        self.groups = {}  # group_id -> list of sample indices
        self.timestamps = []  # parallel to self.samples
        self.group_start_times = {}  # For relative timestamp formatting

        self.load()

    def _get_nested_field(self, item: Dict, field_path: str) -> Any:
        """
        Get a field from a nested dictionary using dot notation.

        Args:
            item: Dictionary to search
            field_path: Path to field (e.g., "metadata.timestamp")

        Returns:
            Field value or None if not found
        """
        parts = field_path.split('.')
        value = item

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _format_timestamp(self, timestamp: datetime, group_id: str) -> str:
        """
        Format timestamp for appending to text.

        Args:
            timestamp: Datetime object
            group_id: Group identifier (for relative formatting)

        Returns:
            Formatted timestamp string
        """
        if self.timestamp_format == "iso":
            return timestamp.isoformat()
        elif self.timestamp_format == "human":
            return timestamp.strftime("%B %d, %Y at %I:%M %p")
        elif self.timestamp_format == "unix":
            return str(int(timestamp.timestamp()))
        elif self.timestamp_format == "relative":
            if group_id not in self.group_start_times:
                return "Hour 0"
            start_time = self.group_start_times[group_id]
            hours = (timestamp - start_time).total_seconds() / 3600
            return f"Hour {int(hours)}"
        else:
            return timestamp.isoformat()

    def load(self) -> None:
        """Load temporal dataset from JSONL file."""
        logger.info(f"Loading temporal dataset from {self.file_path}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        # First pass: collect group start times for relative formatting
        if self.append_timestamp_to_text and self.timestamp_format == "relative":
            self._collect_group_start_times()

        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    # Extract text
                    text = self._get_nested_field(item, self.text_column)
                    if not text:
                        logger.warning(f"Missing '{self.text_column}' field, skipping")
                        continue

                    # Extract timestamp
                    timestamp_str = self._get_nested_field(item, self.timestamp_column)
                    if not timestamp_str:
                        logger.warning(f"Missing '{self.timestamp_column}' field, skipping")
                        continue

                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid timestamp format '{timestamp_str}': {e}, skipping")
                        continue

                    # Extract group_id
                    group_id = self._get_nested_field(item, self.group_id_column)
                    if not group_id:
                        logger.warning(f"Missing '{self.group_id_column}' field, skipping")
                        continue

                    # Extract metadata (entire dict)
                    metadata = item.get("metadata", {})
                    variation_id = metadata.get("variation_id")

                    # Optionally append timestamp to text
                    if self.append_timestamp_to_text:
                        formatted_ts = self._format_timestamp(timestamp, group_id)
                        text = f"{text} [{formatted_ts}]"

                    # Create sample
                    sample = DatasetSample(
                        text1=str(text),
                        text2=None,
                        label=None,  # No labels for temporal retrieval
                        metadata={
                            **metadata,
                            "timestamp": timestamp,
                            "timestamp_str": timestamp_str,
                            "group_id": group_id,
                            "variation_id": variation_id,
                            "sample_idx": len(self.samples),  # Store index for easy lookup
                            "timestamp_appended": self.append_timestamp_to_text,
                        }
                    )

                    self.samples.append(sample)
                    self.timestamps.append(timestamp)

                    # Track groups
                    if group_id not in self.groups:
                        self.groups[group_id] = []
                    self.groups[group_id].append(len(self.samples) - 1)

                    count += 1
                    if self.num_samples and count >= self.num_samples:
                        break

        logger.info(f"Loaded {len(self.samples)} temporal samples across {len(self.groups)} groups")
        logger.info(f"Average group size: {len(self.samples) / len(self.groups):.1f} facts/group")
        if self.append_timestamp_to_text:
            logger.info(f"Timestamps appended to text (format: {self.timestamp_format})")

    def _collect_group_start_times(self) -> None:
        """Collect the earliest timestamp for each group (for relative formatting)."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    timestamp_str = self._get_nested_field(item, self.timestamp_column)
                    group_id = self._get_nested_field(item, self.group_id_column)

                    if timestamp_str and group_id:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if group_id not in self.group_start_times or timestamp < self.group_start_times[group_id]:
                                self.group_start_times[group_id] = timestamp
                        except (ValueError, AttributeError):
                            pass

    def get_group_samples(self, group_id: str) -> list:
        """
        Get all samples belonging to a specific group.

        Args:
            group_id: Group identifier

        Returns:
            List of DatasetSample objects
        """
        if group_id not in self.groups:
            return []

        indices = self.groups[group_id]
        return [self.samples[idx] for idx in indices]

    def get_group_id(self, sample_idx: int) -> str:
        """
        Get group_id for a sample.

        Args:
            sample_idx: Sample index

        Returns:
            Group identifier
        """
        return self.samples[sample_idx].metadata.get("group_id")

    def get_timestamp(self, sample_idx: int) -> datetime:
        """
        Get timestamp for a sample.

        Args:
            sample_idx: Sample index

        Returns:
            Timestamp as datetime object
        """
        return self.timestamps[sample_idx]

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        base_info = super().get_info()

        temporal_info = {
            "num_groups": len(self.groups),
            "avg_group_size": len(self.samples) / len(self.groups) if self.groups else 0,
            "min_timestamp": min(self.timestamps) if self.timestamps else None,
            "max_timestamp": max(self.timestamps) if self.timestamps else None,
            "is_temporal": True,
            "timestamp_appended": self.append_timestamp_to_text,
            "timestamp_format": self.timestamp_format if self.append_timestamp_to_text else None,
        }

        return {**base_info, **temporal_info}


def load_temporal_dataset(file_path: str, **kwargs) -> TemporalDataset:
    """
    Convenience function to load temporal dataset.

    Args:
        file_path: Path to JSONL file
        **kwargs: Arguments passed to TemporalDataset

    Returns:
        TemporalDataset instance
    """
    return TemporalDataset(file_path, **kwargs)
