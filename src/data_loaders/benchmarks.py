"""Standard benchmark dataset loaders."""
from typing import Optional, List
import numpy as np
from datasets import load_dataset  # HuggingFace datasets library
from loguru import logger

from .base import BaseDataset, DatasetSample


class STSBenchmark(BaseDataset):
    """Semantic Textual Similarity Benchmark dataset."""

    def __init__(self, split: str = "test", num_samples: Optional[int] = None):
        """
        Initialize STS-B dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Number of samples to load (None for all)
        """
        super().__init__(f"sts-b-{split}")
        self.split = split
        self.num_samples = num_samples
        self.load()

    def load(self) -> None:
        """Load STS-B dataset from HuggingFace."""
        logger.info(f"Loading STS-B {self.split} split")

        dataset = load_dataset("glue", "stsb", split=self.split)

        count = 0
        for item in dataset:
            if self.num_samples and count >= self.num_samples:
                break

            sample = DatasetSample(
                text1=item["sentence1"],
                text2=item["sentence2"],
                label=item["label"],  # 0-5 similarity score
                metadata={"idx": item["idx"]}
            )
            self.samples.append(sample)
            count += 1

        logger.info(f"Loaded {len(self.samples)} samples from STS-B {self.split}")


class MSMARCODataset(BaseDataset):
    """MS MARCO passage ranking dataset."""
    
    def __init__(
        self,
        split: str = "train",
        num_samples: Optional[int] = 10000
    ):
        """
        Initialize MS MARCO dataset.
        
        Args:
            split: Dataset split
            num_samples: Number of samples to load (None for all)
        """
        super().__init__(f"msmarco-{split}")
        self.split = split
        self.num_samples = num_samples
        self.load()
        
    def load(self) -> None:
        """Load MS MARCO dataset."""
        logger.info(f"Loading MS MARCO {self.split} split")
        
        try:
            dataset = load_dataset("ms_marco", "v1.1", split=self.split)
            
            count = 0
            for item in dataset:
                if self.num_samples and count >= self.num_samples:
                    break
                    
                # Use query and positive passage
                if item["passages"]["is_selected"]:
                    selected_idx = item["passages"]["is_selected"].index(1)
                    passage = item["passages"]["passage_text"][selected_idx]
                    
                    sample = DatasetSample(
                        text1=item["query"],
                        text2=passage,
                        label=1,  # Relevant
                        metadata={"query_id": item["query_id"]}
                    )
                    self.samples.append(sample)
                    count += 1
                    
            logger.info(f"Loaded {len(self.samples)} samples from MS MARCO {self.split}")
            
        except Exception as e:
            logger.warning(f"Could not load MS MARCO: {e}. Using mock data.")
            self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock data for testing."""
        queries = [
            "What is machine learning?",
            "How to learn Python programming?",
            "Best restaurants in New York",
        ]
        passages = [
            "Machine learning is a subset of artificial intelligence...",
            "Python is a high-level programming language...",
            "New York has many excellent restaurants...",
        ]
        
        for i, (q, p) in enumerate(zip(queries, passages)):
            sample = DatasetSample(
                text1=q,
                text2=p,
                label=1,
                metadata={"query_id": f"mock_{i}"}
            )
            self.samples.append(sample)


class AGNewsDataset(BaseDataset):
    """AG News classification dataset."""
    
    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = 5000
    ):
        """
        Initialize AG News dataset.
        
        Args:
            split: Dataset split ('train', 'test')
            num_samples: Number of samples to load
        """
        super().__init__(f"ag-news-{split}")
        self.split = split
        self.num_samples = num_samples
        self.load()
        
    def load(self) -> None:
        """Load AG News dataset."""
        logger.info(f"Loading AG News {self.split} split")
        
        dataset = load_dataset("ag_news", split=self.split)
        
        count = 0
        for item in dataset:
            if self.num_samples and count >= self.num_samples:
                break
                
            sample = DatasetSample(
                text1=item["text"],
                label=item["label"],  # 0: World, 1: Sports, 2: Business, 3: Sci/Tech
                metadata={"category": item["label"]}
            )
            self.samples.append(sample)
            count += 1
            
        logger.info(f"Loaded {len(self.samples)} samples from AG News {self.split}")


class TRECDataset(BaseDataset):
    """TREC question classification dataset."""

    def __init__(self, split: str = "test", num_samples: Optional[int] = None):
        """
        Initialize TREC dataset.

        Args:
            split: Dataset split ('train', 'test')
            num_samples: Number of samples to load (None for all)
        """
        super().__init__(f"trec-{split}")
        self.split = split
        self.num_samples = num_samples
        self.load()

    def load(self) -> None:
        """Load TREC dataset."""
        logger.info(f"Loading TREC {self.split} split")

        try:
            # Try loading with trust_remote_code for compatibility
            try:
                dataset = load_dataset("trec", split=self.split, trust_remote_code=True)
            except TypeError:
                # Fallback for older versions without trust_remote_code
                dataset = load_dataset("trec", split=self.split)

            count = 0
            for item in dataset:
                if self.num_samples and count >= self.num_samples:
                    break

                sample = DatasetSample(
                    text1=item["text"],
                    label=item["label-coarse"],  # Coarse category
                    metadata={
                        "fine_label": item["label-fine"]
                    }
                )
                self.samples.append(sample)
                count += 1

            logger.info(f"Loaded {len(self.samples)} samples from TREC {self.split}")

        except Exception as e:
            logger.warning(f"Could not load TREC: {e}. Using mock data.")
            self._create_mock_data()

    def _create_mock_data(self):
        """Create mock data for testing."""
        questions = [
            ("What is the capital of France?", 3),  # LOC (location)
            ("Who invented the telephone?", 2),  # HUM (human)
            ("When did World War II end?", 5),  # NUM (numeric)
            ("What does HTTP stand for?", 0),  # ABBR (abbreviation)
            ("How do plants photosynthesize?", 1),  # DESC (description)
            ("Which country won the World Cup in 2018?", 3),  # LOC
        ]

        for i, (text, label) in enumerate(questions):
            sample = DatasetSample(
                text1=text,
                label=label,
                metadata={"mock_id": f"mock_{i}", "fine_label": label}
            )
            self.samples.append(sample)


# Factory function to get benchmark datasets
def get_benchmark_dataset(
    name: str,
    split: str = "test",
    **kwargs
) -> BaseDataset:
    """
    Get a benchmark dataset by name.
    
    Args:
        name: Dataset name ('sts-b', 'msmarco', 'ag-news', 'trec')
        split: Dataset split
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dataset instance
    """
    datasets = {
        "sts-b": STSBenchmark,
        "stsb": STSBenchmark,
        "msmarco": MSMARCODataset,
        "ag-news": AGNewsDataset,
        "agnews": AGNewsDataset,
        "trec": TRECDataset,
    }
    
    name_lower = name.lower()
    if name_lower not in datasets:
        raise ValueError(
            f"Unknown benchmark dataset: {name}. "
            f"Available: {list(datasets.keys())}"
        )
    
    return datasets[name_lower](split=split, **kwargs)
