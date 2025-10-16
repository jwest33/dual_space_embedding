"""Basic tests for embedding lab."""
import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import SingleEmbedder, HierarchicalEmbedder
from data_loaders import DatasetSample, CustomDataset
from evaluation import SimilarityEvaluator
from utils import ModelConfig, DatasetConfig, ExperimentConfig


class TestEmbeddings:
    """Test embedding models."""
    
    def test_single_embedder(self):
        """Test single embedder basic functionality."""
        embedder = SingleEmbedder(model_name="all-MiniLM-L6-v2")
        
        # Test single text
        embedding = embedder.encode("Hello world")
        assert embedding.shape == (1, 384)
        
        # Test multiple texts
        embeddings = embedder.encode(["Hello", "World"])
        assert embeddings.shape == (2, 384)
        
        # Test dimension
        assert embedder.get_embedding_dim() == 384
        
    def test_hierarchical_embedder(self):
        """Test hierarchical embedder basic functionality."""
        embedder = HierarchicalEmbedder(
            coarse_model="all-MiniLM-L6-v2",
            fine_model="all-MiniLM-L6-v2",
            combination_method="concat"
        )
        
        # Test single text
        embedding = embedder.encode("Hello world")
        assert embedding.shape == (1, 768)  # 384 + 384
        
        # Test multiple texts
        embeddings = embedder.encode(["Hello", "World"])
        assert embeddings.shape == (2, 768)
        
        # Test dimension
        assert embedder.get_embedding_dim() == 768
        
    def test_hierarchical_normalization(self):
        """Test that normalization works correctly."""
        embedder = HierarchicalEmbedder(
            coarse_model="all-MiniLM-L6-v2",
            fine_model="all-MiniLM-L6-v2",
            normalize=True
        )
        
        embeddings = embedder.encode(["Test text"])
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5


class TestDatasets:
    """Test dataset loading."""
    
    def test_dataset_sample(self):
        """Test DatasetSample creation."""
        sample = DatasetSample(
            text1="Hello",
            text2="World",
            label=1.0
        )
        
        assert sample.text1 == "Hello"
        assert sample.text2 == "World"
        assert sample.label == 1.0
        
    def test_custom_dataset_csv(self, tmp_path):
        """Test loading custom CSV dataset."""
        # Create temporary CSV
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "text,label\n"
            "Hello,1\n"
            "World,0\n"
        )
        
        dataset = CustomDataset(
            file_path=str(csv_path),
            text1_column="text",
            label_column="label"
        )
        
        assert len(dataset) == 2
        assert dataset[0].text1 == "Hello"
        assert dataset[0].label == "1"


class TestConfiguration:
    """Test configuration management."""
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig(
            type="single",
            name="test",
            model_name="all-MiniLM-L6-v2"
        )
        
        assert config.type == "single"
        assert config.name == "test"
        
    def test_dataset_config(self):
        """Test dataset configuration."""
        config = DatasetConfig(
            type="benchmark",
            name="sts-b",
            split="test"
        )
        
        assert config.type == "benchmark"
        assert config.name == "sts-b"
        
    def test_experiment_config(self):
        """Test experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            models=[],
            datasets=[],
            tasks=["similarity"]
        )
        
        assert config.name == "test_experiment"
        assert "similarity" in config.tasks


class TestEvaluation:
    """Test evaluation modules."""
    
    def test_similarity_evaluator(self):
        """Test similarity evaluator with mock data."""
        # Create mock dataset
        from data_loaders.base import BaseDataset
        
        class MockDataset(BaseDataset):
            def __init__(self):
                super().__init__("mock")
                self.samples = [
                    DatasetSample("cat", "dog", 0.8),
                    DatasetSample("hello", "hi", 0.9),
                ]
            
            def load(self):
                pass
        
        dataset = MockDataset()
        embedder = SingleEmbedder("all-MiniLM-L6-v2")
        evaluator = SimilarityEvaluator(embedder)
        
        metrics = evaluator.evaluate(dataset, batch_size=2)
        
        assert "spearman" in metrics
        assert "pearson" in metrics
        assert "mae" in metrics
        assert metrics["num_samples"] == 2


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_single_model(self):
        """Test end-to-end with single model."""
        # Create model
        embedder = SingleEmbedder("all-MiniLM-L6-v2")

        # Create simple dataset
        from data_loaders.base import BaseDataset
        
        class SimpleDataset(BaseDataset):
            def __init__(self):
                super().__init__("simple")
                self.samples = [
                    DatasetSample("I love this", "I hate this", 0.2),
                    DatasetSample("good movie", "great film", 0.9),
                ]
            
            def load(self):
                pass
        
        dataset = SimpleDataset()
        
        # Evaluate
        evaluator = SimilarityEvaluator(embedder)
        metrics = evaluator.evaluate(dataset)
        
        assert isinstance(metrics["spearman"], float)
        assert -1 <= metrics["spearman"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
