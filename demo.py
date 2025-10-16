"""Quick start demo script."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import SingleEmbedder, HierarchicalEmbedder
from data_loaders import DatasetSample, get_benchmark_dataset
from data_loaders.base import BaseDataset
from evaluation import SimilarityEvaluator
from loguru import logger


def demo_embeddings():
    """Demonstrate embedding models."""
    print("\n" + "=" * 80)
    print("DEMO: Embedding Models")
    print("=" * 80 + "\n")
    
    # Single model
    print("1. Single Model (all-MiniLM-L6-v2)")
    single = SingleEmbedder("all-MiniLM-L6-v2")
    embeddings = single.encode(["Hello world", "Good morning"])
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {single.get_embedding_dim()}")
    
    # Hierarchical model
    print("\n2. Hierarchical Model (coarse + fine)")
    hierarchical = HierarchicalEmbedder(
        coarse_model="all-MiniLM-L6-v2",
        fine_model="all-mpnet-base-v2",
        combination_method="concat"
    )
    embeddings = hierarchical.encode(["Hello world", "Good morning"])
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {hierarchical.get_embedding_dim()}")
    print(f"   Info: {hierarchical.get_model_info()}")


def demo_datasets():
    """Demonstrate dataset loading."""
    print("\n" + "=" * 80)
    print("DEMO: Datasets")
    print("=" * 80 + "\n")
    
    # Create a simple mock dataset
    class MockDataset(BaseDataset):
        def __init__(self):
            super().__init__("mock")
            self.load()
        
        def load(self):
            self.samples = [
                DatasetSample("The cat sat", "The dog sat", 0.8),
                DatasetSample("I love pizza", "I hate pizza", 0.2),
                DatasetSample("Good morning", "Good afternoon", 0.7),
            ]
    
    dataset = MockDataset()
    print(f"Dataset: {dataset.name}")
    print(f"Size: {len(dataset)}")
    print(f"Info: {dataset.get_info()}")
    print(f"\nFirst sample:")
    print(f"  Text1: {dataset[0].text1}")
    print(f"  Text2: {dataset[0].text2}")
    print(f"  Label: {dataset[0].label}")


def demo_evaluation():
    """Demonstrate evaluation."""
    print("\n" + "=" * 80)
    print("DEMO: Evaluation")
    print("=" * 80 + "\n")
    
    # Create mock dataset
    class MockDataset(BaseDataset):
        def __init__(self):
            super().__init__("mock")
            self.load()
        
        def load(self):
            self.samples = [
                DatasetSample("The cat sat on the mat", "The dog sat on the mat", 0.9),
                DatasetSample("I love programming", "I hate bugs", 0.3),
                DatasetSample("Good morning everyone", "Good evening all", 0.7),
                DatasetSample("Python is great", "Python is awesome", 0.95),
                DatasetSample("The weather is nice", "The weather is terrible", 0.4),
            ]
    
    dataset = MockDataset()
    
    # Single model
    print("1. Single Model Evaluation")
    single = SingleEmbedder("all-MiniLM-L6-v2")
    evaluator = SimilarityEvaluator(single)
    metrics = evaluator.evaluate(dataset)
    print(f"   Spearman: {metrics['spearman']:.4f}")
    print(f"   Pearson: {metrics['pearson']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    
    # Hierarchical model
    print("\n2. Hierarchical Model Evaluation")
    hierarchical = HierarchicalEmbedder(
        coarse_model="all-MiniLM-L6-v2",
        fine_model="all-MiniLM-L6-v2",  # Using same for speed
        combination_method="concat"
    )
    evaluator = SimilarityEvaluator(hierarchical)
    metrics = evaluator.evaluate(dataset)
    print(f"   Spearman: {metrics['spearman']:.4f}")
    print(f"   Pearson: {metrics['pearson']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")


def main():
    """Run all demos."""
    logger.remove()  # Reduce logging for demo
    
    print("\n" + "#" * 80)
    print("# EMBEDDING LAB - QUICK START DEMO")
    print("#" * 80)
    
    demo_embeddings()
    demo_datasets()
    demo_evaluation()
    
    print("\n" + "#" * 80)
    print("# Demo complete!")
    print("#" * 80)
    print("\nNext steps:")
    print("1. python cli.py init config/experiments/my_experiment.yaml")
    print("2. Edit the config file")
    print("3. python cli.py run config/experiments/my_experiment.yaml")
    print("\nFor more info: python cli.py --help")
    print()


if __name__ == "__main__":
    main()
