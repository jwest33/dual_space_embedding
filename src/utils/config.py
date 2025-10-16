"""Configuration utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    type: str  # 'single' or 'hierarchical'
    name: str
    
    # For single models
    model_name: str = None
    
    # For hierarchical models
    coarse_model: str = None
    fine_model: str = None
    combination_method: str = "concat"
    coarse_weight: float = 0.5
    fine_weight: float = 0.5
    
    # Common parameters
    device: str = None
    normalize: bool = True
    
    def __post_init__(self):
        if self.type == "single" and self.model_name is None:
            raise ValueError("single model requires model_name")
        if self.type == "hierarchical":
            if self.coarse_model is None or self.fine_model is None:
                raise ValueError("hierarchical model requires coarse_model and fine_model")


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    type: str  # 'benchmark' or 'custom'
    name: str
    split: str = "test"
    
    # For custom datasets
    file_path: str = None
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"
    format: str = None
    
    # For benchmark datasets
    num_samples: int = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str = ""
    
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    
    tasks: List[str] = field(default_factory=lambda: ["similarity", "retrieval", "classification", "clustering"])
    
    batch_size: int = 32
    output_dir: str = "results"
    
    # Task-specific settings
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    classification_classifier: str = "logistic_regression"
    clustering_algorithm: str = "kmeans"
    clustering_n_clusters: int = None
    
    # MLflow tracking
    mlflow_tracking: bool = True
    mlflow_uri: str = "mlruns"


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        ExperimentConfig instance
    """
    logger.info(f"Loading configuration from {config_path}")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Parse models
    models = []
    for model_dict in config_dict.get("models", []):
        models.append(ModelConfig(**model_dict))
    
    # Parse datasets
    datasets = []
    for dataset_dict in config_dict.get("datasets", []):
        datasets.append(DatasetConfig(**dataset_dict))
    
    # Create experiment config
    exp_config = ExperimentConfig(
        name=config_dict.get("name", "experiment"),
        description=config_dict.get("description", ""),
        models=models,
        datasets=datasets,
        tasks=config_dict.get("tasks", ["similarity", "retrieval", "classification", "clustering"]),
        batch_size=config_dict.get("batch_size", 32),
        output_dir=config_dict.get("output_dir", "results"),
        retrieval_k_values=config_dict.get("retrieval_k_values", [1, 5, 10, 20]),
        classification_classifier=config_dict.get("classification_classifier", "logistic_regression"),
        clustering_algorithm=config_dict.get("clustering_algorithm", "kmeans"),
        clustering_n_clusters=config_dict.get("clustering_n_clusters"),
        mlflow_tracking=config_dict.get("mlflow_tracking", True),
        mlflow_uri=config_dict.get("mlflow_uri", "mlruns"),
    )
    
    logger.info(
        f"Configuration loaded: {len(models)} models, "
        f"{len(datasets)} datasets, {len(exp_config.tasks)} tasks"
    )
    
    return exp_config


def save_config(config: ExperimentConfig, output_path: str) -> None:
    """
    Save experiment configuration to YAML file.
    
    Args:
        config: ExperimentConfig to save
        output_path: Output path for YAML file
    """
    # Convert to dict
    config_dict = {
        "name": config.name,
        "description": config.description,
        "models": [
            {k: v for k, v in vars(m).items() if v is not None}
            for m in config.models
        ],
        "datasets": [
            {k: v for k, v in vars(d).items() if v is not None}
            for d in config.datasets
        ],
        "tasks": config.tasks,
        "batch_size": config.batch_size,
        "output_dir": config.output_dir,
        "retrieval_k_values": config.retrieval_k_values,
        "classification_classifier": config.classification_classifier,
        "clustering_algorithm": config.clustering_algorithm,
        "mlflow_tracking": config.mlflow_tracking,
        "mlflow_uri": config.mlflow_uri,
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_path}")
