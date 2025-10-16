# Embedding Lab Architecture

## System Overview

Embedding Lab is a modular, production-ready platform for evaluating embedding models. The architecture follows clean separation of concerns with clearly defined layers.

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│                   (User Interface)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Experiment Layer                           │
│              (Orchestration & Control)                       │
│  ┌─────────────────────────────────────────────────┐        │
│  │           ExperimentRunner                       │        │
│  │  - Loads configurations                          │        │
│  │  - Coordinates evaluations                       │        │
│  │  - Tracks metrics                                │        │
│  │  - Manages MLflow                                │        │
│  └─────────────────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼────────┐ ┌──▼───────┐ ┌───▼─────────┐
│   Embeddings   │ │ Datasets │ │ Evaluation  │
│     Layer      │ │  Layer   │ │   Layer     │
└────────────────┘ └──────────┘ └─────────────┘
```

## Layer Details

### 1. CLI Layer (`cli.py`)

**Purpose**: User-facing command-line interface

**Responsibilities**:
- Parse command-line arguments
- Display formatted output
- Validate user inputs
- Invoke experiment runner

**Key Commands**:
- `run`: Execute experiments
- `init`: Create configuration templates
- `validate`: Check configuration validity
- `list-models`: Show available models
- `list-datasets`: Show available datasets

**Technologies**: Click, Rich (formatting)

### 2. Experiment Layer (`experiments/`)

**Purpose**: Orchestrate the entire experiment workflow

**Components**:

#### ExperimentRunner (`experiments/runner.py`)
- Central orchestrator
- Loads configurations
- Instantiates models and datasets
- Executes evaluations
- Tracks and saves results
- Manages MLflow logging

**Key Methods**:
```python
run()                      # Execute full experiment
_run_model_experiments()   # Run for one model
_run_dataset_experiments() # Run for one dataset
_run_task()               # Run single task evaluation
```

**Flow**:
```
1. Load configuration
2. For each model:
   a. Load embedder
   b. For each dataset:
      i. Load dataset
      ii. For each task:
          - Run evaluation
          - Track metrics
3. Save all results
4. Generate report
```

### 3. Embeddings Layer (`embeddings/`)

**Purpose**: Provide embedding model abstractions

**Components**:

#### BaseEmbedder (`embeddings/base.py`)
Abstract interface defining:
- `encode()`: Generate embeddings
- `get_embedding_dim()`: Get dimensionality
- `get_model_info()`: Get model metadata

#### SingleEmbedder (`embeddings/single.py`)
- Wraps sentence-transformer models
- Single model encoding
- Configurable normalization

#### HierarchicalEmbedder (`embeddings/hierarchical.py`)
- **Core innovation**: Dual-layer architecture
- Coarse model: Fast, broad semantics
- Fine model: Detailed, nuanced semantics
- Combination strategies:
  - Concatenation (default)
  - Weighted sum
  - Learned projection (future)

**Hierarchical Flow**:
```
Input Text
    │
    ├──> Coarse Model  ──┐
    │    (Fast)          │
    │                    ├──> Combine ──> Final Embedding
    └──> Fine Model   ──┘
         (Detailed)
```

### 4. Datasets Layer (`datasets/`)

**Purpose**: Load and manage datasets

**Components**:

#### BaseDataset (`datasets/base.py`)
Abstract interface defining:
- `load()`: Load dataset
- `get_texts()`: Extract texts
- `get_pairs()`: Extract pairs
- `get_labels()`: Extract labels
- `get_info()`: Dataset metadata

#### BenchmarkDatasets (`datasets/benchmarks.py`)
Standard datasets:
- STS-B: Semantic similarity
- MS MARCO: Information retrieval
- AG News: Classification
- TREC: Question classification

#### CustomDataset (`datasets/custom.py`)
User dataset loader:
- Supports CSV, TSV, JSON, JSONL
- Flexible column mapping
- Auto-format detection

**Dataset Structure**:
```python
DatasetSample:
    text1: str          # Primary text
    text2: str | None   # Second text (for pairs)
    label: Any | None   # Label (for supervised tasks)
    metadata: dict      # Additional info
```

### 5. Evaluation Layer (`evaluation/`)

**Purpose**: Implement task-specific evaluations

**Components**:

#### SimilarityEvaluator (`evaluation/similarity.py`)
- Computes cosine similarities
- Metrics: Spearman, Pearson correlation
- For semantic similarity tasks

#### RetrievalEvaluator (`evaluation/retrieval.py`)
- Ranks documents for queries
- Metrics: MRR, Recall@k, Precision@k
- For information retrieval tasks

#### ClassificationEvaluator (`evaluation/classification.py`)
- Trains linear classifier on embeddings
- Metrics: Accuracy, F1, Precision, Recall
- For text classification tasks

#### ClusteringEvaluator (`evaluation/clustering.py`)
- Clusters embeddings (K-Means, Agglomerative)
- Metrics: Silhouette, Davies-Bouldin, ARI, NMI
- For clustering tasks

**Evaluation Flow**:
```
Dataset → Embedder.encode() → Embeddings → Task-Specific Evaluation → Metrics
```

### 6. Utilities Layer (`utils/`)

**Purpose**: Support functionality

**Components**:

#### Configuration (`utils/config.py`)
- YAML parsing
- Configuration dataclasses
- Validation
- Save/load functionality

**Configuration Hierarchy**:
```
ExperimentConfig
    ├── ModelConfig[]
    ├── DatasetConfig[]
    └── Task settings
```

#### Metrics (`utils/metrics.py`)
- MetricsTracker: Accumulate results
- Aggregation functions
- Result formatting
- CSV/JSON export

## Data Flow

### Complete Experiment Flow

```
1. User runs CLI command
   ↓
2. CLI loads configuration
   ↓
3. ExperimentRunner initializes
   ↓
4. For each model:
   ├─> Load embedder (Single or Hierarchical)
   ├─> For each dataset:
   │   ├─> Load dataset
   │   └─> For each task:
   │       ├─> Encode texts
   │       ├─> Run evaluation
   │       └─> Track metrics
   └─> Log to MLflow
   ↓
5. Save results (JSON, CSV, report)
   ↓
6. Display summary
```

### Hierarchical Embedding Flow

```
Input: ["text1", "text2", ...]
    ↓
Coarse Model (e.g., all-MiniLM-L6-v2)
    ├─> Encode → [384-dim embeddings]
    ↓
Fine Model (e.g., all-mpnet-base-v2)
    ├─> Encode → [768-dim embeddings]
    ↓
Combine (concat)
    ├─> Concatenate → [1152-dim embeddings]
    ↓
Normalize (optional)
    ├─> L2 normalize
    ↓
Output: Final embeddings
```

## Design Patterns

### 1. Abstract Base Classes
All layers use ABC pattern for extensibility:
- BaseEmbedder
- BaseDataset
- Configuration dataclasses

### 2. Factory Pattern
Used for creating instances:
- `get_benchmark_dataset(name)`
- `load_custom_dataset(path)`

### 3. Strategy Pattern
Different combination methods in HierarchicalEmbedder:
- Concat strategy
- Weighted sum strategy
- Learned strategy (extensible)

### 4. Template Method
ExperimentRunner defines template:
```python
def run():
    for model in models:
        for dataset in datasets:
            for task in tasks:
                evaluate()
```

## Extension Points

### Adding New Models
1. Extend `BaseEmbedder`
2. Implement `encode()`, `get_embedding_dim()`, `get_model_info()`
3. Add to embeddings package

### Adding New Datasets
1. Extend `BaseDataset`
2. Implement `load()`
3. Add to datasets package

### Adding New Tasks
1. Create evaluator class
2. Implement `evaluate()` method
3. Add to evaluation package
4. Update ExperimentRunner

### Adding New Combination Methods
1. Add method to HierarchicalEmbedder
2. Implement in `_combine_embeddings()`

## Configuration System

### YAML Structure
```yaml
name: string
description: string
models:
  - type: single | hierarchical
    name: string
    ...
datasets:
  - type: benchmark | custom
    name: string
    ...
tasks: [similarity, retrieval, classification, clustering]
settings:
  batch_size: int
  output_dir: string
  ...
```

### Configuration Loading
```
YAML file → load_config() → ExperimentConfig dataclass → Runner
```

## Metrics & Tracking

### Metrics Flow
```
Evaluation → Metrics dict → MetricsTracker → Results files
                                           → MLflow
```

### Output Files
- `results.json`: Raw results
- `results.csv`: Tabular format
- `comparison_*.csv`: Metric comparisons
- `report.txt`: Human-readable summary

### MLflow Integration
- Automatic experiment tracking
- Parameter logging
- Metric logging
- Run comparison

## Error Handling

### Levels
1. **CLI**: User-friendly error messages
2. **Runner**: Graceful degradation, continue on errors
3. **Components**: Raise specific exceptions
4. **Logging**: Comprehensive logging with loguru

### Validation
- Configuration validation before run
- Dataset validation on load
- Model validation on instantiation

## Performance Considerations

### Batch Processing
- Configurable batch sizes
- Memory-efficient streaming
- GPU utilization

### Caching
- Model caching (sentence-transformers)
- Dataset caching (HuggingFace)

### Parallelization
- Batch encoding
- Future: Multi-GPU support

## Testing Strategy

### Unit Tests
- Test individual components
- Mock dependencies
- Test edge cases

### Integration Tests
- Test component interactions
- End-to-end flows
- Real model tests

### Test Structure
```
tests/
├── test_embeddings.py
├── test_datasets.py
├── test_evaluation.py
└── test_integration.py
```

## Dependencies

### Core
- sentence-transformers: Embedding models
- torch: Deep learning backend
- scikit-learn: Evaluation metrics

### Data
- datasets: Benchmark loading
- pandas: Data manipulation

### Infrastructure
- click: CLI framework
- rich: Terminal formatting
- mlflow: Experiment tracking
- loguru: Logging

## Security Considerations

- No credential storage
- Safe file operations
- Input validation
- Resource limits

## Future Enhancements

### Planned
- [ ] Learned combination methods
- [ ] Cross-encoder support
- [ ] More benchmarks (BEIR, MTEB)
- [ ] Multi-GPU support
- [ ] REST API
- [ ] Web UI

### Extensibility
System designed for easy extension:
- Modular architecture
- Abstract interfaces
- Plugin system (future)
- Configuration-driven

## Conclusion

The architecture prioritizes:
- **Modularity**: Clear component separation
- **Extensibility**: Easy to add new features
- **Usability**: Simple CLI and configuration
- **Production-readiness**: Proper error handling, logging, tracking
- **Performance**: Batch processing, GPU support
- **Reproducibility**: Configuration files, experiment tracking
