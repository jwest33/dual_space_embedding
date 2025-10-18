# Embedding Lab

Platform for evaluating embedding models across similarity, retrieval, classification, clustering, and temporal tasks. Supports single models and hierarchical dual-layer combinations.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python verify_installation.py
```

## Quick Start

### Run a Similarity Experiment

```bash
# Run pre-configured experiment
python cli.py run config/experiments/stsb_hierarchal_concat.yaml

# View results
cat results/my_experiment/report.txt
```

### Run a Temporal Experiment

```bash
# Without timestamp augmentation (pure semantic)
python cli.py run config/experiments/temporal_experiment.yaml

# With timestamp augmentation (temporal-aware)
python cli.py run config/experiments/temporal_experiment_with_timestamps.yaml

# View examples
cat results/temporal_facts_experiment/examples/baseline_minilm_temporal_facts_social_examples.txt
```

### View Results in MLflow

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## Configuration

### Models

**Single Model:**
```yaml
models:
- type: single
  name: baseline
  model_name: all-MiniLM-L6-v2  # Any sentence-transformer model
  normalize: true
```

**Hierarchical Model:**
```yaml
models:
- type: hierarchical
  name: hierarchical_concat
  coarse_model: all-MiniLM-L6-v2
  fine_model: all-mpnet-base-v2
  combination_method: concat  # concat, weighted_sum, or hyperbolic
  normalize: true
```

**Combination Methods:**
- `concat`: Concatenate embeddings (coarse_dim + fine_dim)
- `weighted_sum`: Weighted average (requires same dimensions)
- `hyperbolic`: Hyperbolic geometry combination (fine_dim output)

### Datasets

**Benchmark Dataset:**
```yaml
datasets:
- type: benchmark
  name: sts-b  # sts-b, msmarco, ag-news, trec
  split: validation
```

**Custom Dataset:**
```yaml
datasets:
- type: custom
  name: my_data
  file_path: datasets/my_data.csv
  text1_column: text
  label_column: label
  format: csv  # csv, tsv, json, jsonl
```

**Temporal Dataset:**
```yaml
datasets:
- type: temporal
  name: temporal_facts
  file_path: datasets/temporal_facts_social_20251018_055003.jsonl
  text_column: instruction
  timestamp_column: metadata.timestamp
  append_timestamp_to_text: false  # Set true to include timestamps in embeddings
  timestamp_format: relative  # iso, human, unix, relative
```

### Tasks

```yaml
tasks:
- similarity      # Spearman/Pearson correlation
- retrieval       # MRR, Recall@k, Precision@k
- classification  # Accuracy, F1
- clustering      # Silhouette score
- temporal        # Temporal ordering, group discrimination
```

### Full Example

```yaml
name: my_experiment
description: 'Compare single vs hierarchical models'

models:
- type: single
  name: baseline
  model_name: all-MiniLM-L6-v2

- type: hierarchical
  name: hierarchical
  coarse_model: all-MiniLM-L6-v2
  fine_model: all-mpnet-base-v2
  combination_method: concat

datasets:
- type: benchmark
  name: sts-b
  split: validation

tasks:
- similarity
- retrieval

batch_size: 32
output_dir: results
mlflow_tracking: true
```

## Tasks & Metrics

### Similarity
Evaluates semantic text similarity.

**Metrics:** spearman, pearson, mae, rmse
**Requires:** Text pairs with similarity scores

### Retrieval
Evaluates document retrieval.

**Metrics:** mrr, recall@k, precision@k
**Requires:** Query-document pairs

### Classification
Evaluates text classification (trains logistic regression on embeddings).

**Metrics:** accuracy, precision, recall, f1
**Requires:** Texts with labels

### Clustering
Evaluates clustering quality.

**Metrics:** silhouette, davies_bouldin, calinski_harabasz
**Requires:** Texts (labels optional)

### Temporal
Evaluates temporal fact retrieval and ordering.

**Metrics:**
- `within_group_temporal_order_correlation`: Temporal ordering within fact groups
- `within_group_nearest_fact_mrr`: Finding temporally adjacent facts
- `cross_group_mrr`: Discriminating between fact groups
- `cross_group_purity@k`: Group purity in top-k results
- `temporal_drift_correlation`: Similarity decay over time

**Requires:** JSONL with timestamp and group_id in metadata

**See:** `TEMPORAL_METRICS_GUIDE.md` for metric interpretations

## Results Files

After running an experiment:

```
results/my_experiment/
├── results.json          # Full metrics (all data)
├── results.csv           # Tabular metrics (numeric only)
├── report.txt            # Human-readable summary
├── comparison_*.csv      # Per-task metric comparisons
└── examples/             # Temporal examples (if temporal task)
    ├── README.md
    ├── model_dataset_examples.txt   # Human-readable
    └── model_dataset_examples.json  # Machine-readable
```

## CLI Reference

```bash
# Run experiment
python cli.py run <config.yaml> [--output-dir DIR] [--verbose]

# Validate config
python cli.py validate <config.yaml>

# Create template
python cli.py init <output.yaml>

# List models
python cli.py list-models

# List datasets
python cli.py list-datasets
```

## Available Models

| Model | Dimensions | Size | Speed |
|-------|------------|------|-------|
| all-MiniLM-L6-v2 | 384 | 80MB | Fast |
| all-MiniLM-L12-v2 | 384 | 120MB | Fast |
| all-mpnet-base-v2 | 768 | 420MB | Slower, higher quality |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 970MB | Multilingual |

See [SBERT Models](https://www.sbert.net/docs/pretrained_models.html) for more.

## Dataset Formats

**CSV:**
```csv
text,label
"Sample text",0
```

**JSONL:**
```jsonl
{"text": "Sample text", "label": 0}
```

**Temporal JSONL:**
```jsonl
{"instruction": "Fact text", "metadata": {"timestamp": "2025-01-01T00:00:00", "group_id": "group_001"}}
```

**Pairs (similarity/retrieval):**
```csv
text1,text2,label
"Query","Document",0.8
```

## Troubleshooting

**Out of memory:**
```yaml
batch_size: 16  # Reduce in config
```

**Slow performance:**
```yaml
device: cuda     # Enable GPU
batch_size: 64   # Increase batch size
```

**Import errors:**
```bash
pip install -e .  # Reinstall
```

**Temporal dataset not loading groups:**
```yaml
# Ensure nested paths are correct
timestamp_column: metadata.timestamp
group_id_column: metadata.group_id  # Optional, defaults to this
```

## Testing

```bash
# Run tests
pytest test_basic.py -v

# With coverage
pytest test_basic.py --cov=src --cov-report=html
```

## License

MIT License
