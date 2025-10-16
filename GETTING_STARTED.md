# Getting Started with Embedding Lab

This guide will help you set up and run your first experiment comparing hierarchical dual-layer embeddings with single embeddings.

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### 2. Install Dependencies

```bash
# Navigate to project directory
cd embedding-lab

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 3. Verify Installation

```bash
# Check CLI is working
python cli.py --help

# List available models
python cli.py list-models

# List available datasets
python cli.py list-datasets
```

## Your First Experiment

### Step 1: Create Configuration

Initialize a new experiment configuration:

```bash
python cli.py init config/experiments/my_first_experiment.yaml \
    --name "my_first_experiment" \
    --description "Comparing single vs hierarchical embeddings"
```

This creates a template configuration file.

### Step 2: Edit Configuration

Open `config/experiments/my_first_experiment.yaml` and customize:

```yaml
name: my_first_experiment
description: Comparing single vs hierarchical embeddings on STS-B

models:
  # Baseline: Single embedding model
  - type: single
    name: single_baseline
    model_name: all-MiniLM-L6-v2
    normalize: true

  # Test: Hierarchical dual-layer model
  - type: hierarchical
    name: hierarchical_concat
    coarse_model: all-MiniLM-L6-v2
    fine_model: all-mpnet-base-v2
    combination_method: concat
    normalize: true

datasets:
  # Semantic Textual Similarity Benchmark
  - type: benchmark
    name: sts-b
    split: test

tasks:
  - similarity  # Semantic similarity evaluation

batch_size: 32
output_dir: results/my_first_experiment
```

### Step 3: Validate Configuration

```bash
python cli.py validate config/experiments/my_first_experiment.yaml
```

### Step 4: Run Experiment

```bash
python cli.py run config/experiments/my_first_experiment.yaml --verbose
```

The experiment will:
1. Download models (first run only)
2. Load datasets
3. Generate embeddings
4. Evaluate on tasks
5. Save results

### Step 5: View Results

Results are saved in `results/my_first_experiment/`:

```bash
# View JSON results
cat results/my_first_experiment/results.json

# View CSV results
cat results/my_first_experiment/results.csv

# View report
cat results/my_first_experiment/report.txt

# View metric comparisons
cat results/my_first_experiment/comparison_spearman.csv
```

Example results:
```
Model                    Spearman    Pearson    MAE
single_baseline          0.8234      0.8156     0.1234
hierarchical_concat      0.8567      0.8489     0.1089
```

## Running the Demo

Try the quick demo script:

```bash
python demo.py
```

This demonstrates:
- Creating embedding models
- Loading datasets
- Running evaluations

## Advanced Configuration

### Multiple Tasks

Test on all tasks:

```yaml
tasks:
  - similarity      # Semantic similarity
  - retrieval       # Information retrieval
  - classification  # Text classification
  - clustering      # Text clustering
```

### Multiple Datasets

```yaml
datasets:
  - type: benchmark
    name: sts-b
    split: test
  
  - type: benchmark
    name: ag-news
    split: test
    num_samples: 2000  # Limit samples for speed
  
  - type: benchmark
    name: trec
    split: test
```

### Custom Dataset

Create `data/my_data.csv`:
```csv
text,label
"This is great!",positive
"This is terrible!",negative
```

Add to configuration:
```yaml
datasets:
  - type: custom
    name: my_data
    file_path: data/my_data.csv
    text1_column: text
    label_column: label
```

### GPU Acceleration

Enable GPU:
```yaml
models:
  - type: single
    name: gpu_model
    model_name: all-MiniLM-L6-v2
    device: cuda  # Use GPU
```

### Different Combination Methods

Test weighted combination:
```yaml
- type: hierarchical
  name: hierarchical_weighted
  coarse_model: all-MiniLM-L6-v2
  fine_model: all-MiniLM-L6-v2  # Must be same dim
  combination_method: weighted_sum
  coarse_weight: 0.3  # 30% coarse
  fine_weight: 0.7    # 70% fine
```

## Experiment Tracking

View experiments in MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns

# Open browser to http://localhost:5000
```

## Common Use Cases

### 1. Baseline Comparison
```bash
# Create config comparing multiple single models
python cli.py init config/experiments/baseline_comparison.yaml
# Edit to include multiple single models
python cli.py run config/experiments/baseline_comparison.yaml
```

### 2. Hierarchical Architecture Search
```bash
# Test different coarse-fine combinations
# Edit config with multiple hierarchical models
python cli.py run config/experiments/architecture_search.yaml
```

### 3. Task-Specific Optimization
```bash
# Test models on specific task
# Set tasks: [classification] in config
python cli.py run config/experiments/classification_only.yaml
```

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
batch_size: 16  # or 8

# Or use smaller models
models:
  - type: single
    model_name: all-MiniLM-L6-v2  # Smaller model
```

### Slow Performance
```bash
# Enable GPU
device: cuda

# Or increase batch size
batch_size: 64
```

### Model Download Issues
```bash
# Manually download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Import Errors
```bash
# Reinstall package
pip install -e .

# Or check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Next Steps

1. **Read the full README**: `README.md`
2. **Review example configs**: `config/experiments/`
3. **Try different models**: `python cli.py list-models`
4. **Add custom datasets**: See custom dataset section
5. **Run comprehensive experiments**: Test all tasks

## Tips

- Start with small datasets to test configurations
- Use `--verbose` flag for debugging
- Check MLflow for detailed metrics
- Compare results in CSV files
- Save successful configs for reproducibility

## Getting Help

- Review documentation in README.md
- Check example configurations
- Run `python cli.py --help`
- Review test files in `tests/`

Happy experimenting!
