# Changelog

All notable changes to Embedding Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-16

### Added
- Initial release of Embedding Lab
- Hierarchical dual-layer embedding implementation
- Support for sentence-transformer models
- Four evaluation tasks:
  - Semantic similarity (STS-B)
  - Information retrieval (MS MARCO)
  - Text classification (AG News, TREC)
  - Text clustering
- Benchmark dataset loaders
- Custom dataset support (CSV, TSV, JSON, JSONL)
- CLI interface with commands:
  - run: Execute experiments
  - init: Create configurations
  - validate: Check configurations
  - list-models: Show available models
  - list-datasets: Show available datasets
- Configuration system (YAML-based)
- MLflow experiment tracking
- Comprehensive metrics tracking and reporting
- Batch processing support
- GPU acceleration support
- Three combination methods for hierarchical embeddings:
  - Concatenation
  - Weighted sum
  - Learned (placeholder)
- Complete test suite
- Documentation:
  - README.md
  - GETTING_STARTED.md
  - ARCHITECTURE.md
- Demo script
- Verification script
- Example configurations

### Features
- Production-ready codebase
- Modular architecture
- Extensible design
- Type hints throughout
- Comprehensive logging
- Error handling
- Results export (JSON, CSV, TXT)
- Comparison tables
- Experiment reports

### Supported Models
- all-MiniLM-L6-v2
- all-mpnet-base-v2
- all-MiniLM-L12-v2
- paraphrase-multilingual-mpnet-base-v2
- Any sentence-transformer model

### Supported Datasets
- STS-B (Semantic Similarity)
- MS MARCO (Retrieval)
- AG News (Classification)
- TREC (Classification)
- Custom datasets (CSV, JSON, JSONL, TSV)

### Dependencies
- sentence-transformers >= 2.2.2
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- datasets >= 2.14.0
- click >= 8.1.0
- mlflow >= 2.7.0
- And more (see requirements.txt)

## [Future] - Planned Features

### To Be Added
- [ ] Learned combination methods with trainable parameters
- [ ] Cross-encoder support
- [ ] Additional benchmarks (BEIR, MTEB)
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] REST API for embeddings
- [ ] Web UI for experiment management
- [ ] Visualization tools
- [ ] AutoML for hyperparameter tuning
- [ ] Docker support
- [ ] Cloud deployment guides
- [ ] More combination strategies
- [ ] Ensemble methods
- [ ] Model distillation
- [ ] Quantization support
- [ ] ONNX export
- [ ] Streaming evaluation for large datasets

### Improvements
- [ ] Performance optimizations
- [ ] Memory efficiency improvements
- [ ] Better error messages
- [ ] More examples
- [ ] Video tutorials
- [ ] Interactive notebooks
- [ ] Community templates

## Notes

This is the first production-ready release. The system is designed to be:
- **Complete**: Full functionality for comparing embeddings
- **Extensible**: Easy to add new models, datasets, and tasks
- **User-friendly**: Simple CLI and configuration
- **Production-ready**: Proper error handling, logging, and tracking
- **Well-documented**: Comprehensive documentation and examples

For issues, feature requests, or contributions, please open an issue on GitHub.
