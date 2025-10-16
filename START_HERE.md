# 🚀 START HERE - Embedding Lab Navigation

## Welcome!

You have a **complete, production-ready Python application** for evaluating hierarchical dual-layer embedding models. This guide will help you navigate the project.

## 📋 Essential Reading Order

### 1. First: Read This File
You're doing it! ✓

### 2. Quick Overview
→ **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** (5 min read)
- What you have
- Key features
- Quick statistics

### 3. Getting Started
→ **[GETTING_STARTED.md](GETTING_STARTED.md)** (15 min read + 30 min practice)
- Installation steps
- Your first experiment
- Common use cases

### 4. Main Documentation
→ **[README.md](README.md)** (20 min read)
- Complete feature documentation
- Configuration reference
- CLI commands
- Examples

### 5. System Architecture (Optional)
→ **[ARCHITECTURE.md](ARCHITECTURE.md)** (30 min read)
- System design
- Component details
- Extension points

## ⚡ Quick Actions

### I want to...

#### ...install and verify it works
```bash
cd embedding-lab
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python verify_installation.py
```

#### ...see a quick demo
```bash
python demo.py
```

#### ...run my first experiment
```bash
python cli.py init config/experiments/my_experiment.yaml
# Edit the file, then:
python cli.py run config/experiments/my_experiment.yaml
```

#### ...understand the code structure
Look at [ARCHITECTURE.md](ARCHITECTURE.md), then browse `src/` directory

#### ...see example configurations
Look in `config/experiments/` directory

#### ...run tests
```bash
pytest tests/ -v
```

## 📁 Key Files & Directories

### Must Read
| File | Purpose | Time |
|------|---------|------|
| PROJECT_SUMMARY.md | Overview | 5 min |
| GETTING_STARTED.md | Tutorial | 15 min |
| README.md | Full docs | 20 min |

### Core Application
| Directory | Purpose |
|-----------|---------|
| src/embeddings/ | Embedding models (single & hierarchical) |
| src/datasets/ | Dataset loaders |
| src/evaluation/ | Task evaluators |
| src/experiments/ | Experiment orchestration |
| src/utils/ | Configuration & utilities |

### User Interface
| File | Purpose |
|------|---------|
| cli.py | Command-line interface |
| demo.py | Demonstration script |

### Configuration
| Directory/File | Purpose |
|----------------|---------|
| config/experiments/ | Example configurations |
| config/experiments/example_experiment.yaml | Complete example |

### Testing & Verification
| File | Purpose |
|------|---------|
| verify_installation.py | Check installation |
| tests/ | Test suite |

### Documentation
| File | Purpose |
|------|---------|
| ARCHITECTURE.md | System design |
| CHANGELOG.md | Version history |
| LICENSE | MIT license |

## 🎯 Common Use Cases

### Use Case 1: Compare Dual vs Single Embeddings
1. Look at `config/experiments/example_experiment.yaml`
2. Run: `python cli.py run config/experiments/example_experiment.yaml`
3. Check results in `results/` directory

### Use Case 2: Test on Custom Dataset
1. Create dataset CSV (see `data/sample_dataset.csv` for format)
2. Look at `config/experiments/custom_dataset_example.yaml`
3. Adapt for your data
4. Run experiment

### Use Case 3: Add New Model
1. Read `src/embeddings/base.py` for interface
2. Look at `src/embeddings/single.py` as example
3. Create your model class
4. Add to configuration

## 🔍 Code Navigation

### Want to understand...

#### How hierarchical embeddings work?
→ `src/embeddings/hierarchical.py`
- See `encode()` method
- See `_combine_embeddings()` method

#### How experiments run?
→ `src/experiments/runner.py`
- See `run()` method
- Follow the flow

#### How evaluation works?
→ `src/evaluation/`
- Look at any evaluator (e.g., `similarity.py`)
- See `evaluate()` method

#### How to load datasets?
→ `src/datasets/`
- `benchmarks.py` for standard datasets
- `custom.py` for custom datasets

## 📊 Project Statistics

- **40+ files**
- **5,000+ lines of code**
- **100+ functions/methods**
- **20+ classes**
- **4 evaluation tasks**
- **2 embedding types**
- **Multiple benchmarks**
- **Complete documentation**

## 🚦 Quick Start Path

### Beginner Path (30 minutes)
1. Read PROJECT_SUMMARY.md → 5 min
2. Read GETTING_STARTED.md → 15 min
3. Run verify_installation.py → 2 min
4. Run demo.py → 3 min
5. Run example experiment → 5 min

### Advanced Path (1 hour)
1. Read all documentation → 40 min
2. Review code structure → 10 min
3. Run tests → 5 min
4. Create custom experiment → 5 min

### Developer Path (2 hours)
1. Read ARCHITECTURE.md → 30 min
2. Review all source code → 60 min
3. Run and modify tests → 20 min
4. Extend with custom component → 10 min

## 🔧 Common Commands

```bash
# Installation
make install

# Verification
python verify_installation.py

# Demo
python demo.py

# Create config
python cli.py init my_config.yaml

# Run experiment
python cli.py run my_config.yaml

# List models
python cli.py list-models

# List datasets
python cli.py list-datasets

# Run tests
pytest tests/

# Clean
make clean
```

## 💡 Tips

1. **Start Simple**: Begin with the example configuration
2. **Use Verbose**: Add `--verbose` flag to see detailed logs
3. **Check MLflow**: View experiments with `mlflow ui`
4. **Read Configs**: Examples in `config/experiments/`
5. **Test First**: Run verification before experiments

## 🆘 Troubleshooting

### Issue: Import errors
**Solution**: `pip install -e .`

### Issue: Out of memory
**Solution**: Reduce `batch_size` in config

### Issue: Slow performance
**Solution**: Set `device: cuda` for GPU

### Issue: Need help
**Check**:
- README.md troubleshooting section
- Example configurations
- Test files for usage patterns

## 📚 Learning Resources

### In This Project
- Code comments and docstrings
- Test files show usage
- Demo script shows examples
- Multiple example configs

### External
- [Sentence Transformers Docs](https://www.sbert.net/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 🎓 Next Steps

After reading the essential files:

1. ✅ Verify installation works
2. ✅ Run demo script
3. ✅ Run example experiment
4. ✅ Create your own experiment
5. ✅ Explore the code
6. ✅ Extend with custom components

## 📝 Remember

- This is **production-ready** code
- Everything is **fully implemented**
- Code is **well-documented**
- System is **easily extensible**
- You have **complete control**

## 🎉 You're Ready!

Everything you need is here. Start with GETTING_STARTED.md and enjoy experimenting!

---

**Questions?** Check the documentation files listed above.
**Issues?** Review troubleshooting sections.
**Ready?** Run `python verify_installation.py` to start!
