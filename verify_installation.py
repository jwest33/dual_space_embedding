#!/usr/bin/env python
"""Verification script to test installation and basic functionality."""
import sys
import subprocess
from pathlib import Path
import time


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_step(step_num, text):
    """Print formatted step."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 80)


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✓ Success")
            if result.stdout:
                print(f"Output:\n{result.stdout[:500]}")
            return True
        else:
            print("✗ Failed")
            if result.stderr:
                print(f"Error:\n{result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Timeout (5 minutes exceeded)")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def check_imports():
    """Check if all imports work."""
    print("Checking imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from embeddings import SingleEmbedder, HierarchicalEmbedder
        from datasets import get_benchmark_dataset, load_custom_dataset
        from evaluation import (
            SimilarityEvaluator,
            RetrievalEvaluator,
            ClassificationEvaluator,
            ClusteringEvaluator,
        )
        from experiments import ExperimentRunner
        from utils import load_config, ExperimentConfig
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_embeddings():
    """Test embedding models."""
    print("Testing embedding models...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from embeddings import SingleEmbedder, HierarchicalEmbedder
        
        # Test single embedder
        print("  Testing SingleEmbedder...")
        single = SingleEmbedder("all-MiniLM-L6-v2")
        embeddings = single.encode(["test"])
        assert embeddings.shape == (1, 384)
        print("    ✓ SingleEmbedder works")
        
        # Test hierarchical embedder
        print("  Testing HierarchicalEmbedder...")
        hierarchical = HierarchicalEmbedder(
            coarse_model="all-MiniLM-L6-v2",
            fine_model="all-MiniLM-L6-v2"
        )
        embeddings = hierarchical.encode(["test"])
        assert embeddings.shape == (1, 768)
        print("    ✓ HierarchicalEmbedder works")
        
        print("✓ Embedding tests passed")
        return True
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")
        return False


def main():
    """Run all verification steps."""
    print_header("EMBEDDING LAB - Installation Verification")
    
    results = []
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version >= (3, 8):
        print("✓ Python version OK (>= 3.8)")
        results.append(True)
    else:
        print("✗ Python version too old (need >= 3.8)")
        results.append(False)
    
    # Step 2: Check dependencies
    print_step(2, "Checking Dependencies")
    packages = [
        "sentence-transformers",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "click",
        "pyyaml",
        "loguru",
    ]
    
    all_installed = True
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("✓ All dependencies installed")
        results.append(True)
    else:
        print("✗ Some dependencies missing")
        print("Run: pip install -r requirements.txt")
        results.append(False)
    
    # Step 3: Check imports
    print_step(3, "Checking Project Imports")
    results.append(check_imports())
    
    # Step 4: Test CLI
    print_step(4, "Testing CLI")
    results.append(run_command(
        "python cli.py --help",
        "CLI help command"
    ))
    
    # Step 5: Test configuration
    print_step(5, "Testing Configuration System")
    if Path("config/experiments/example_experiment.yaml").exists():
        results.append(run_command(
            "python cli.py validate config/experiments/example_experiment.yaml",
            "Validate example configuration"
        ))
    else:
        print("✗ Example configuration not found")
        results.append(False)
    
    # Step 6: Test embeddings
    print_step(6, "Testing Embedding Models (this may take a minute...)")
    results.append(test_embeddings())
    
    # Step 7: Test demo
    print_step(7, "Testing Demo Script")
    if Path("demo.py").exists():
        results.append(run_command(
            "timeout 120 python demo.py",
            "Run demo script"
        ))
    else:
        print("✗ Demo script not found")
        results.append(False)
    
    # Summary
    print_header("Verification Summary")
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYour installation is ready!")
        print("\nNext steps:")
        print("1. Read GETTING_STARTED.md")
        print("2. python cli.py init config/experiments/my_experiment.yaml")
        print("3. python cli.py run config/experiments/my_experiment.yaml")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease fix the issues above and run this script again.")
        print("\nCommon fixes:")
        print("- pip install -r requirements.txt")
        print("- pip install -e .")
        print("- Check Python version >= 3.8")
        return 1


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    sys.exit(exit_code)
