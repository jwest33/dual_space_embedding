from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="embedding-lab",
    version="1.0.0",
    author="Embedding Lab",
    description="Production-ready hierarchical dual-layer embedding experiment platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "nltk>=3.8.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "mlflow>=2.7.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.7.0",
        "loguru>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "embedding-lab=cli:cli",
        ],
    },
)
