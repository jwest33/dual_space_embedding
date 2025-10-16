# Makefile for Embedding Lab

.PHONY: help install test verify demo clean run lint format

help:
	@echo "Embedding Lab - Available Commands:"
	@echo ""
	@echo "  make install      - Install dependencies and package"
	@echo "  make verify       - Verify installation"
	@echo "  make test         - Run tests"
	@echo "  make demo         - Run demo script"
	@echo "  make run          - Run example experiment"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Installation complete"

verify:
	python verify_installation.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

demo:
	python demo.py

run:
	python cli.py run config/experiments/example_experiment.yaml

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	pylint src/ --disable=C0111,R0903,R0913

format:
	black src/ tests/ cli.py demo.py --line-length=100
	isort src/ tests/ cli.py demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf results/ mlruns/
	@echo "✓ Cleaned generated files"

.DEFAULT_GOAL := help
