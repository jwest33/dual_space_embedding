"""Command-line interface for embedding lab."""
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments import ExperimentRunner
from utils import load_config, save_config, ExperimentConfig, ModelConfig, DatasetConfig


console = Console()


# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Embedding Lab: Hierarchical dual-layer embedding experiment platform.
    
    A production-ready system for evaluating and comparing embedding models
    across multiple tasks: similarity, retrieval, classification, and clustering.
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Override output directory from config"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def run(config_path: str, output_dir: str, verbose: bool):
    """
    Run experiments from a configuration file.
    
    Example:
        embedding-lab run config/experiments/my_experiment.yaml
    """
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="DEBUG"
        )
    
    console.print(f"[bold blue]Loading configuration:[/bold blue] {config_path}")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override output dir if specified
        if output_dir:
            config.output_dir = output_dir
        
        # Display configuration
        _display_config(config)
        
        # Run experiment
        runner = ExperimentRunner(config)
        metrics_tracker = runner.run()
        
        # Display results summary
        _display_results(metrics_tracker)
        
        console.print(f"\n[bold green]✓ Experiment complete![/bold green]")
        console.print(f"Results saved to: [cyan]{config.output_dir}[/cyan]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Experiment failed")
        sys.exit(1)


@cli.command()
@click.argument("output_path", type=click.Path())
@click.option("--name", "-n", default="my_experiment", help="Experiment name")
@click.option("--description", "-d", default="", help="Experiment description")
def init(output_path: str, name: str, description: str):
    """
    Initialize a new experiment configuration file.
    
    Example:
        embedding-lab init config/experiments/new_experiment.yaml
    """
    console.print(f"[bold blue]Creating configuration:[/bold blue] {output_path}")
    
    # Create default configuration
    config = ExperimentConfig(
        name=name,
        description=description,
        models=[
            ModelConfig(
                type="single",
                name="baseline",
                model_name="all-MiniLM-L6-v2"
            ),
            ModelConfig(
                type="hierarchical",
                name="hierarchical",
                coarse_model="all-MiniLM-L6-v2",
                fine_model="all-mpnet-base-v2",
                combination_method="concat"
            ),
        ],
        datasets=[
            DatasetConfig(
                type="benchmark",
                name="sts-b",
                split="test"
            ),
        ],
        tasks=["similarity", "retrieval", "classification", "clustering"],
    )
    
    # Save configuration
    save_config(config, output_path)
    
    console.print(f"[bold green]✓ Configuration created![/bold green]")
    console.print(f"Edit the file and run: [cyan]embedding-lab run {output_path}[/cyan]")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """
    Validate a configuration file.
    
    Example:
        embedding-lab validate config/experiments/my_experiment.yaml
    """
    console.print(f"[bold blue]Validating configuration:[/bold blue] {config_path}")
    
    try:
        config = load_config(config_path)
        _display_config(config)
        console.print(f"[bold green]✓ Configuration is valid![/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Invalid configuration:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def list_models():
    """List available pre-configured models."""
    console.print("[bold blue]Available Models:[/bold blue]\n")
    
    models = [
        ("all-MiniLM-L6-v2", "Fast, lightweight (80MB)", "384"),
        ("all-mpnet-base-v2", "Good quality (420MB)", "768"),
        ("all-MiniLM-L12-v2", "Balanced (120MB)", "384"),
        ("paraphrase-multilingual-mpnet-base-v2", "Multilingual (970MB)", "768"),
    ]
    
    table = Table(title="Sentence-Transformer Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Dimensions", style="green")
    
    for model_name, desc, dim in models:
        table.add_row(model_name, desc, dim)
    
    console.print(table)
    console.print("\nFor more models, visit: https://www.sbert.net/docs/pretrained_models.html")


@cli.command()
def list_datasets():
    """List available benchmark datasets."""
    console.print("[bold blue]Available Benchmark Datasets:[/bold blue]\n")
    
    datasets = [
        ("sts-b", "Semantic Textual Similarity", "Similarity"),
        ("msmarco", "MS MARCO Passage Ranking", "Retrieval"),
        ("ag-news", "AG News Classification", "Classification"),
        ("trec", "TREC Question Classification", "Classification"),
    ]
    
    table = Table(title="Benchmark Datasets")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Task", style="green")
    
    for name, desc, task in datasets:
        table.add_row(name, desc, task)
    
    console.print(table)


def _display_config(config: ExperimentConfig):
    """Display configuration in a nice format."""
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Name: [cyan]{config.name}[/cyan]")
    console.print(f"  Description: {config.description}")
    console.print(f"  Models: {len(config.models)}")
    for model in config.models:
        console.print(f"    - {model.name} ({model.type})")
    console.print(f"  Datasets: {len(config.datasets)}")
    for dataset in config.datasets:
        console.print(f"    - {dataset.name} ({dataset.type})")
    console.print(f"  Tasks: {', '.join(config.tasks)}")
    console.print()


def _display_results(metrics_tracker):
    """Display results summary in a nice format."""
    summary = metrics_tracker.get_summary()
    
    if summary.empty:
        return
    
    console.print("\n[bold]Results Summary:[/bold]\n")
    
    # Group by task and show key metrics
    for task in summary["task"].unique():
        task_results = summary[summary["task"] == task]
        
        table = Table(title=f"{task.capitalize()} Task")
        table.add_column("Model", style="cyan")
        table.add_column("Dataset", style="white")
        
        # Add metric columns dynamically
        metric_cols = [col for col in task_results.columns 
                      if col not in ["model", "dataset", "task", "metadata"]]
        for col in metric_cols[:5]:  # Show top 5 metrics
            table.add_column(col, style="green")
        
        for _, row in task_results.iterrows():
            values = [row["model"], row["dataset"]]
            values.extend([f"{row[col]:.4f}" for col in metric_cols[:5]])
            table.add_row(*values)
        
        console.print(table)
        console.print()


if __name__ == "__main__":
    cli()
