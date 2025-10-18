"""Utility for formatting and saving temporal retrieval examples."""
import json
from pathlib import Path
from typing import Dict, List, Any


def format_within_group_example(example: Dict, index: int) -> str:
    """
    Format a within-group retrieval example as human-readable text.

    Args:
        example: Example dictionary from TemporalRetrievalEvaluator
        index: Example number

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Within-Group Example {index + 1}")
    lines.append(f"{'='*80}")
    lines.append(f"\nQuery (Group: {example['query_group']}, Position: {example['query_temporal_position']}):")
    lines.append(f"Timestamp: {example['query_timestamp']}")
    lines.append(f'"{example["query_text"]}"')
    lines.append(f"\nRetrieved Facts (by similarity):")

    for fact in example["retrieved_facts"]:
        same_marker = "" if fact["temporal_position"] else "[UNKNOWN]"
        lines.append(
            f"\n  {fact['rank']}. [Pos: {fact['temporal_position']}, Sim: {fact['similarity']:.3f}] {same_marker}"
        )
        lines.append(f"     Timestamp: {fact['timestamp']}")
        lines.append(f'     "{fact["text"]}"')

    corr = example.get("temporal_order_correlation")
    corr_str = f"{corr:.3f}" if corr is not None else "N/A"
    lines.append(f"\nTemporal Order Correlation: {corr_str}")
    lines.append(f"Nearest Fact Rank: {example['nearest_fact_rank']}")

    return "\n".join(lines)


def format_cross_group_example(example: Dict, index: int) -> str:
    """
    Format a cross-group retrieval example as human-readable text.

    Args:
        example: Example dictionary from TemporalRetrievalEvaluator
        index: Example number

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Cross-Group Example {index + 1}")
    lines.append(f"{'='*80}")
    lines.append(f"\nQuery (Group: {example['query_group']}):")
    lines.append(f'"{example["query_text"]}"')
    lines.append(f"\nTop 10 Retrieved Facts:")

    for fact in example["retrieved_facts"]:
        group_marker = "✓ SAME" if fact["is_same_group"] else f"✗ {fact['group']}"
        lines.append(
            f"\n  {fact['rank']}. [Sim: {fact['similarity']:.3f}] {group_marker}"
        )
        lines.append(f'     "{fact["text"]}"')

    lines.append(f"\nSame-group facts in top-5: {example['same_group_in_top_5']}/5")
    lines.append(f"Same-group facts in top-10: {example['same_group_in_top_10']}/10")
    lines.append(f"First same-group fact rank: {example['first_same_group_rank']}")

    return "\n".join(lines)


def save_examples(
    model_name: str,
    dataset_name: str,
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save temporal retrieval examples to files.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        metrics: Metrics dictionary containing examples
        output_dir: Output directory path
    """
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Extract examples
    within_examples = metrics.get("within_group_examples", [])
    cross_examples = metrics.get("cross_group_examples", [])

    if not within_examples and not cross_examples:
        return  # No examples to save

    # Save as JSON
    json_path = examples_dir / f"{model_name}_{dataset_name}_examples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "dataset": dataset_name,
            "within_group_examples": within_examples,
            "cross_group_examples": cross_examples,
        }, f, indent=2)

    # Save as human-readable text
    txt_path = examples_dir / f"{model_name}_{dataset_name}_examples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Temporal Retrieval Examples\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"{'='*80}\n")

        if within_examples:
            f.write(f"\n\n{'#'*80}\n")
            f.write(f"# WITHIN-GROUP RETRIEVAL EXAMPLES\n")
            f.write(f"{'#'*80}\n")
            for i, example in enumerate(within_examples):
                f.write(format_within_group_example(example, i))

        if cross_examples:
            f.write(f"\n\n{'#'*80}\n")
            f.write(f"# CROSS-GROUP RETRIEVAL EXAMPLES\n")
            f.write(f"{'#'*80}\n")
            for i, example in enumerate(cross_examples):
                f.write(format_cross_group_example(example, i))


def generate_examples_summary(output_dir: Path) -> str:
    """
    Generate a markdown summary of all examples.

    Args:
        output_dir: Output directory containing examples

    Returns:
        Summary text
    """
    examples_dir = output_dir / "examples"
    if not examples_dir.exists():
        return "No examples found."

    summary_lines = []
    summary_lines.append("# Temporal Retrieval Examples Summary\n")
    summary_lines.append("This directory contains example retrievals from each model.\n")
    summary_lines.append("\n## Files\n")

    for json_file in sorted(examples_dir.glob("*_examples.json")):
        txt_file = json_file.with_suffix(".txt")
        model_dataset = json_file.stem.replace("_examples", "")

        summary_lines.append(f"- **{model_dataset}**")
        summary_lines.append(f"  - JSON: `{json_file.name}`")
        summary_lines.append(f"  - Text: `{txt_file.name}`\n")

    summary_lines.append("\n## How to Use\n")
    summary_lines.append("- `.json` files: Machine-readable, can be loaded for analysis")
    summary_lines.append("- `.txt` files: Human-readable, easier to review examples\n")
    summary_lines.append("\n## Example Structure\n")
    summary_lines.append("**Within-Group Examples**: Show how well the model retrieves facts in temporal order within a group")
    summary_lines.append("**Cross-Group Examples**: Show how well the model discriminates between different fact groups\n")

    return "\n".join(summary_lines)
