#!/usr/bin/env python3
"""
Visualization script for benchmark results.

Loads benchmark results from src/outputs/ and creates visualizations
comparing forward and backward pass times across different model configurations.

Example usage:
    # Visualize all results from default output directory
    python src/visualize_benchmark.py

    # Visualize from custom directory
    python src/visualize_benchmark.py --input_dir results/

    # Save plot to file instead of displaying
    python src/visualize_benchmark.py --output plot.png

    # Filter by specific configs
    python src/visualize_benchmark.py --configs small medium

    # Show individual run times (not just mean)
    python src/visualize_benchmark.py --show-individual
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
import sys

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Default paths
DEFAULT_INPUT_DIR = Path(__file__).parent / "outputs"


def load_results_from_csv(csv_path: Path) -> list[dict]:
    """Load benchmark results from CSV summary file."""
    results = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            result = {
                "run_id": row["run_id"],
                "timestamp": row["timestamp"],
                "config_name": row["config_name"],
                "d_model": int(row["d_model"]),
                "num_layers": int(row["num_layers"]),
                "num_heads": int(row["num_heads"]),
                "d_ff": int(row["d_ff"]),
                "vocab_size": int(row["vocab_size"]),
                "context_length": int(row["context_length"]),
                "batch_size": int(row["batch_size"]),
                "device": row["device"],
                "precision": row["precision"],
                "warmup_steps": int(row["warmup_steps"]),
                "num_steps": int(row["num_steps"]),
                "mode": row["mode"],
            }

            # Add forward results if present
            if "forward_mean_ms" in row and row["forward_mean_ms"]:
                result["forward_mean_ms"] = float(row["forward_mean_ms"])
                result["forward_std_ms"] = float(row["forward_std_ms"])
                result["forward_min_ms"] = float(row["forward_min_ms"])
                result["forward_max_ms"] = float(row["forward_max_ms"])

            # Add backward results if present
            if "backward_mean_ms" in row and row["backward_mean_ms"]:
                result["backward_mean_ms"] = float(row["backward_mean_ms"])
                result["backward_std_ms"] = float(row["backward_std_ms"])
                result["backward_min_ms"] = float(row["backward_min_ms"])
                result["backward_max_ms"] = float(row["backward_max_ms"])

            results.append(result)

    return results


def load_results_from_json_files(input_dir: Path) -> list[dict]:
    """Load benchmark results from individual JSON files."""
    results = []
    json_files = sorted(input_dir.glob("benchmark_*.json"))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        config = data["config"]
        bench_results = data["results"]

        result = {
            "run_id": data["run_id"],
            "timestamp": data["timestamp"],
            "config_name": config.get("config_name", "custom"),
            "d_model": config["d_model"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "d_ff": config["d_ff"],
            "vocab_size": config["vocab_size"],
            "context_length": config["context_length"],
            "batch_size": config["batch_size"],
            "device": config["device"],
            "precision": config["precision"],
            "warmup_steps": config["warmup_steps"],
            "num_steps": config["num_steps"],
            "mode": config["mode"],
        }

        # Add forward results
        if "forward" in bench_results:
            fwd = bench_results["forward"]
            result["forward_mean_ms"] = fwd["mean_time_ms"]
            result["forward_std_ms"] = fwd["std_time_ms"]
            result["forward_min_ms"] = fwd["min_time_ms"]
            result["forward_max_ms"] = fwd["max_time_ms"]
            result["forward_times"] = fwd.get("times", [])

        # Add backward results
        if "backward" in bench_results:
            bwd = bench_results["backward"]
            result["backward_mean_ms"] = bwd["mean_time_ms"]
            result["backward_std_ms"] = bwd["std_time_ms"]
            result["backward_min_ms"] = bwd["min_time_ms"]
            result["backward_max_ms"] = bwd["max_time_ms"]
            result["backward_times"] = bwd.get("times", [])

        results.append(result)

    return results


def load_results(input_dir: Path) -> list[dict]:
    """Load benchmark results, preferring JSON files for more detail."""
    json_files = list(input_dir.glob("benchmark_*.json"))
    csv_path = input_dir / "benchmark_summary.csv"

    if json_files:
        print(f"Loading {len(json_files)} JSON result files...")
        return load_results_from_json_files(input_dir)
    elif csv_path.exists():
        print(f"Loading results from CSV summary...")
        return load_results_from_csv(csv_path)
    else:
        print(f"No benchmark results found in {input_dir}")
        return []


def get_latest_by_config(results: list[dict]) -> dict[str, dict]:
    """Get the latest result for each config name."""
    latest = {}
    for result in results:
        config_name = result["config_name"]
        if config_name not in latest:
            latest[config_name] = result
        else:
            # Compare timestamps
            current_time = datetime.fromisoformat(result["timestamp"])
            existing_time = datetime.fromisoformat(latest[config_name]["timestamp"])
            if current_time > existing_time:
                latest[config_name] = result
    return latest


def print_results_table(results: dict[str, dict]):
    """Print a formatted table of benchmark results."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Config':<10} {'Params':<10} {'Forward (ms)':<22} {'Backward (ms)':<22} {'Device':<8}")
    print("-" * 90)

    # Sort by model size (d_model * num_layers as proxy)
    sorted_configs = sorted(
        results.items(),
        key=lambda x: x[1]["d_model"] * x[1]["num_layers"]
    )

    for config_name, result in sorted_configs:
        d = result["d_model"]
        L = result["num_layers"]
        V = result["vocab_size"]
        params_m = (12 * L * d * d + V * d) / 1e6

        fwd_str = "N/A"
        bwd_str = "N/A"

        if "forward_mean_ms" in result:
            fwd_str = f"{result['forward_mean_ms']:.2f} ± {result['forward_std_ms']:.2f}"

        if "backward_mean_ms" in result:
            bwd_str = f"{result['backward_mean_ms']:.2f} ± {result['backward_std_ms']:.2f}"

        print(f"{config_name:<10} {params_m:>7.1f}M   {fwd_str:<22} {bwd_str:<22} {result['device']:<8}")

    print("=" * 90)


def plot_results(
    results: dict[str, dict],
    output_path: Path | None = None,
    show_individual: bool = False,
    title: str = "Transformer Benchmark Results"
):
    """Create visualization of benchmark results."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Falling back to text-only output.")
        return

    # Sort configs by model size
    sorted_configs = sorted(
        results.items(),
        key=lambda x: x[1]["d_model"] * x[1]["num_layers"]
    )

    config_names = [name for name, _ in sorted_configs]

    # Extract data
    forward_means = []
    forward_stds = []
    backward_means = []
    backward_stds = []
    param_counts = []

    for _, result in sorted_configs:
        d = result["d_model"]
        L = result["num_layers"]
        V = result["vocab_size"]
        params_m = (12 * L * d * d + V * d) / 1e6
        param_counts.append(params_m)

        forward_means.append(result.get("forward_mean_ms", 0))
        forward_stds.append(result.get("forward_std_ms", 0))
        backward_means.append(result.get("backward_mean_ms", 0))
        backward_stds.append(result.get("backward_std_ms", 0))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(config_names))
    width = 0.35

    # Plot 1: Forward and Backward times side by side
    ax1 = axes[0]
    bars1 = ax1.bar([i - width/2 for i in x], forward_means, width,
                    yerr=forward_stds, label='Forward', capsize=5, color='steelblue')
    bars2 = ax1.bar([i + width/2 for i in x], backward_means, width,
                    yerr=backward_stds, label='Forward + Backward', capsize=5, color='darkorange')

    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward vs Forward+Backward Pass Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{name}\n({param_counts[i]:.0f}M)" for i, name in enumerate(config_names)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars1, forward_means, forward_stds):
        if mean > 0:
            ax1.annotate(f'{mean:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    for bar, mean, std in zip(bars2, backward_means, backward_stds):
        if mean > 0:
            ax1.annotate(f'{mean:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Plot 2: Scaling with model size
    ax2 = axes[1]
    ax2.errorbar(param_counts, forward_means, yerr=forward_stds,
                 fmt='o-', label='Forward', capsize=5, color='steelblue', markersize=8)
    ax2.errorbar(param_counts, backward_means, yerr=backward_stds,
                 fmt='s-', label='Forward + Backward', capsize=5, color='darkorange', markersize=8)

    ax2.set_xlabel('Model Parameters (M)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Time vs Model Size')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add config labels
    for i, (params, fwd, bwd, name) in enumerate(zip(param_counts, forward_means, backward_means, config_names)):
        ax2.annotate(name, (params, max(fwd, bwd)), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def plot_variability(
    results: dict[str, dict],
    output_path: Path | None = None,
):
    """Create visualization showing timing variability."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed.")
        return

    # Sort configs by model size
    sorted_configs = sorted(
        results.items(),
        key=lambda x: x[1]["d_model"] * x[1]["num_layers"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot coefficient of variation (std/mean) for each config
    config_names = []
    forward_cv = []
    backward_cv = []

    for name, result in sorted_configs:
        config_names.append(name)

        if "forward_mean_ms" in result and result["forward_mean_ms"] > 0:
            forward_cv.append(result["forward_std_ms"] / result["forward_mean_ms"] * 100)
        else:
            forward_cv.append(0)

        if "backward_mean_ms" in result and result["backward_mean_ms"] > 0:
            backward_cv.append(result["backward_std_ms"] / result["backward_mean_ms"] * 100)
        else:
            backward_cv.append(0)

    x = range(len(config_names))
    width = 0.35

    ax1 = axes[0]
    ax1.bar([i - width/2 for i in x], forward_cv, width, label='Forward', color='steelblue')
    ax1.bar([i + width/2 for i in x], backward_cv, width, label='Forward + Backward', color='darkorange')
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Coefficient of Variation (%)')
    ax1.set_title('Timing Variability (lower is more consistent)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot min/max range
    ax2 = axes[1]
    forward_ranges = []
    backward_ranges = []

    for name, result in sorted_configs:
        if "forward_min_ms" in result:
            forward_ranges.append(result["forward_max_ms"] - result["forward_min_ms"])
        else:
            forward_ranges.append(0)

        if "backward_min_ms" in result:
            backward_ranges.append(result["backward_max_ms"] - result["backward_min_ms"])
        else:
            backward_ranges.append(0)

    ax2.bar([i - width/2 for i in x], forward_ranges, width, label='Forward', color='steelblue')
    ax2.bar([i + width/2 for i in x], backward_ranges, width, label='Forward + Backward', color='darkorange')
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Max - Min Time (ms)')
    ax2.set_title('Timing Range (Max - Min)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Benchmark Variability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        # Modify filename for variability plot
        stem = output_path.stem
        suffix = output_path.suffix
        var_path = output_path.parent / f"{stem}_variability{suffix}"
        plt.savefig(var_path, dpi=150, bbox_inches='tight')
        print(f"Variability plot saved to: {var_path}")
    else:
        plt.show()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing benchmark results"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for plot (e.g., plot.png). If not specified, displays interactively."
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific config names (e.g., --configs small medium)"
    )

    parser.add_argument(
        "--show-variability",
        action="store_true",
        help="Also show variability analysis plot"
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only print text table, no plots"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load results
    all_results = load_results(input_dir)

    if not all_results:
        print("No benchmark results found.")
        sys.exit(1)

    print(f"Loaded {len(all_results)} benchmark results")

    # Get latest result per config
    latest_by_config = get_latest_by_config(all_results)

    # Filter configs if specified
    if args.configs:
        latest_by_config = {
            k: v for k, v in latest_by_config.items()
            if k in args.configs
        }

    if not latest_by_config:
        print("No results match the specified filters.")
        sys.exit(1)

    # Print text table
    print_results_table(latest_by_config)

    # Create plots unless text-only
    if not args.text_only:
        output_path = Path(args.output) if args.output else None

        plot_results(
            latest_by_config,
            output_path=output_path,
        )

        if args.show_variability:
            plot_variability(
                latest_by_config,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
