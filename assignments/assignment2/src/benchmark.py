#!/usr/bin/env python3
"""
Performance benchmarking script for BasicsTransformerLM.

This script profiles forward and backward passes of the model with various configurations,
measuring speed and memory usage. It uses random weights and data for testing.

Results are saved to src/outputs/ by default for easy analysis and visualization.

Example usage:
    # Basic benchmark with default parameters (saves to src/outputs/)
    python src/benchmark.py

    # Benchmark with custom model size
    python src/benchmark.py --d_model 1024 --num_layers 12 --num_heads 16

    # Benchmark with mixed precision
    python src/benchmark.py --precision bf16

    # Benchmark with different context lengths and batch sizes
    python src/benchmark.py --context_length 2048 --batch_size 8

    # Run on CPU
    python src/benchmark.py --device cpu

    # Forward-only benchmark
    python src/benchmark.py --mode forward

    # Forward + backward benchmark
    python src/benchmark.py --mode both

    # Use predefined model configuration
    python src/benchmark.py --config small
    python src/benchmark.py --config medium
    python src/benchmark.py --config large
    python src/benchmark.py --config xl

    # Run without warmup (for analysis)
    python src/benchmark.py --warmup_steps 0

    # Disable auto-save
    python src/benchmark.py --no-save

    # Custom output directory
    python src/benchmark.py --output_dir results/

    # Run all predefined configs
    python src/benchmark.py --config all
"""

import argparse
import timeit
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import logging
import statistics

import torch
import torch.nn as nn
from tqdm import tqdm

# Add cs336-basics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cs336-basics"))

from cs336_basics.model import BasicsTransformerLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"

# Predefined model configurations (based on common Transformer sizes)
# These correspond to typical GPT-style model sizes
MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "medium": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    "large": {
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
    },
    "xl": {
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
    },
}


def generate_run_id(config_name: str | None, args) -> str:
    """Generate a unique run ID based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_name:
        return f"{config_name}_{timestamp}"
    else:
        return f"custom_d{args.d_model}_l{args.num_layers}_{timestamp}"


def save_results(
    results: dict,
    config: dict,
    output_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Save benchmark results in multiple formats.

    Returns:
        Dictionary mapping format names to file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Save detailed JSON results
    json_path = output_dir / f"benchmark_{run_id}.json"
    json_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    saved_files["json"] = json_path

    # Append to CSV summary file for easy comparison
    csv_path = output_dir / "benchmark_summary.csv"
    csv_exists = csv_path.exists()

    # Flatten results for CSV
    csv_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
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

    # Add forward pass results
    if "forward" in results:
        csv_row["forward_mean_ms"] = results["forward"]["mean_time_ms"]
        csv_row["forward_std_ms"] = results["forward"]["std_time_ms"]
        csv_row["forward_min_ms"] = results["forward"]["min_time_ms"]
        csv_row["forward_max_ms"] = results["forward"]["max_time_ms"]

    # Add backward pass results
    if "backward" in results:
        csv_row["backward_mean_ms"] = results["backward"]["mean_time_ms"]
        csv_row["backward_std_ms"] = results["backward"]["std_time_ms"]
        csv_row["backward_min_ms"] = results["backward"]["min_time_ms"]
        csv_row["backward_max_ms"] = results["backward"]["max_time_ms"]

    fieldnames = list(csv_row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_row)
    saved_files["csv"] = csv_path

    return saved_files


def print_summary_table(all_results: list[dict]):
    """Print a summary table of multiple benchmark runs."""
    if not all_results:
        return

    logger.info("\n" + "=" * 100)
    logger.info("BENCHMARK SUMMARY TABLE")
    logger.info("=" * 100)

    # Header
    header = f"{'Config':<10} {'Params':<12} {'Forward (ms)':<20} {'Backward (ms)':<20} {'Device':<8}"
    logger.info(header)
    logger.info("-" * 100)

    for run in all_results:
        config = run["config"]
        results = run["results"]
        config_name = config.get("config_name", "custom")

        # Calculate approximate params (rough estimate)
        d = config["d_model"]
        L = config["num_layers"]
        V = config["vocab_size"]
        params_m = (12 * L * d * d + V * d) / 1e6

        fwd_str = "N/A"
        bwd_str = "N/A"

        if "forward" in results:
            fwd = results["forward"]
            fwd_str = f"{fwd['mean_time_ms']:.1f} ± {fwd['std_time_ms']:.1f}"

        if "backward" in results:
            bwd = results["backward"]
            bwd_str = f"{bwd['mean_time_ms']:.1f} ± {bwd['std_time_ms']:.1f}"

        row = f"{config_name:<10} {params_m:>8.1f}M    {fwd_str:<20} {bwd_str:<20} {config['device']:<8}"
        logger.info(row)

    logger.info("=" * 100)


class ModelBenchmark:
    """Benchmark utility for profiling model performance."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        batch_size: int,
        device: str,
        precision: str,
        warmup_iters: int,
        benchmark_iters: int,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.precision = precision
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters

        # Set up device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Set up precision
        self.dtype = self._get_dtype(precision)

        # Create model with random weights
        logger.info("Initializing model...")
        self.model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        )

        self.model = self.model.to(self.device).to(self.dtype)

        # Log model info
        num_params = self.model.get_num_params(non_embedding=False)
        logger.info(f"Total parameters: {num_params / 1e6:.2f}M")
        logger.info(f"Non-embedding parameters: {self.model.get_num_params() / 1e6:.2f}M")

    def _get_dtype(self, precision: str) -> torch.dtype:
        """Convert precision string to torch dtype."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported precision: {precision}. Choose from {list(dtype_map.keys())}")
        return dtype_map[precision]

    def generate_random_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random input tokens and targets."""
        # Random token IDs
        input_ids = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.context_length),
            device=self.device
        )

        # Random targets (shifted by 1 for language modeling)
        targets = torch.randint(
            0, self.vocab_size,
            (self.batch_size, self.context_length),
            device=self.device
        )

        return input_ids, targets

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        if self.device == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        else:
            return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}

    def reset_memory_stats(self):
        """Reset memory statistics."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def synchronize(self):
        """Synchronize device for accurate timing."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    def benchmark_forward(self) -> dict:
        """Benchmark forward pass only."""
        logger.info("Benchmarking forward pass...")

        # Reset memory stats
        self.reset_memory_stats()

        # Warmup
        if self.warmup_iters > 0:
            with tqdm(total=self.warmup_iters, desc="Warmup (forward)", unit="iter",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for _ in range(self.warmup_iters):
                    input_ids, _ = self.generate_random_batch()
                    with torch.no_grad():
                        _ = self.model(input_ids)
                    self.synchronize()
                    pbar.update(1)
        else:
            logger.info("Skipping warmup (warmup_steps=0)")

        self.synchronize()

        # Benchmark
        times = []
        with tqdm(total=self.benchmark_iters, desc="Benchmark (forward)", unit="iter") as pbar:
            for i in range(self.benchmark_iters):
                input_ids, _ = self.generate_random_batch()

                self.synchronize()
                start_time = timeit.default_timer()

                with torch.no_grad():
                    logits = self.model(input_ids)

                self.synchronize()
                end_time = timeit.default_timer()

                iter_time = end_time - start_time
                times.append(iter_time)

                # Update progress bar with current stats
                postfix = {
                    'current': f'{iter_time*1000:.2f}ms',
                    'mean': f'{statistics.mean(times)*1000:.2f}ms'
                }

                # Add memory stats if on CUDA
                if self.device == "cuda":
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    postfix['mem'] = f'{mem_mb:.0f}MB'

                pbar.set_postfix(postfix)
                pbar.update(1)

        # Get memory stats
        memory_stats = self.get_memory_stats()

        # Compute statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        return {
            "times": times,
            "mean_time_ms": mean_time * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": std_time * 1000,
            **{f"forward_{k}": v for k, v in memory_stats.items()},
        }

    def benchmark_backward(self) -> dict:
        """Benchmark forward + backward pass."""
        logger.info("Benchmarking forward + backward pass...")

        # Create a simple loss function
        criterion = nn.CrossEntropyLoss()

        # Reset memory stats
        self.reset_memory_stats()

        # Warmup
        if self.warmup_iters > 0:
            with tqdm(total=self.warmup_iters, desc="Warmup (fwd+bwd)", unit="iter",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for _ in range(self.warmup_iters):
                    input_ids, targets = self.generate_random_batch()

                    # Forward
                    logits = self.model(input_ids)

                    # Compute loss
                    loss = criterion(
                        logits.view(-1, self.vocab_size),
                        targets.view(-1)
                    )

                    # Backward
                    loss.backward()

                    # Synchronize after each warmup step
                    self.synchronize()

                    # Clear gradients
                    self.model.zero_grad()

                    pbar.update(1)
        else:
            logger.info("Skipping warmup (warmup_steps=0)")

        self.synchronize()

        # Benchmark
        times = []
        with tqdm(total=self.benchmark_iters, desc="Benchmark (fwd+bwd)", unit="iter") as pbar:
            for i in range(self.benchmark_iters):
                input_ids, targets = self.generate_random_batch()

                self.synchronize()
                start_time = timeit.default_timer()

                # Forward
                logits = self.model(input_ids)

                # Compute loss
                loss = criterion(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1)
                )

                # Backward
                loss.backward()

                self.synchronize()
                end_time = timeit.default_timer()

                iter_time = end_time - start_time
                times.append(iter_time)

                # Update progress bar with current stats
                postfix = {
                    'current': f'{iter_time*1000:.2f}ms',
                    'mean': f'{statistics.mean(times)*1000:.2f}ms',
                    'loss': f'{loss.item():.4f}'
                }

                # Add memory stats if on CUDA
                if self.device == "cuda":
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    postfix['mem'] = f'{mem_mb:.0f}MB'

                pbar.set_postfix(postfix)
                pbar.update(1)

                # Clear gradients
                self.model.zero_grad()

        # Get memory stats
        memory_stats = self.get_memory_stats()

        # Compute statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        return {
            "times": times,
            "mean_time_ms": mean_time * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": std_time * 1000,
            **{f"backward_{k}": v for k, v in memory_stats.items()},
        }

    def run_benchmark(self, mode: str = "both") -> dict:
        """Run benchmark suite.

        Args:
            mode: One of "forward" (forward pass only) or "both" (forward + backward)
        """
        logger.info("=" * 80)
        logger.info("Starting benchmark...")
        logger.info("=" * 80)
        logger.info(f"Model config: vocab_size={self.vocab_size}, context_length={self.context_length}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}, Precision: {self.precision}")
        logger.info(f"Warmup steps: {self.warmup_iters}, Measurement steps: {self.benchmark_iters}")
        logger.info(f"Mode: {mode}")
        logger.info("=" * 80)

        results = {}

        if mode in ("forward", "both"):
            # Benchmark forward pass
            forward_results = self.benchmark_forward()
            results["forward"] = forward_results

        if mode == "both":
            # Benchmark backward pass (which includes forward)
            backward_results = self.benchmark_backward()
            results["backward"] = backward_results

        return results

    def print_results(self, results: dict):
        """Print benchmark results in a formatted way."""
        logger.info("=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)

        # Forward pass results
        if "forward" in results:
            fwd = results['forward']
            logger.info("\nForward Pass (inference only):")
            logger.info(f"  Mean ± Std: {fwd['mean_time_ms']:.2f} ± {fwd['std_time_ms']:.2f} ms")
            logger.info(f"  Min time:   {fwd['min_time_ms']:.2f} ms")
            logger.info(f"  Max time:   {fwd['max_time_ms']:.2f} ms")

            if self.device == "cuda":
                logger.info(f"  Memory allocated: {fwd['forward_allocated_mb']:.2f} MB")
                logger.info(f"  Peak memory:      {fwd['forward_max_allocated_mb']:.2f} MB")

        # Backward pass results
        if "backward" in results:
            bwd = results['backward']
            logger.info("\nForward + Backward Pass (training step):")
            logger.info(f"  Mean ± Std: {bwd['mean_time_ms']:.2f} ± {bwd['std_time_ms']:.2f} ms")
            logger.info(f"  Min time:   {bwd['min_time_ms']:.2f} ms")
            logger.info(f"  Max time:   {bwd['max_time_ms']:.2f} ms")

            if self.device == "cuda":
                logger.info(f"  Memory allocated: {bwd['backward_allocated_mb']:.2f} MB")
                logger.info(f"  Peak memory:      {bwd['backward_max_allocated_mb']:.2f} MB")

        logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark BasicsTransformerLM performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Predefined configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help=f"Use a predefined model configuration: {list(MODEL_CONFIGS.keys())} or 'all' to run all configs"
    )

    # Model architecture arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="Maximum context length"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=768,
        help="Model dimensionality"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=3072,
        help="Feed-forward dimension"
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter"
    )

    # Benchmark configuration arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmark on"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Numerical precision"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps before timing (use 0 to skip warmup)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of measurement steps for timing"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["forward", "both"],
        help="Benchmark mode: 'forward' for forward pass only, 'both' for forward and backward"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving results to files"
    )

    return parser.parse_args()


def run_single_benchmark(args, config_name: str | None = None) -> dict:
    """Run a single benchmark with the given args.

    Returns:
        Dictionary with config and results.
    """
    # Create benchmark
    benchmark = ModelBenchmark(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        warmup_iters=args.warmup_steps,
        benchmark_iters=args.num_steps,
    )

    # Run benchmark
    results = benchmark.run_benchmark(mode=args.mode)

    # Print results
    benchmark.print_results(results)

    # Build config dict
    config = {
        "config_name": config_name or "custom",
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "batch_size": args.batch_size,
        "device": args.device,
        "precision": args.precision,
        "warmup_steps": args.warmup_steps,
        "num_steps": args.num_steps,
        "mode": args.mode,
    }

    return {"config": config, "results": results}


def main():
    """Main benchmark entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_results = []

    # Determine which configs to run
    if args.config == "all":
        configs_to_run = list(MODEL_CONFIGS.keys())
    elif args.config:
        configs_to_run = [args.config]
    else:
        configs_to_run = [None]  # Custom config

    for config_name in configs_to_run:
        # Apply predefined config if specified
        if config_name:
            config = MODEL_CONFIGS[config_name]
            logger.info(f"\n{'='*80}")
            logger.info(f"Running benchmark with config '{config_name}': {config}")
            logger.info(f"{'='*80}")
            args.d_model = config["d_model"]
            args.num_layers = config["num_layers"]
            args.num_heads = config["num_heads"]
            args.d_ff = config["d_ff"]

        # Run the benchmark
        run_result = run_single_benchmark(args, config_name)
        all_results.append(run_result)

        # Save results unless disabled
        if not getattr(args, 'no_save', False):
            run_id = generate_run_id(config_name, args)
            saved_files = save_results(
                results=run_result["results"],
                config=run_result["config"],
                output_dir=output_dir,
                run_id=run_id,
            )
            logger.info(f"\nResults saved:")
            for fmt, path in saved_files.items():
                logger.info(f"  {fmt}: {path}")

    # Print summary table if multiple configs were run
    if len(all_results) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
