#!/usr/bin/env python3
"""
Performance benchmarking script for BasicsTransformerLM.

This script profiles forward and backward passes of the model with various configurations,
measuring speed and memory usage. It uses random weights and data for testing.

Example usage:
    # Basic benchmark with default parameters
    python src/benchmark.py

    # Benchmark with custom model size
    python src/benchmark.py --d_model 1024 --num_layers 12 --num_heads 16

    # Benchmark with mixed precision
    python src/benchmark.py --precision bf16

    # Benchmark with different context lengths and batch sizes
    python src/benchmark.py --context_length 2048 --batch_size 8

    # Run on CPU
    python src/benchmark.py --device cpu
"""

import argparse
import time
import sys
from pathlib import Path
import logging

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
        """Benchmark forward pass."""
        logger.info("Benchmarking forward pass...")

        # Reset memory stats
        self.reset_memory_stats()

        # Warmup
        with tqdm(total=self.warmup_iters, desc="Warmup (forward)", unit="iter",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for _ in range(self.warmup_iters):
                input_ids, _ = self.generate_random_batch()
                with torch.no_grad():
                    _ = self.model(input_ids)
                pbar.update(1)

        self.synchronize()

        # Benchmark
        times = []
        with tqdm(total=self.benchmark_iters, desc="Benchmark (forward)", unit="iter") as pbar:
            for i in range(self.benchmark_iters):
                input_ids, _ = self.generate_random_batch()

                self.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad():
                    logits = self.model(input_ids)

                self.synchronize()
                end_time = time.perf_counter()

                iter_time = end_time - start_time
                times.append(iter_time)

                # Update progress bar with current stats
                postfix = {
                    'current': f'{iter_time*1000:.2f}ms',
                    'mean': f'{sum(times)/len(times)*1000:.2f}ms'
                }

                # Add memory stats if on CUDA
                if self.device == "cuda":
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    postfix['mem'] = f'{mem_mb:.0f}MB'

                pbar.set_postfix(postfix)
                pbar.update(1)

        # Get memory stats
        memory_stats = self.get_memory_stats()

        return {
            "mean_time_ms": sum(times) / len(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
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

                # Clear gradients
                self.model.zero_grad()

                pbar.update(1)

        self.synchronize()

        # Benchmark
        times = []
        with tqdm(total=self.benchmark_iters, desc="Benchmark (fwd+bwd)", unit="iter") as pbar:
            for i in range(self.benchmark_iters):
                input_ids, targets = self.generate_random_batch()

                self.synchronize()
                start_time = time.perf_counter()

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
                end_time = time.perf_counter()

                iter_time = end_time - start_time
                times.append(iter_time)

                # Update progress bar with current stats
                postfix = {
                    'current': f'{iter_time*1000:.2f}ms',
                    'mean': f'{sum(times)/len(times)*1000:.2f}ms',
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

        return {
            "mean_time_ms": sum(times) / len(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
            **{f"backward_{k}": v for k, v in memory_stats.items()},
        }

    def run_benchmark(self) -> dict:
        """Run full benchmark suite."""
        logger.info("=" * 80)
        logger.info("Starting benchmark...")
        logger.info("=" * 80)
        logger.info(f"Model config: vocab_size={self.vocab_size}, context_length={self.context_length}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}, Precision: {self.precision}")
        logger.info("=" * 80)

        results = {}

        # Benchmark forward pass
        forward_results = self.benchmark_forward()
        results["forward"] = forward_results

        # Benchmark backward pass
        backward_results = self.benchmark_backward()
        results["backward"] = backward_results

        return results

    def print_results(self, results: dict):
        """Print benchmark results in a formatted way."""
        logger.info("=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)

        # Forward pass results
        logger.info("\nForward Pass:")
        logger.info(f"  Mean time: {results['forward']['mean_time_ms']:.2f} ms")
        logger.info(f"  Min time:  {results['forward']['min_time_ms']:.2f} ms")
        logger.info(f"  Max time:  {results['forward']['max_time_ms']:.2f} ms")
        logger.info(f"  Std dev:   {results['forward']['std_time_ms']:.2f} ms")

        if self.device == "cuda":
            logger.info(f"\n  Memory allocated: {results['forward']['forward_allocated_mb']:.2f} MB")
            logger.info(f"  Memory reserved:  {results['forward']['forward_reserved_mb']:.2f} MB")
            logger.info(f"  Peak memory:      {results['forward']['forward_max_allocated_mb']:.2f} MB")

        # Backward pass results
        logger.info("\nForward + Backward Pass:")
        logger.info(f"  Mean time: {results['backward']['mean_time_ms']:.2f} ms")
        logger.info(f"  Min time:  {results['backward']['min_time_ms']:.2f} ms")
        logger.info(f"  Max time:  {results['backward']['max_time_ms']:.2f} ms")
        logger.info(f"  Std dev:   {results['backward']['std_time_ms']:.2f} ms")

        if self.device == "cuda":
            logger.info(f"\n  Memory allocated: {results['backward']['backward_allocated_mb']:.2f} MB")
            logger.info(f"  Memory reserved:  {results['backward']['backward_reserved_mb']:.2f} MB")
            logger.info(f"  Peak memory:      {results['backward']['backward_max_allocated_mb']:.2f} MB")

        logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark BasicsTransformerLM performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--warmup_iters",
        type=int,
        default=5,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )

    # Optional output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file to save results (JSON format)"
    )

    return parser.parse_args()


def main():
    """Main benchmark entry point."""
    args = parse_args()

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
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    # Run benchmark
    results = benchmark.run_benchmark()

    # Print results
    benchmark.print_results(results)

    # Save results if output file specified
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results for JSON serialization
        json_results = {
            "config": {
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
            },
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
