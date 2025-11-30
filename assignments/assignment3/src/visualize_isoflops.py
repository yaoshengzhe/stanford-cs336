#!/usr/bin/env python3
"""Visualize isoflops curves from training data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def load_data(filepath: str) -> list[dict]:
    """Load isoflops data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def format_flops(flops: float) -> str:
    """Format FLOPs value for legend display."""
    exponent = int(np.log10(flops))
    mantissa = flops / (10**exponent)
    if mantissa == 1.0:
        return f"$10^{{{exponent}}}$ FLOPs"
    return f"${mantissa:.0f} \\times 10^{{{exponent}}}$ FLOPs"


def main():
    data_path = Path(__file__).parent.parent / "data" / "isoflops_curves.json"
    data = load_data(data_path)

    # Group data by compute budget
    budgets: dict[float, list[tuple[float, float]]] = {}
    for entry in data:
        budget = entry["compute_budget"]
        if budget not in budgets:
            budgets[budget] = []
        budgets[budget].append((entry["final_loss"], entry["parameters"]))

    # Sort budgets for consistent coloring
    sorted_budgets = sorted(budgets.keys())

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color map for different compute budgets
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_budgets)))

    # Different markers for black-and-white readability
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "<"]

    # Track optimal points (lowest loss) for each compute budget
    optimal_points: list[tuple[float, float, float]] = []  # (flops, params, loss)

    for i, (budget, color) in enumerate(zip(sorted_budgets, colors)):
        points = sorted(budgets[budget], key=lambda x: x[1])  # Sort by params
        losses = np.array([p[0] for p in points])
        params = np.array([p[1] for p in points])
        marker = markers[i % len(markers)]

        # Plot data points (x=params, y=loss)
        ax.plot(
            params,
            losses,
            marker,
            color=color,
            label=format_flops(budget),
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        # Use spline interpolation in log space for smooth curve
        log_params = np.log10(params)
        spline = UnivariateSpline(log_params, losses, s=0.01, k=3)

        # Generate smooth fit curve
        log_params_smooth = np.linspace(log_params.min(), log_params.max(), 100)
        losses_smooth = spline(log_params_smooth)
        params_smooth = 10**log_params_smooth

        ax.plot(params_smooth, losses_smooth, "-", color=color, alpha=0.7, linewidth=2)

        # Find optimal point (minimum loss) from the spline
        min_idx = np.argmin(losses_smooth)
        optimal_params = params_smooth[min_idx]
        optimal_loss = losses_smooth[min_idx]
        optimal_points.append((budget, optimal_params, optimal_loss))

    # Plot the optimal frontier curve connecting lowest loss points
    if len(optimal_points) >= 2:
        optimal_points.sort(key=lambda x: x[0])  # Sort by compute budget
        opt_flops = np.array([p[0] for p in optimal_points])
        opt_params = np.array([p[1] for p in optimal_points])
        opt_losses = np.array([p[2] for p in optimal_points])

        # Plot optimal points with distinct markers
        ax.scatter(
            opt_params,
            opt_losses,
            s=120,
            c="red",
            marker="*",
            edgecolors="black",
            linewidths=1,
            zorder=10,
            label="Optimal (min loss)",
        )

        # Linear fit on optimal points: loss = m * params + c
        # Use log10(params) for x-axis since plot is log-scale
        log_opt_params = np.log10(opt_params)
        linear_coeffs = np.polyfit(log_opt_params, opt_losses, 1)
        m = linear_coeffs[0]  # slope
        c = linear_coeffs[1]  # intercept

        # Generate line for plotting (extended to prediction)
        # First, predict optimal params for 10^23 FLOPs using power law fit
        log_C = np.log10(opt_flops)
        log_N = np.log10(opt_params)
        params_coeffs = np.polyfit(log_C, log_N, 1)
        C_target = 1e23
        log_N_predicted = params_coeffs[0] * np.log10(C_target) + params_coeffs[1]
        N_predicted = 10**log_N_predicted

        # Predict loss using linear fit
        loss_predicted = m * log_N_predicted + c

        print(f"Linear fit: loss = {m:.4f} * log10(params) + {c:.4f}")
        print(f"Predicted optimal model size for 10^23 FLOPs: {N_predicted:.2e} ({N_predicted/1e9:.0f}B params)")
        print(f"Predicted loss: {loss_predicted:.4f}")

        # Plot the linear fit line through optimal points
        log_params_range = np.linspace(log_opt_params.min(), log_N_predicted, 100)
        loss_fit_line = m * log_params_range + c
        ax.plot(
            10**log_params_range,
            loss_fit_line,
            "--",
            color="red",
            linewidth=2,
            alpha=0.8,
            zorder=9,
            label="Linear fit (optimal)",
        )

        # Plot prediction point
        ax.scatter(
            [N_predicted],
            [loss_predicted],
            s=200,
            c="orange",
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=11,
            label=f"Predicted @ $10^{{23}}$ FLOPs\n({N_predicted/1e9:.0f}B params)",
        )

    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("IsoFLOPs Curves: Model Parameters vs Training Loss", fontsize=14)
    ax.legend(title="Compute Budget", loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "figures" / "isoflops_curves.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
