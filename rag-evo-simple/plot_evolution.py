#!/usr/bin/env python3
"""Create a generation-vs-fitness chart for a ShinkaEvolve run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_plot(results_dir: Path, output_path: Path) -> Path:
    try:
        from shinka.plots import plot_evals_performance
        from shinka.utils.load_df import load_programs_to_df
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the shinka package in the active environment."
        ) from exc

    programs_df = load_programs_to_df(str(results_dir), verbose=True)
    if programs_df is None or programs_df.empty:
        raise RuntimeError(f"No program data found in {results_dir}")

    fig, ax = plot_evals_performance(
        programs_df,
        title="MultiHop-RAG Evolution: Generation vs Fitness",
        xlabel="Generation",
        ylabel="Combined Score",
        plot_path_to_best_node=True,
        scatter_improvements_only=False,
        annotate=False,
        show_cost=False,
    )

    correct_df = programs_df[programs_df["correct"]].copy()
    if not correct_df.empty:
        best_row = correct_df.loc[correct_df["combined_score"].idxmax()]
        best_generation = int(best_row["generation"])
        best_score = float(best_row["combined_score"])
        ax.axhline(best_score, color="#2E8B57", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.text(
            0.99,
            0.02,
            f"Best: gen {best_generation}, score {best_score:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a PNG chart of generation vs fitness for a ShinkaEvolve run."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/results_rag_evo_simple_smoke",
        help="Path to the Shinka results directory or programs.sqlite parent directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults to <results_dir>/plots/generation_vs_fitness.png.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if results_dir.name.endswith(".sqlite"):
        results_dir = results_dir.parent

    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else results_dir / "plots" / "generation_vs_fitness.png"
    )

    saved_path = build_plot(results_dir, output_path)
    print(f"Saved generation-vs-fitness plot to {saved_path}")


if __name__ == "__main__":
    main()
