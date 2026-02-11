"""Analyze replicate results from cached data.

Usage:
    uv run python experiments/analyze_replicate_results.py --cache-dir ../replicates_cache/yiutl75n_64_5
"""

import argparse
import os

import matplotlib.pyplot as plt

from plotting import load_cache
from plotting.metrics import analyze_batch_results, create_simple_summary_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Path to cached replicate results directory")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save figures (defaults to cache-dir/figures)")
    args = parser.parse_args()

    data = load_cache(args.cache_dir)

    save_dir = args.save_dir or os.path.join(args.cache_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    def save_fn(filename):
        return os.path.join(save_dir, filename)

    fig, df = analyze_batch_results(
        [data["metrics"]], [data["txm"]], [data["weights"]], [data["pred"]],
        save_fn=save_fn,
    )

    fig2, axes2 = create_simple_summary_plot([data["metrics"]], save_fn=save_fn)

    print(f"Figures saved to {save_dir}")
