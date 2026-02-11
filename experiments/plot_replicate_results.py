"""Plot metrics from cached replicate results.

Usage:
    uv run python experiments/plot_replicate_results.py --cache-dir ../replicates_cache/yiutl75n_64_5
"""

import argparse
import os

import matplotlib.pyplot as plt

from plotting import load_cache
from plotting.metrics import plot_batch_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Path to cached replicate results directory")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save the figure (optional)")
    args = parser.parse_args()

    data = load_cache(args.cache_dir)

    # plot_batch_metrics expects a list of EvalMetrics, but load_cache
    # returns a single concatenated EvalMetrics — wrap it in a list
    fig, axes = plot_batch_metrics(
        [data["metrics"]],
        save_fn=lambda f: os.path.join(args.cache_dir, f) if not args.save_path else args.save_path,
    )

    if args.save_path:
        fig.savefig(args.save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
