"""Plotting utilities for experiment visualization.

All plotting functions accept data (arrays/dicts), an optional ``save_fn``
parameter (default ``None``), and return ``(fig, ax)`` tuples.
"""

import os

import joblib

from .ground_truth import plot_gt_comparison, plot_gt_metrics, plot_gt_vs_naive
from .histograms import plot_extreme_histograms, plot_histogram_heatmap, segmentation_histogram
from .metrics import (
    analyze_batch_results,
    create_simple_summary_plot,
    plot_batch_metrics,
    plot_metric_distributions,
    plot_metrics_boxplot,
)
from .samples import plot_percentile_samples, plot_sample_comparison
from .weights import plot_weights_boxplot

__all__ = [
    "segmentation_histogram",
    "plot_histogram_heatmap",
    "plot_extreme_histograms",
    "plot_metric_distributions",
    "plot_metrics_boxplot",
    "plot_batch_metrics",
    "analyze_batch_results",
    "create_simple_summary_plot",
    "plot_percentile_samples",
    "plot_sample_comparison",
    "plot_weights_boxplot",
    "plot_gt_comparison",
    "plot_gt_vs_naive",
    "plot_gt_metrics",
    "load_cache",
]


def load_cache(cache_dir: str) -> dict:
    """Load cached replicate results.

    Returns dict with keys: imgs, masks, txm, pred, weights, metrics.
    """
    keys = ["imgs", "masks", "txm", "pred", "weights", "metrics"]
    data = {}
    for key in keys:
        path = os.path.join(cache_dir, f"{key}.joblib")
        data[key] = joblib.load(path)
    return data
