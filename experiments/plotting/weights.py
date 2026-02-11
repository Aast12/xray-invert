"""Weight parameter distribution plots."""

import matplotlib.pyplot as plt


def plot_weights_boxplot(weights, save_fn=None):
    """Plot boxplots of each weight parameter.

    Args:
        weights: Dict mapping parameter names to arrays of shape (N, ...).
        save_fn: Optional callable for saving.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(1, len(weights.keys()), figsize=(15, 5))
    for i, (key, value) in enumerate(weights.items()):
        ax[i].boxplot(value)
        ax[i].set_title(f"{key}")

    if save_fn is not None:
        fig.savefig(save_fn("weights_boxplot.svg"))
        plt.close(fig)

    return fig, ax
