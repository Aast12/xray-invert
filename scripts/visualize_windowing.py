#!/usr/bin/env python
"""
Visualization script for the windowing function in chest_xray_sim.inverse.operators.

This script creates plots to visualize how different windowing parameters affect
the transformation of pixel values in the [0, 1] range.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from matplotlib.gridspec import GridSpec


def windowing(image, window_center, window_width, gamma):
    """
    Apply windowing transformation to an image.
    
    Args:
        image: Input image or values
        window_center: Center of the window
        window_width: Width of the window
        gamma: Power for gamma correction
    
    Returns:
        Transformed image or values
    """
    x = image
    x = (x - window_center) / window_width
    x = jax.nn.sigmoid(x) ** gamma
    return x


def plot_windowing_effects():
    """Plot the effects of different windowing parameters on pixel values."""
    # Create input values in [0, 1] range
    x = np.linspace(0, 1, 1000)
    
    # Figure setup
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Varying window center
    ax1 = fig.add_subplot(gs[0, 0])
    centers = [0.3, 0.5, 0.7]
    for center in centers:
        y = windowing(x, center, 0.2, 1.0)
        ax1.plot(x, y, label=f"center={center:.1f}")
    
    ax1.set_title("Effect of Window Center\n(width=0.2, gamma=1.0)")
    ax1.set_xlabel("Input Pixel Value")
    ax1.set_ylabel("Output Pixel Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Varying window width
    ax2 = fig.add_subplot(gs[0, 1])
    widths = [0.1, 0.2, 0.4]
    for width in widths:
        y = windowing(x, 0.5, width, 1.0)
        ax2.plot(x, y, label=f"width={width:.1f}")
    
    ax2.set_title("Effect of Window Width\n(center=0.5, gamma=1.0)")
    ax2.set_xlabel("Input Pixel Value")
    ax2.set_ylabel("Output Pixel Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Varying gamma
    ax3 = fig.add_subplot(gs[1, 0])
    gammas = [0.5, 1.0, 2.0]
    for gamma in gammas:
        y = windowing(x, 0.5, 0.2, gamma)
        ax3.plot(x, y, label=f"gamma={gamma:.1f}")
    
    ax3.set_title("Effect of Gamma\n(center=0.5, width=0.2)")
    ax3.set_xlabel("Input Pixel Value")
    ax3.set_ylabel("Output Pixel Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined effect with different parameter sets
    ax4 = fig.add_subplot(gs[1, 1])
    params = [
        (0.3, 0.1, 0.8),  # (center, width, gamma)
        (0.5, 0.2, 1.0),
        (0.7, 0.3, 1.2),
    ]
    
    for center, width, gamma in params:
        y = windowing(x, center, width, gamma)
        ax4.plot(x, y, label=f"c={center:.1f}, w={width:.1f}, γ={gamma:.1f}")
    
    ax4.set_title("Combined Effects of Different Parameter Sets")
    ax4.set_xlabel("Input Pixel Value")
    ax4.set_ylabel("Output Pixel Value")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add identity line to all plots for reference
    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot(x, x, 'k--', alpha=0.5, label="_Identity" if ax != ax1 else "Identity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("figures/windowing_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_windowing_lookup_tables():
    """
    Create a visualization of windowing lookup tables for different parameter combinations.
    This shows how input pixel values map to output values as lookup tables.
    """
    # Create a fine-grained input range
    x = np.linspace(0, 1, 500)
    
    # Define parameter ranges to visualize
    centers = [0.3, 0.5, 0.7]
    widths = [0.1, 0.2, 0.3]
    gammas = [0.8, 1.0, 1.5]
    
    # Create figure
    fig, axes = plt.subplots(len(centers), len(widths), figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle("Windowing Lookup Tables: Mapping [0,1] → [0,1]", fontsize=16)
    
    # Plot for each parameter combination
    for i, center in enumerate(centers):
        for j, width in enumerate(widths):
            ax = axes[i, j]
            
            # Plot curves for different gamma values
            for gamma in gammas:
                y = windowing(x, center, width, gamma)
                ax.plot(x, y, label=f"γ={gamma:.1f}")
            
            # Add identity line
            ax.plot(x, x, 'k--', alpha=0.5)
            
            # Set title and labels
            ax.set_title(f"c={center:.1f}, w={width:.1f}")
            if i == len(centers) - 1:
                ax.set_xlabel("Input Pixel Value")
            if j == 0:
                ax.set_ylabel("Output Pixel Value")
            
            ax.grid(True, alpha=0.3)
            
            # Only add legend to the first plot
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("figures/windowing_lookup_tables.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)
    
    # Generate both visualizations
    plot_windowing_effects()
    plot_windowing_lookup_tables()
    
    print("Visualization complete. Images saved to figures/ directory.")