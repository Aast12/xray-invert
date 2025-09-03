#!/usr/bin/env python3
"""
Visualization script for the window function in chest_xray_sim.inverse.operators.

Creates visualizations to show how the window function transforms pixel values,
with gamma representing the value at the middle point (x=0.5).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def window(image, window_center, window_width, gamma):
    """
    Apply window transformation to an image.
    
    Args:
        image: Input image or values
        window_center: Center of the window
        window_width: Width of the window
        gamma: Value at the middle point (x=0.5)
        
    Returns:
        Transformed image or values
    """
    x = image
    x = (x - window_center) / window_width + (window_center - window_center / 2)
    alpha = jnp.log(gamma) / jnp.log(0.5)
    
    return x.clip(0.0, 1.0) ** alpha


def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "figures", "window_function")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_window_basic_effects(output_dir):
    """Create a plot showing basic effects of window parameters"""
    x = np.linspace(0, 1, 1000)
    
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle("Window Function Basic Effects", fontsize=18, y=0.98)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Effect of window center
    ax1 = fig.add_subplot(gs[0, 0])
    centers = [0.3, 0.5, 0.7]
    for center in centers:
        y = window(x, center, 0.2, 0.5)
        ax1.plot(x, y, linewidth=2, label=f"center={center:.1f}")
    
    # Add reference line and point
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.6)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    
    ax1.set_title("Effect of Window Center", fontsize=14)
    ax1.set_xlabel("Input Pixel Value", fontsize=12)
    ax1.set_ylabel("Output Pixel Value", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Effect of window width
    ax2 = fig.add_subplot(gs[0, 1])
    widths = [0.1, 0.2, 0.4, 0.8]
    for width in widths:
        y = window(x, 0.5, width, 0.5)
        ax2.plot(x, y, linewidth=2, label=f"width={width:.1f}")
    
    # Add reference line and point
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.6)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    
    ax2.set_title("Effect of Window Width", fontsize=14)
    ax2.set_xlabel("Input Pixel Value", fontsize=12)
    ax2.set_ylabel("Output Pixel Value", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Effect of gamma (midpoint value)
    ax3 = fig.add_subplot(gs[1, 0])
    gammas = [0.2, 0.3, 0.5, 0.7, 0.8]
    
    for gamma in gammas:
        y = window(x, 0.5, 0.2, gamma)
        ax3.plot(x, y, linewidth=2, label=f"γ={gamma:.1f}")
        # Mark the midpoint value
        ax3.plot(0.5, gamma, 'o', markersize=6, markeredgecolor='black')
    
    # Add reference line
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.6)
    
    ax3.set_title("Effect of Gamma (Value at x=0.5)", fontsize=14)
    ax3.set_xlabel("Input Pixel Value", fontsize=12)
    ax3.set_ylabel("Output Pixel Value", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Combined effects with practical examples
    ax4 = fig.add_subplot(gs[1, 1])
    params = [
        (0.5, 0.2, 0.5, "Standard S-curve"),
        (0.5, 2.0, 0.5, "Near-linear"),
        (0.3, 0.2, 0.7, "Enhance shadows"),
        (0.7, 0.2, 0.3, "Enhance highlights"),
    ]
    
    for center, width, gamma, label in params:
        y = window(x, center, width, gamma)
        ax4.plot(x, y, linewidth=2, label=label)
    
    # Add reference line
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Identity")
    
    ax4.set_title("Practical Window Function Examples", fontsize=14)
    ax4.set_xlabel("Input Pixel Value", fontsize=12)
    ax4.set_ylabel("Output Pixel Value", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.savefig(os.path.join(output_dir, "window_basic_effects.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_gamma_midpoint_demo(output_dir):
    """Create a detailed visualization of how gamma controls the midpoint value"""
    x = np.linspace(0, 1, 1000)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a range of gamma values
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))
    
    # Plot each curve
    for i, gamma in enumerate(gammas):
        y = window(x, 0.5, 0.2, gamma)
        line, = ax.plot(x, y, linewidth=2.5, color=colors[i], label=f"γ = {gamma:.1f}")
        
        # Mark the midpoint with a dot
        ax.plot(0.5, gamma, 'o', markersize=8, color=colors[i], markeredgecolor='black')
    
    # Add reference lines
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Identity (y=x)")
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label="x = 0.5")
    
    # Add annotations
    ax.annotate("γ = value at x=0.5", xy=(0.5, 0.5), xytext=(0.6, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center')
    
    # Configure plot
    ax.set_title("Window Function: Gamma Controls the Value at x=0.5", fontsize=18)
    ax.set_xlabel("Input Pixel Value", fontsize=14)
    ax.set_ylabel("Output Pixel Value", fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add ticks at each 0.1 interval
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.savefig(os.path.join(output_dir, "gamma_midpoint_demo.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_linear_approximations(output_dir):
    """Create visualizations of linear-like LUTs with different max values"""
    x = np.linspace(0, 1, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Linear-like curves
    linear_params = [
        # (center, width, gamma, max_val)
        (0.5, 8.0, 0.5, 0.90),
        (0.5, 4.0, 0.5, 0.85),
        (0.5, 2.0, 0.5, 0.75),
        (0.5, 1.0, 0.5, 0.60),
        (0.5, 0.6, 0.5, 0.50),
    ]
    
    for center, width, gamma, max_val in linear_params:
        y = window(x, center, width, gamma)
        ax1.plot(x, y, linewidth=2.5, label=f"width={width:.1f}, max≈{max_val:.2f}")
    
    # Add identity line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Identity")
    
    ax1.set_title("Linear-like Window Functions\nwith Different Maximum Values", fontsize=16)
    ax1.set_xlabel("Input Pixel Value", fontsize=14)
    ax1.set_ylabel("Output Pixel Value", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Derivatives to show linearity
    for center, width, gamma, max_val in linear_params:
        y = window(x, center, width, gamma)
        # Calculate derivative
        y_diff = np.diff(y) / np.diff(x)
        x_diff = x[:-1] + 0.5/len(x)
        ax2.plot(x_diff, y_diff, linewidth=2.5, label=f"width={width:.1f}")
    
    # Add horizontal lines at key slope values
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.4)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    
    ax2.set_title("Derivatives of Linear-like Window Functions\nShowing Degree of Linearity", fontsize=16)
    ax2.set_xlabel("Input Pixel Value", fontsize=14)
    ax2.set_ylabel("Slope (dy/dx)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.2)  # Allow slightly higher values to show the slopes clearly
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "linear_approximations.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_window_lookup_tables(output_dir):
    """Create a grid of window function lookup tables for different parameter combinations"""
    x = np.linspace(0, 1, 500)
    
    # Define parameter ranges
    centers = [0.3, 0.5, 0.7]
    widths = [0.1, 0.2, 0.5]
    gammas = [0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(len(centers), len(widths), figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle("Window Function Lookup Tables\n(γ = value at x=0.5)", fontsize=18)
    
    # Plot each parameter combination
    for i, center in enumerate(centers):
        for j, width in enumerate(widths):
            ax = axes[i, j]
            
            for gamma in gammas:
                y = window(x, center, width, gamma)
                ax.plot(x, y, linewidth=2, label=f"γ={gamma:.1f}")
                
                # Add a dot at the midpoint value
                ax.plot(0.5, gamma, 'o', markersize=4)
            
            # Add identity line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
            
            # Set title and labels
            ax.set_title(f"center={center:.1f}, width={width:.1f}", fontsize=12)
            if i == len(centers)-1:
                ax.set_xlabel("Input Pixel Value", fontsize=10)
            if j == 0:
                ax.set_ylabel("Output Pixel Value", fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Only add legend to the first plot
            if i == 0 and j == 0:
                ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_dir, "window_lookup_tables.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_window_vs_linear(output_dir):
    """Compare window function with linear contrast adjustment"""
    x = np.linspace(0, 1, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Window function parameters for different effects
    window_params = [
        # (center, width, gamma, label)
        (0.5, 0.1, 0.5, "Window (high contrast)"),
        (0.5, 0.2, 0.5, "Window (medium contrast)"),
        (0.5, 0.5, 0.5, "Window (low contrast)"),
        (0.5, 5.0, 0.5, "Window (linear-like, max≈0.85)"),
    ]
    
    # Plot window functions
    for center, width, gamma, label in window_params:
        y = window(x, center, width, gamma)
        ax.plot(x, y, linewidth=2.5, label=label)
    
    # Plot linear contrast adjustments for comparison
    linear_adjustments = [
        # (slope, intercept, label)
        (1.0, 0.0, "Linear (y=x)"),
        (0.8, 0.1, "Linear (y=0.8x+0.1)"),
        (0.6, 0.2, "Linear (y=0.6x+0.2)"),
    ]
    
    for slope, intercept, label in linear_adjustments:
        y = slope * x + intercept
        y_clipped = np.clip(y, 0, 1)  # Clip to [0,1] range
        ax.plot(x, y_clipped, '--', linewidth=2, label=label)
    
    # Configure plot
    ax.set_title("Window Function vs. Linear Contrast Adjustment", fontsize=18)
    ax.set_xlabel("Input Pixel Value", fontsize=14)
    ax.set_ylabel("Output Pixel Value", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.savefig(os.path.join(output_dir, "window_vs_linear.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all window function visualizations"""
    output_dir = create_output_dir()
    
    # Generate all plots
    plot_window_basic_effects(output_dir)
    plot_gamma_midpoint_demo(output_dir)
    plot_linear_approximations(output_dir)
    plot_window_lookup_tables(output_dir)
    plot_window_vs_linear(output_dir)
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()