"""THRML-specific visualization functions for sampling and energy landscapes."""

from typing import Any, List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .core import apply_standard_formatting, create_figure, ensure_matplotlib, get_config, save_figure


def plot_sampling_trajectory(
    samples: Float[Array, "n_samples n_dims"],
    dims: Optional[tuple[int, int]] = (0, 1),
    true_distribution: Optional[Any] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot trajectory of samples through state space.

    **Arguments:**

    - `samples`: Array of samples (n_samples, n_dimensions)
    - `dims`: Which dimensions to plot (default first two)
    - `true_distribution`: Optional true distribution for comparison
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    # Plot trajectory
    ax.plot(samples[:, dims[0]], samples[:, dims[1]], alpha=0.3, linewidth=0.5, color=get_config().primary_color)
    ax.scatter(samples[:, dims[0]], samples[:, dims[1]], s=1, alpha=0.5, color=get_config().primary_color)

    # Mark start and end
    ax.scatter(samples[0, dims[0]], samples[0, dims[1]], s=100, marker="o", color="green", label="Start", zorder=5)
    ax.scatter(samples[-1, dims[0]], samples[-1, dims[1]], s=100, marker="*", color="red", label="End", zorder=5)

    ax.set_xlabel(f"Dimension {dims[0]}")
    ax.set_ylabel(f"Dimension {dims[1]}")
    ax.set_title("Sampling Trajectory")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="trajectories", close=False)

    return fig, ax


def plot_sample_statistics(
    samples: Float[Array, "n_samples ..."],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot statistical properties of samples.

    **Arguments:**

    - `samples`: Array of samples
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, axes = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=3)

    # Mean evolution
    running_mean = jnp.cumsum(samples, axis=0) / jnp.arange(1, len(samples) + 1)[:, None]
    for i in range(min(5, samples.shape[1])):  # Plot first 5 dimensions
        axes[0].plot(running_mean[:, i], label=f"Dim {i}", alpha=0.7)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Running Mean")
    axes[0].set_title("Mean Convergence")
    axes[0].legend()
    apply_standard_formatting(axes[0])

    # Variance evolution
    running_var = jnp.var(samples[: len(samples)], axis=0)
    for i in range(min(5, samples.shape[1])):
        axes[1].plot(running_var[..., i] if running_var.ndim > 1 else [running_var], label=f"Dim {i}", alpha=0.7)
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Variance Evolution")
    axes[1].legend()
    apply_standard_formatting(axes[1])

    # Sample histogram (first dimension)
    axes[2].hist(samples[:, 0], bins=30, color=get_config().primary_color, alpha=0.7, edgecolor="black")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Sample Distribution (Dim 0)")
    apply_standard_formatting(axes[2])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, axes


def plot_energy_landscape(
    energy_fn: Any,
    x_range: tuple[float, float] = (-3, 3),
    y_range: tuple[float, float] = (-3, 3),
    resolution: int = 100,
    samples: Optional[Float[Array, "n_samples 2"]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot 2D energy landscape with optional sample overlay.

    **Arguments:**

    - `energy_fn`: Function that computes energy given state
    - `x_range`: Range for x-axis
    - `y_range`: Range for y-axis
    - `resolution`: Grid resolution
    - `samples`: Optional samples to overlay
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    # Create grid
    x = jnp.linspace(x_range[0], x_range[1], resolution)
    y = jnp.linspace(y_range[0], y_range[1], resolution)
    X, Y = jnp.meshgrid(x, y)

    # Compute energies
    points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    energies = jnp.array([energy_fn(p) for p in points])
    Z = energies.reshape(resolution, resolution)

    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="Energy")

    # Overlay samples if provided
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=10, color="red", alpha=0.5, label="Samples")
        ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Energy Landscape")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_block_structure(
    blocks: List[Any], figsize: Optional[tuple[int, int]] = None, save_path: Optional[str] = None, **kwargs
):
    """Visualize THRML block structure.

    **Arguments:**

    - `blocks`: List of THRML Block objects
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    block_sizes = [len(block.nodes) for block in blocks]
    block_labels = [f"Block {i}" for i in range(len(blocks))]

    ax.bar(range(len(blocks)), block_sizes, color=get_config().primary_color, alpha=0.7)
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Block Structure")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_mixing_diagnostics(
    samples: Float[Array, "n_samples n_chains ..."],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot chain mixing diagnostics.

    **Arguments:**

    - `samples`: Array of samples (n_samples, n_chains, ...)
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_samples, n_chains = samples.shape[:2]

    fig, axes = create_figure(figsize=figsize or get_config().large_figsize, nrows=2, ncols=1)

    # Trace plots
    for chain in range(n_chains):
        axes[0].plot(samples[:, chain, 0], alpha=0.7, label=f"Chain {chain}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Trace Plots (First Dimension)")
    axes[0].legend()
    apply_standard_formatting(axes[0])

    # Densities
    for chain in range(n_chains):
        axes[1].hist(samples[:, chain, 0], bins=30, alpha=0.5, label=f"Chain {chain}", density=True)
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Marginal Distributions")
    axes[1].legend()
    apply_standard_formatting(axes[1])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes


def plot_autocorrelation(
    samples: Float[Array, "n_samples"],
    max_lag: int = 50,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot autocorrelation function of samples.

    **Arguments:**

    - `samples`: 1D array of samples
    - `max_lag`: Maximum lag to compute
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    # Compute autocorrelation
    mean = jnp.mean(samples)
    var = jnp.var(samples)

    autocorr = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            cov = jnp.mean((samples[:-lag] - mean) * (samples[lag:] - mean))
            autocorr.append(cov / var)

    ax.plot(range(max_lag + 1), autocorr, linewidth=2, color=get_config().primary_color)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function")
    ax.set_ylim([-0.2, 1.0])
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_acceptance_rates(
    acceptance_rates: Float[Array, "n_steps"],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot acceptance rates over sampling iterations.

    **Arguments:**

    - `acceptance_rates`: Array of acceptance rates
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    ax.plot(acceptance_rates, linewidth=2, color=get_config().primary_color)
    ax.axhline(y=0.234, color="red", linestyle="--", alpha=0.5, label="Optimal (0.234)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("MCMC Acceptance Rate")
    ax.set_ylim([0, 1])
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_ising_spins(
    spins: Float[Array, "n_samples n_spins"],
    timesteps: Optional[List[int]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot Ising model spin configurations over time.

    **Arguments:**

    - `spins`: Array of spin configurations (n_samples, n_spins)
    - `timesteps`: Optional specific timesteps to show
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    if timesteps is None:
        # Show evenly spaced samples
        n_show = min(10, len(spins))
        timesteps = jnp.linspace(0, len(spins) - 1, n_show, dtype=int)

    n_rows = int(jnp.ceil(len(timesteps) / 5))
    n_cols = min(5, len(timesteps))

    fig, axes = create_figure(figsize=figsize or get_config().large_figsize, nrows=n_rows, ncols=n_cols)

    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, t in enumerate(timesteps):
        ax = axes[idx]

        # Try to reshape into 2D grid if possible
        n_spins = len(spins[t])
        grid_size = int(jnp.sqrt(n_spins))

        if grid_size * grid_size == n_spins:
            spin_grid = spins[t].reshape(grid_size, grid_size)
            im = ax.imshow(spin_grid, cmap="RdBu", vmin=-1, vmax=1)
        else:
            im = ax.imshow(spins[t].reshape(1, -1), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")

        ax.set_title(f"t={t}")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(timesteps), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes


def plot_categorical_states(
    states: Float[Array, "n_samples n_variables"],
    n_categories: int,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot categorical state evolution.

    **Arguments:**

    - `states`: Array of categorical states
    - `n_categories`: Number of categories
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    # Plot as heatmap
    im = ax.imshow(states.T, aspect="auto", cmap="tab10", vmin=0, vmax=n_categories - 1)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Variable")
    ax.set_title("Categorical State Evolution")
    plt.colorbar(im, ax=ax, label="Category", ticks=range(n_categories))

    apply_standard_formatting(ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax
