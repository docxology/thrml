"""Statistical plotting and diagnostics."""

from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .core import apply_standard_formatting, create_figure, ensure_matplotlib, ensure_seaborn, get_config, save_figure

# Import statistical analysis utilities
try:
    from ..utils.statistical_analysis import linear_regression, pearson_correlation

    HAS_STATS = True
except ImportError:
    HAS_STATS = False


def plot_distribution(
    samples: Float[Array, "n_samples"],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    bins: int = 30,
    **kwargs,
):
    """Plot distribution of samples with histogram and KDE.

    **Arguments:**

    - `samples`: Array of samples
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `bins`: Number of histogram bins
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    fig, ax = create_figure(figsize=figsize)

    # Histogram
    ax.hist(samples, bins=bins, density=True, alpha=0.6, color=get_config().primary_color, edgecolor="black")

    # KDE
    sns.kdeplot(data=samples, ax=ax, linewidth=2, color=get_config().accent_color)

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Sample Distribution")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_distribution_comparison(
    samples_list: List[Float[Array, "n_samples"]],
    labels: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Compare multiple distributions.

    **Arguments:**

    - `samples_list`: List of sample arrays
    - `labels`: Optional labels for each distribution
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    fig, ax = create_figure(figsize=figsize)

    if labels is None:
        labels = [f"Distribution {i}" for i in range(len(samples_list))]

    for samples, label in zip(samples_list, labels):
        sns.kdeplot(data=samples, ax=ax, linewidth=2, label=label, alpha=0.7)

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Comparison")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_histogram(
    samples: Float[Array, "n_samples"],
    bins: Union[int, str] = "auto",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot histogram of samples.

    **Arguments:**

    - `samples`: Array of samples
    - `bins`: Number of bins or binning strategy
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    ax.hist(samples, bins=bins, color=get_config().primary_color, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title("Histogram")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_kde(
    samples: Float[Array, "n_samples"],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot kernel density estimate.

    **Arguments:**

    - `samples`: Array of samples
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    fig, ax = create_figure(figsize=figsize)

    sns.kdeplot(data=samples, ax=ax, linewidth=2, color=get_config().primary_color, fill=True, alpha=0.3)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Kernel Density Estimate")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_qq_plot(
    samples: Float[Array, "n_samples"],
    theoretical_dist: str = "norm",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot Q-Q plot against theoretical distribution.

    **Arguments:**

    - `samples`: Array of samples
    - `theoretical_dist`: Theoretical distribution ('norm', 'uniform', etc.)
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required for Q-Q plots. Install with: pip install scipy")

    fig, ax = create_figure(figsize=figsize)

    if theoretical_dist == "norm":
        stats.probplot(samples, dist="norm", plot=ax)
    else:
        stats.probplot(samples, dist=theoretical_dist, plot=ax)

    ax.set_title("Q-Q Plot")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_convergence_diagnostics(
    samples: Float[Array, "n_samples n_chains ..."],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot comprehensive convergence diagnostics.

    **Arguments:**

    - `samples`: Array of samples (n_samples, n_chains, ...)
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, axes = create_figure(figsize=figsize or get_config().large_figsize, nrows=2, ncols=2)
    axes = axes.flatten()

    n_samples, n_chains = samples.shape[:2]

    # 1. Trace plots
    for chain in range(n_chains):
        axes[0].plot(samples[:, chain, 0], alpha=0.7, label=f"Chain {chain}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Trace Plot")
    axes[0].legend()
    apply_standard_formatting(axes[0])

    # 2. Running mean
    for chain in range(n_chains):
        running_mean = jnp.cumsum(samples[:, chain, 0]) / jnp.arange(1, n_samples + 1)
        axes[1].plot(running_mean, alpha=0.7, label=f"Chain {chain}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Running Mean")
    axes[1].set_title("Mean Convergence")
    axes[1].legend()
    apply_standard_formatting(axes[1])

    # 3. Autocorrelation
    mean = jnp.mean(samples[:, 0, 0])
    var = jnp.var(samples[:, 0, 0])
    max_lag = min(50, n_samples // 2)

    autocorr = []
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            cov = jnp.mean((samples[:-lag, 0, 0] - mean) * (samples[lag:, 0, 0] - mean))
            autocorr.append(cov / var)

    axes[2].plot(range(max_lag), autocorr, linewidth=2)
    axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_title("Autocorrelation")
    apply_standard_formatting(axes[2])

    # 4. Density comparison
    for chain in range(n_chains):
        axes[3].hist(samples[:, chain, 0], bins=30, alpha=0.5, density=True, label=f"Chain {chain}")
    axes[3].set_xlabel("Value")
    axes[3].set_ylabel("Density")
    axes[3].set_title("Marginal Distributions")
    axes[3].legend()
    apply_standard_formatting(axes[3])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, axes


def plot_rhat(
    rhat_values: Float[Array, "n_parameters"],
    parameter_names: Optional[List[str]] = None,
    threshold: float = 1.1,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot R-hat convergence diagnostics.

    **Arguments:**

    - `rhat_values`: R-hat values for each parameter
    - `parameter_names`: Optional parameter names
    - `threshold`: Threshold for convergence (default 1.1)
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_params = len(rhat_values)
    if parameter_names is None:
        parameter_names = [f"Param {i}" for i in range(n_params)]

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    colors = [get_config().accent_color if r < threshold else get_config().error_color for r in rhat_values]

    ax.bar(range(n_params), rhat_values, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("R-hat")
    ax.set_title("R-hat Convergence Diagnostic")
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, ax


def plot_effective_sample_size(
    ess_values: Float[Array, "n_parameters"],
    n_samples: int,
    parameter_names: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot effective sample size diagnostics.

    **Arguments:**

    - `ess_values`: Effective sample sizes for each parameter
    - `n_samples`: Total number of samples
    - `parameter_names`: Optional parameter names
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_params = len(ess_values)
    if parameter_names is None:
        parameter_names = [f"Param {i}" for i in range(n_params)]

    fig, (ax1, ax2) = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=2)

    # Absolute ESS
    ax1.bar(range(n_params), ess_values, color=get_config().primary_color, alpha=0.7)
    ax1.axhline(y=n_samples, color="red", linestyle="--", linewidth=2, label=f"Total Samples ({n_samples})")
    ax1.set_xlabel("Parameter")
    ax1.set_ylabel("Effective Sample Size")
    ax1.set_title("Effective Sample Size")
    ax1.set_xticks(range(n_params))
    ax1.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax1.legend()
    apply_standard_formatting(ax1)

    # Relative ESS
    relative_ess = ess_values / n_samples
    ax2.bar(range(n_params), relative_ess, color=get_config().secondary_color, alpha=0.7)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Perfect (1.0)")
    ax2.set_xlabel("Parameter")
    ax2.set_ylabel("Relative ESS")
    ax2.set_title("Relative Effective Sample Size")
    ax2.set_xticks(range(n_params))
    ax2.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    apply_standard_formatting(ax2)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="distributions", close=False)

    return fig, (ax1, ax2)


def plot_scatter_with_regression(
    x: Float[Array, "n"],
    y: Float[Array, "n"],
    x_label: str = "X",
    y_label: str = "Y",
    title: Optional[str] = None,
    show_stats: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot scatter plot with regression line and statistics.

    **Arguments:**

    - `x`: X-axis values
    - `y`: Y-axis values
    - `x_label`: Label for x-axis
    - `y_label`: Label for y-axis
    - `title`: Plot title
    - `show_stats`: Whether to show statistics on plot
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes

    **Example:**
    ```python
    # Analyze relationship between free energy and reward
    fig, ax = plot_scatter_with_regression(
        free_energies, rewards,
        x_label="Free Energy",
        y_label="Reward",
        title="FE vs Reward",
        save_path="fe_reward_regression.png"
    )
    ```
    """
    if not HAS_STATS:
        raise ImportError("Statistical analysis module not available. Install scipy.")

    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50, color=get_config().primary_color, edgecolors="black", linewidth=0.5)

    # Compute regression
    reg_results = linear_regression(x, y)

    # Plot regression line
    x_line = jnp.linspace(jnp.min(x), jnp.max(x), 100)
    y_line = reg_results.intercept + reg_results.slope * x_line
    ax.plot(x_line, y_line, "r--", linewidth=2, label="Regression Line")

    # Plot confidence band (approximate using residual std)
    residual_std = jnp.std(reg_results.residuals)
    y_upper = y_line + 1.96 * residual_std
    y_lower = y_line - 1.96 * residual_std
    ax.fill_between(x_line, y_lower, y_upper, alpha=0.2, color="red", label="95% CI")

    # Add statistics text
    if show_stats:
        stats_text = (
            f"y = {reg_results.intercept:.3f} + {reg_results.slope:.3f}x\n"
            f"R² = {reg_results.r_squared:.3f}\n"
            f"p = {reg_results.p_value:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_correlation_matrix(
    data: Dict[str, Float[Array, "n"]],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot correlation matrix heatmap.

    **Arguments:**

    - `data`: Dictionary mapping variable names to data arrays
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes

    **Example:**
    ```python
    data = {
        'Free Energy': fe_values,
        'Reward': rewards,
        'Entropy': entropies,
        'Precision': precisions,
    }
    fig, ax = plot_correlation_matrix(data, save_path="correlation_matrix.png")
    ```
    """
    if not HAS_STATS:
        raise ImportError("Statistical analysis module not available. Install scipy.")

    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    # Compute correlation matrix
    var_names = list(data.keys())
    n_vars = len(var_names)
    corr_matrix = jnp.zeros((n_vars, n_vars))

    for i, name_i in enumerate(var_names):
        for j, name_j in enumerate(var_names):
            if i == j:
                corr_matrix = corr_matrix.at[i, j].set(1.0)
            else:
                corr_result = pearson_correlation(data[name_i], data[name_j], compute_significance=False)
                corr_matrix = corr_matrix.at[i, j].set(corr_result.correlation)

    # Create heatmap
    fig, ax = create_figure(figsize=figsize or (8, 8))

    sns.heatmap(
        np.asarray(corr_matrix),
        annot=True,
        fmt=".3f",
        xticklabels=var_names,
        yticklabels=var_names,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation Coefficient"},
    )

    ax.set_title("Correlation Matrix")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_residuals(
    x: Float[Array, "n"],
    y: Float[Array, "n"],
    x_label: str = "Predicted",
    y_label: str = "Residual",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot residual diagnostics for regression.

    **Arguments:**

    - `x`: X-axis values (typically predicted values)
    - `y`: Y-axis values (actual values)
    - `x_label`: Label for x-axis
    - `y_label`: Label for y-axis
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    if not HAS_STATS:
        raise ImportError("Statistical analysis module not available. Install scipy.")

    plt = ensure_matplotlib()

    # Compute regression
    reg_results = linear_regression(x, y)

    fig, axes = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=3)

    # 1. Residuals vs Fitted
    axes[0].scatter(reg_results.predictions, reg_results.residuals, alpha=0.6, s=50)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    apply_standard_formatting(axes[0])

    # 2. Q-Q plot
    from scipy import stats

    stats.probplot(np.asarray(reg_results.residuals), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")
    apply_standard_formatting(axes[1])

    # 3. Scale-Location
    standardized_residuals = reg_results.residuals / jnp.std(reg_results.residuals)
    sqrt_abs_std_resid = jnp.sqrt(jnp.abs(standardized_residuals))
    axes[2].scatter(reg_results.predictions, sqrt_abs_std_resid, alpha=0.6, s=50)
    axes[2].set_xlabel("Fitted Values")
    axes[2].set_ylabel("√|Standardized Residuals|")
    axes[2].set_title("Scale-Location")
    apply_standard_formatting(axes[2])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes


def plot_pairwise_relationships(
    data: Dict[str, Float[Array, "n"]],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot pairwise relationship matrix (scatter plots).

    **Arguments:**

    - `data`: Dictionary mapping variable names to data arrays
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes

    **Example:**
    ```python
    data = {
        'Free Energy': fe_values,
        'Reward': rewards,
        'Entropy': entropies,
    }
    fig, axes = plot_pairwise_relationships(data, save_path="pairwise.png")
    ```
    """
    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    # Prepare data for seaborn
    import pandas as pd

    # Find minimum length
    min_len = min(len(v) for v in data.values())

    # Create dataframe with truncated arrays
    df_data = {k: np.asarray(v[:min_len]) for k, v in data.items()}
    df = pd.DataFrame(df_data)

    # Create pair plot
    pairplot = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30})
    fig = pairplot.fig

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, pairplot.axes
