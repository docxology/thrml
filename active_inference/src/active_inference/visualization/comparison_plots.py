"""Comparison and multi-agent visualization functions."""

from typing import Dict, List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .core import apply_standard_formatting, create_figure, ensure_matplotlib, ensure_seaborn, get_config, save_figure


def plot_multi_agent_comparison(
    metrics_by_agent: Dict[str, Float[Array, "n_steps"]],
    metric_name: str = "Performance",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Compare performance of multiple agents.

    **Arguments:**

    - `metrics_by_agent`: Dict mapping agent names to metric arrays
    - `metric_name`: Name of the metric being plotted
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    for agent_name, metrics in metrics_by_agent.items():
        ax.plot(metrics, label=agent_name, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison Across Agents")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_learning_curves(
    curves: Dict[str, Float[Array, "n_episodes"]],
    labels: Optional[Dict[str, str]] = None,
    smooth_window: int = 10,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot learning curves with smoothing.

    **Arguments:**

    - `curves`: Dict mapping curve names to episode reward arrays
    - `labels`: Optional custom labels
    - `smooth_window`: Window size for smoothing
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    for name, curve in curves.items():
        label = labels.get(name, name) if labels else name

        # Plot raw curve
        ax.plot(curve, alpha=0.3, linewidth=1)

        # Plot smoothed curve
        if len(curve) >= smooth_window:
            smoothed = jnp.convolve(curve, jnp.ones(smooth_window) / smooth_window, mode="valid")
            ax.plot(range(smooth_window - 1, len(curve)), smoothed, label=label, linewidth=2)
        else:
            ax.plot(curve, label=label, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curves")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_parameter_sweep(
    param_values: Float[Array, "n_params"],
    metrics: Float[Array, "n_params"],
    param_name: str = "Parameter",
    metric_name: str = "Metric",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot results of parameter sweep.

    **Arguments:**

    - `param_values`: Array of parameter values tested
    - `metrics`: Corresponding metric values
    - `param_name`: Name of the parameter
    - `metric_name`: Name of the metric
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    ax.plot(param_values, metrics, "o-", linewidth=2, markersize=8, color=get_config().primary_color)

    # Mark best value
    best_idx = jnp.argmax(metrics)
    ax.plot(param_values[best_idx], metrics[best_idx], "r*", markersize=15, label=f"Best: {param_values[best_idx]:.3f}")

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs {param_name}")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_ablation_study(
    conditions: List[str],
    metrics: Float[Array, "n_conditions"],
    metric_name: str = "Performance",
    baseline_idx: int = 0,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot ablation study results.

    **Arguments:**

    - `conditions`: List of condition names
    - `metrics`: Metric values for each condition
    - `metric_name`: Name of the metric
    - `baseline_idx`: Index of baseline condition
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    # Color baseline differently
    colors = [
        get_config().accent_color if i == baseline_idx else get_config().primary_color for i in range(len(conditions))
    ]

    bars = ax.bar(range(len(conditions)), metrics, color=colors, alpha=0.7)

    # Highlight baseline
    bars[baseline_idx].set_edgecolor("red")
    bars[baseline_idx].set_linewidth(2)

    ax.set_xlabel("Condition")
    ax.set_ylabel(metric_name)
    ax.set_title("Ablation Study Results")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    # Add baseline reference line
    ax.axhline(y=metrics[baseline_idx], color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend()

    apply_standard_formatting(ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_hyperparameter_heatmap(
    param1_values: Float[Array, "n_param1"],
    param2_values: Float[Array, "n_param2"],
    metrics: Float[Array, "n_param1 n_param2"],
    param1_name: str = "Parameter 1",
    param2_name: str = "Parameter 2",
    metric_name: str = "Metric",
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot 2D hyperparameter search results as heatmap.

    **Arguments:**

    - `param1_values`: Values for first parameter
    - `param2_values`: Values for second parameter
    - `metrics`: Grid of metric values
    - `param1_name`: Name of first parameter
    - `param2_name`: Name of second parameter
    - `metric_name`: Name of the metric
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    im = ax.imshow(metrics, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label=metric_name)

    # Mark best configuration
    best_idx = jnp.unravel_index(jnp.argmax(metrics), metrics.shape)
    ax.plot(best_idx[1], best_idx[0], "r*", markersize=20, markeredgecolor="white", markeredgewidth=2)

    # Set ticks
    ax.set_xticks(range(len(param2_values)))
    ax.set_yticks(range(len(param1_values)))
    ax.set_xticklabels([f"{v:.2g}" for v in param2_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.2g}" for v in param1_values])

    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title(f"{metric_name} Across Hyperparameters")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_multi_metric_comparison(
    agents: List[str],
    metrics: Dict[str, Float[Array, "n_agents"]],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Compare multiple metrics across agents using grouped bar chart.

    **Arguments:**

    - `agents`: List of agent names
    - `metrics`: Dict of metric_name -> array of values per agent
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().large_figsize)

    n_agents = len(agents)
    n_metrics = len(metrics)

    # Set up bar positions
    x = jnp.arange(n_agents)
    width = 0.8 / n_metrics

    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - n_metrics / 2 + 0.5)
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Value")
    ax.set_title("Multi-Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.legend()
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_statistical_comparison(
    groups: Dict[str, Float[Array, "n_samples"]],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot statistical comparison with box plots and significance tests.

    **Arguments:**

    - `groups`: Dict mapping group names to sample arrays
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    sns = ensure_seaborn()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    # Prepare data for seaborn
    import pandas as pd

    data = []
    for group_name, samples in groups.items():
        for sample in samples:
            data.append({"Group": group_name, "Value": float(sample)})

    df = pd.DataFrame(data)

    # Box plot
    sns.boxplot(data=df, x="Group", y="Value", ax=ax, palette="Set2")

    # Add strip plot for individual points
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color="black", alpha=0.3, size=3)

    ax.set_title("Statistical Comparison")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax
