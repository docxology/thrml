"""Active inference specific visualization functions."""

from typing import List, Optional, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from .core import apply_standard_formatting, create_figure, ensure_matplotlib, get_config, save_figure


def plot_belief_trajectory(
    beliefs: Union[List[Float[Array, "n_states"]], Float[Array, "n_steps n_states"]],
    true_states: Optional[List[int]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_entropy: bool = True,
    **kwargs,
):
    """Plot belief trajectory over time with optional entropy.

    **Arguments:**

    - `beliefs`: List of belief distributions or array of shape (n_steps, n_states)
    - `true_states`: Optional list of true state indices
    - `figsize`: Figure size (default from config)
    - `save_path`: If provided, save figure to this path
    - `show_entropy`: Whether to show entropy subplot
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes objects
    """
    plt = ensure_matplotlib()

    # Convert to array if list
    if isinstance(beliefs, list):
        beliefs_array = jnp.stack(beliefs)
    else:
        beliefs_array = beliefs

    n_steps, n_states = beliefs_array.shape

    if show_entropy:
        fig, (ax1, ax2) = create_figure(figsize=figsize, nrows=1, ncols=2)
    else:
        fig, ax1 = create_figure(figsize=figsize, nrows=1, ncols=1)
        ax2 = None

    # Plot belief heatmap
    im = ax1.imshow(beliefs_array.T, aspect="auto", cmap=kwargs.get("cmap", "viridis"), origin="lower")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    ax1.set_title("Belief Distribution Over Time")
    plt.colorbar(im, ax=ax1, label="Probability")

    # Plot true states if available
    if true_states is not None:
        ax1.plot(range(len(true_states)), true_states, "r--", linewidth=2, label="True State", alpha=0.8)
        ax1.legend()

    apply_standard_formatting(ax1)

    # Plot belief entropy
    if show_entropy and ax2 is not None:
        entropies = []
        for belief in beliefs_array:
            b = belief + 1e-16
            b = b / jnp.sum(b)
            entropy = -jnp.sum(b * jnp.log(b))
            entropies.append(float(entropy))

        ax2.plot(entropies, linewidth=2, color=get_config().primary_color)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Entropy (nats)")
        ax2.set_title("Belief Entropy Over Time")
        ax2.axhline(y=jnp.log(n_states), color="red", linestyle="--", alpha=0.5, label="Max Entropy")
        ax2.legend()
        apply_standard_formatting(ax2)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, (ax1, ax2) if show_entropy else (fig, ax1)


def plot_belief_evolution(
    beliefs_history: List[Float[Array, "n_states"]],
    state_labels: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot evolution of individual state beliefs over time.

    **Arguments:**

    - `beliefs_history`: List of belief distributions
    - `state_labels`: Optional labels for each state
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    beliefs_array = jnp.stack(beliefs_history)
    n_steps, n_states = beliefs_array.shape

    if state_labels is None:
        state_labels = [f"State {i}" for i in range(n_states)]

    fig, ax = create_figure(figsize=figsize)

    # Plot each state's belief trajectory
    for i in range(n_states):
        ax.plot(beliefs_array[:, i], label=state_labels[i], linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Belief Probability")
    ax.set_title("State Belief Evolution")
    ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_free_energy(
    free_energies: Float[Array, "n_steps"],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_moving_average: bool = True,
    window: int = 10,
    **kwargs,
):
    """Plot variational free energy over time.

    **Arguments:**

    - `free_energies`: Array of free energy values
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `show_moving_average`: Whether to show moving average
    - `window`: Window size for moving average
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    ax.plot(free_energies, linewidth=2, color=get_config().primary_color, label="Free Energy")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Variational Free Energy")
    ax.set_title("Free Energy Over Time")

    # Add moving average
    if show_moving_average and len(free_energies) > window:
        moving_avg = jnp.convolve(
            free_energies,
            jnp.ones(window) / window,
            mode="valid",
        )
        ax.plot(
            range(window - 1, len(free_energies)),
            moving_avg,
            "--",
            linewidth=2,
            color=get_config().secondary_color,
            label=f"Moving Avg (window={window})",
        )
        ax.legend()

    apply_standard_formatting(ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_free_energy_components(
    vfe: Float[Array, "n_steps"],
    accuracy: Optional[Float[Array, "n_steps"]] = None,
    complexity: Optional[Float[Array, "n_steps"]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot free energy decomposed into accuracy and complexity.

    **Arguments:**

    - `vfe`: Variational free energy values
    - `accuracy`: Accuracy term (negative log likelihood)
    - `complexity`: Complexity term (KL divergence)
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, axes = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=3)

    # VFE
    axes[0].plot(vfe, linewidth=2, color=get_config().primary_color)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("VFE")
    axes[0].set_title("Variational Free Energy")
    apply_standard_formatting(axes[0])

    # Accuracy
    if accuracy is not None:
        axes[1].plot(accuracy, linewidth=2, color=get_config().accent_color)
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy (Neg. Log Likelihood)")
        apply_standard_formatting(axes[1])

    # Complexity
    if complexity is not None:
        axes[2].plot(complexity, linewidth=2, color=get_config().secondary_color)
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Complexity")
        axes[2].set_title("Complexity (KL Divergence)")
        apply_standard_formatting(axes[2])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes


def plot_expected_free_energy(
    efe_per_policy: Float[Array, "n_policies"],
    policy_labels: Optional[List[str]] = None,
    selected_policy: Optional[int] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot expected free energy for different policies.

    **Arguments:**

    - `efe_per_policy`: EFE values for each policy
    - `policy_labels`: Optional labels for policies
    - `selected_policy`: Index of selected policy to highlight
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_policies = len(efe_per_policy)

    if policy_labels is None:
        policy_labels = [f"Policy {i}" for i in range(n_policies)]

    fig, ax = create_figure(figsize=figsize)

    colors = [
        get_config().accent_color if i == selected_policy else get_config().primary_color for i in range(n_policies)
    ]

    bars = ax.bar(range(n_policies), efe_per_policy, color=colors, alpha=0.7)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Expected Free Energy")
    ax.set_title("Expected Free Energy by Policy")
    ax.set_xticks(range(n_policies))
    ax.set_xticklabels(policy_labels, rotation=45, ha="right")

    if selected_policy is not None:
        bars[selected_policy].set_edgecolor("red")
        bars[selected_policy].set_linewidth(2)

    apply_standard_formatting(ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_action_distribution(
    action_probs: Float[Array, "n_actions"],
    action_names: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot action probability distribution.

    **Arguments:**

    - `action_probs`: Probability distribution over actions
    - `action_names`: Optional names for actions
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_actions = len(action_probs)
    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]

    fig, ax = create_figure(figsize=figsize or get_config().small_figsize)

    ax.bar(action_names, action_probs, color=get_config().primary_color, alpha=0.7)
    ax.set_ylabel("Probability")
    ax.set_title("Action Distribution")
    ax.set_ylim([0, 1])
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_policy_selection(
    policy_history: List[int],
    n_policies: Optional[int] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot policy selection over time.

    **Arguments:**

    - `policy_history`: List of selected policy indices
    - `n_policies`: Total number of policies
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    if n_policies is None:
        n_policies = max(policy_history) + 1

    fig, (ax1, ax2) = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=2)

    # Timeline of policy selection
    ax1.plot(policy_history, marker="o", linestyle="-", linewidth=2, markersize=4)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Selected Policy")
    ax1.set_title("Policy Selection Over Time")
    ax1.set_yticks(range(n_policies))
    apply_standard_formatting(ax1)

    # Histogram of policy usage
    ax2.hist(
        policy_history,
        bins=range(n_policies + 1),
        align="left",
        color=get_config().primary_color,
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_xlabel("Policy")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Policy Usage Distribution")
    ax2.set_xticks(range(n_policies))
    apply_standard_formatting(ax2)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, (ax1, ax2)


def plot_precision_trajectory(
    precisions: Union[Float[Array, "n_steps"], dict],
    labels: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot precision parameters over time.

    **Arguments:**

    - `precisions`: Array of precision values or dict of precision types
    - `labels`: Optional labels for different precision types
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    if isinstance(precisions, dict):
        for i, (name, values) in enumerate(precisions.items()):
            ax.plot(values, label=name, linewidth=2, alpha=0.8)
        ax.legend()
    else:
        ax.plot(precisions, linewidth=2, color=get_config().primary_color)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Parameters Over Time")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_agent_performance(
    rewards: Optional[Float[Array, "n_steps"]] = None,
    free_energies: Optional[Float[Array, "n_steps"]] = None,
    entropies: Optional[Float[Array, "n_steps"]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot comprehensive agent performance metrics.

    **Arguments:**

    - `rewards`: Reward values over time
    - `free_energies`: Free energy values
    - `entropies`: Entropy values
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_plots = sum([rewards is not None, free_energies is not None, entropies is not None])

    if n_plots == 0:
        raise ValueError("At least one metric must be provided")

    fig, axes = create_figure(figsize=figsize or get_config().large_figsize, nrows=1, ncols=n_plots)

    if n_plots == 1:
        axes = [axes]

    idx = 0

    if rewards is not None:
        axes[idx].plot(rewards, linewidth=2, color=get_config().accent_color)
        axes[idx].set_xlabel("Time Step")
        axes[idx].set_ylabel("Reward")
        axes[idx].set_title("Cumulative Reward")
        apply_standard_formatting(axes[idx])
        idx += 1

    if free_energies is not None:
        axes[idx].plot(free_energies, linewidth=2, color=get_config().primary_color)
        axes[idx].set_xlabel("Time Step")
        axes[idx].set_ylabel("Free Energy")
        axes[idx].set_title("Free Energy")
        apply_standard_formatting(axes[idx])
        idx += 1

    if entropies is not None:
        axes[idx].plot(entropies, linewidth=2, color=get_config().secondary_color)
        axes[idx].set_xlabel("Time Step")
        axes[idx].set_ylabel("Entropy")
        axes[idx].set_title("Belief Entropy")
        apply_standard_formatting(axes[idx])
        idx += 1

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes


def plot_state_occupancy(
    state_visits: Float[Array, "n_states"],
    state_labels: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot state occupancy/visitation frequency.

    **Arguments:**

    - `state_visits`: Count or frequency of state visits
    - `state_labels`: Optional labels for states
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    n_states = len(state_visits)
    if state_labels is None:
        state_labels = [f"State {i}" for i in range(n_states)]

    fig, ax = create_figure(figsize=figsize)

    # Normalize to probabilities
    state_probs = state_visits / jnp.sum(state_visits)

    ax.bar(range(n_states), state_probs, color=get_config().primary_color, alpha=0.7)
    ax.set_xlabel("State")
    ax.set_ylabel("Occupancy Probability")
    ax.set_title("State Occupancy Distribution")
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(state_labels, rotation=45, ha="right")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax
