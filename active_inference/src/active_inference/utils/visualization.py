"""Visualization utilities for active inference."""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float


def plot_belief_trajectory(
    beliefs: list[Float[Array, "n_states"]],
    true_states: Optional[list[int]] = None,
    figsize: tuple[int, int] = (12, 6),
):
    """Plot belief trajectory over time.

    **Arguments:**

    - `beliefs`: List of belief distributions over time
    - `true_states`: Optional list of true state indices
    - `figsize`: Figure size

    **Returns:**

    - Matplotlib figure and axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    beliefs_array = jnp.stack(beliefs)
    n_steps, n_states = beliefs_array.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot belief heatmap
    im = ax1.imshow(beliefs_array.T, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    ax1.set_title("Belief Trajectory")
    plt.colorbar(im, ax=ax1)

    # Plot true states if available
    if true_states is not None:
        ax1.plot(range(len(true_states)), true_states, "r--", linewidth=2, label="True State")
        ax1.legend()

    # Plot belief entropy over time
    entropies = []
    for belief in beliefs:
        b = belief + 1e-16
        b = b / jnp.sum(b)
        entropy = -jnp.sum(b * jnp.log(b))
        entropies.append(float(entropy))

    ax2.plot(entropies, linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Belief Entropy")
    ax2.set_title("Uncertainty Over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_free_energy(
    free_energies: Float[Array, "n_steps"],
    figsize: tuple[int, int] = (10, 6),
):
    """Plot free energy over time.

    **Arguments:**

    - `free_energies`: Array of free energy values
    - `figsize`: Figure size

    **Returns:**

    - Matplotlib figure and axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(free_energies, linewidth=2, color="darkblue")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Variational Free Energy")
    ax.set_title("Free Energy Over Time")
    ax.grid(True, alpha=0.3)

    # Add moving average
    if len(free_energies) > 10:
        window = 10
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
            color="red",
            label=f"Moving Avg (window={window})",
        )
        ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_action_distribution(
    action_probs: Float[Array, "n_actions"],
    action_names: Optional[list[str]] = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Plot action probability distribution.

    **Arguments:**

    - `action_probs`: Probability distribution over actions
    - `action_names`: Optional names for actions
    - `figsize`: Figure size

    **Returns:**

    - Matplotlib figure and axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    n_actions = len(action_probs)
    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(action_names, action_probs, color="steelblue", alpha=0.7)
    ax.set_ylabel("Probability")
    ax.set_title("Action Distribution")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, ax
