"""Environment-specific visualization functions."""

from typing import List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .core import apply_standard_formatting, create_figure, ensure_matplotlib, get_config, save_figure


def plot_grid_world(
    grid_shape: tuple[int, int],
    agent_pos: Optional[tuple[int, int]] = None,
    goal_pos: Optional[tuple[int, int]] = None,
    obstacles: Optional[List[tuple[int, int]]] = None,
    values: Optional[Float[Array, "height width"]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Visualize a grid world environment.

    **Arguments:**

    - `grid_shape`: (height, width) of the grid
    - `agent_pos`: Agent position (row, col)
    - `goal_pos`: Goal position (row, col)
    - `obstacles`: List of obstacle positions
    - `values`: Optional value map to display as heatmap
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    height, width = grid_shape

    # Create base grid
    if values is not None:
        im = ax.imshow(values, cmap="Blues", alpha=0.6, origin="upper")
        plt.colorbar(im, ax=ax, label="Value")
    else:
        ax.imshow(jnp.zeros((height, width)), cmap="gray", alpha=0.1, origin="upper")

    # Draw obstacles
    if obstacles:
        for obs_row, obs_col in obstacles:
            ax.add_patch(plt.Rectangle((obs_col - 0.5, obs_row - 0.5), 1, 1, fill=True, color="black", alpha=0.8))

    # Draw goal
    if goal_pos:
        goal_row, goal_col = goal_pos
        ax.add_patch(plt.Circle((goal_col, goal_row), 0.3, color="gold", alpha=0.8, zorder=5))
        ax.text(goal_col, goal_row, "G", ha="center", va="center", fontsize=12, weight="bold", color="red")

    # Draw agent
    if agent_pos:
        agent_row, agent_col = agent_pos
        ax.add_patch(plt.Circle((agent_col, agent_row), 0.3, color=get_config().accent_color, alpha=0.8, zorder=6))
        ax.text(agent_col, agent_row, "A", ha="center", va="center", fontsize=12, weight="bold", color="white")

    # Grid lines
    ax.set_xticks(jnp.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(jnp.arange(0, width, 1))
    ax.set_yticks(jnp.arange(0, height, 1))

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Grid World")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_tmaze(
    agent_pos: int,
    cue: Optional[str] = None,
    reward_left: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Visualize a T-maze environment.

    **Arguments:**

    - `agent_pos`: Agent position (0-3: bottom, center, left arm, right arm)
    - `cue`: Cue presented ('left' or 'right')
    - `reward_left`: Whether reward is on the left side
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().small_figsize)

    # Define T-maze structure
    # Draw maze
    ax.plot([0.5, 0.5], [0, 1], "k-", linewidth=3)  # Vertical stem
    ax.plot([0, 1], [1, 1], "k-", linewidth=3)  # Horizontal top

    # Position markers
    positions = {
        0: (0.5, 0.25),  # Bottom
        1: (0.5, 0.75),  # Center
        2: (0.1, 1.0),  # Left arm
        3: (0.9, 1.0),  # Right arm
    }

    # Draw agent
    agent_x, agent_y = positions[agent_pos]
    ax.add_patch(plt.Circle((agent_x, agent_y), 0.08, color=get_config().accent_color, alpha=0.8, zorder=5))
    ax.text(agent_x, agent_y, "A", ha="center", va="center", fontsize=10, weight="bold", color="white")

    # Draw reward
    reward_pos = 2 if reward_left else 3
    reward_x, reward_y = positions[reward_pos]
    ax.add_patch(plt.Circle((reward_x, reward_y), 0.06, color="gold", alpha=0.8, zorder=4))
    ax.text(reward_x, reward_y - 0.15, "R", ha="center", fontsize=10, weight="bold", color="orange")

    # Draw cue if present
    if cue:
        cue_text = f"Cue: {cue.upper()}"
        ax.text(
            0.5,
            0.05,
            cue_text,
            ha="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("T-Maze")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_agent_trajectory(
    positions: List[tuple[int, int]],
    grid_shape: tuple[int, int],
    obstacles: Optional[List[tuple[int, int]]] = None,
    goal_pos: Optional[tuple[int, int]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot agent trajectory through environment.

    **Arguments:**

    - `positions`: List of (row, col) positions over time
    - `grid_shape`: (height, width) of the grid
    - `obstacles`: Optional obstacle positions
    - `goal_pos`: Optional goal position
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    height, width = grid_shape

    # Base grid
    ax.imshow(jnp.zeros((height, width)), cmap="gray", alpha=0.1, origin="upper")

    # Draw obstacles
    if obstacles:
        for obs_row, obs_col in obstacles:
            ax.add_patch(plt.Rectangle((obs_col - 0.5, obs_row - 0.5), 1, 1, fill=True, color="black", alpha=0.5))

    # Draw goal
    if goal_pos:
        goal_row, goal_col = goal_pos
        ax.add_patch(plt.Circle((goal_col, goal_row), 0.3, color="gold", alpha=0.6, zorder=3))

    # Draw trajectory
    if positions:
        rows, cols = zip(*positions)
        ax.plot(
            cols, rows, "o-", linewidth=2, markersize=6, color=get_config().primary_color, alpha=0.6, label="Trajectory"
        )

        # Mark start and end
        ax.plot(cols[0], rows[0], "o", markersize=12, color="green", label="Start", zorder=5)
        ax.plot(cols[-1], rows[-1], "*", markersize=15, color="red", label="End", zorder=5)

    # Grid lines
    ax.set_xticks(jnp.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Agent Trajectory")
    ax.legend()

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="trajectories", close=False)

    return fig, ax


def plot_observation_history(
    observations: Float[Array, "n_steps n_obs_features"],
    obs_labels: Optional[List[str]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot observation history over time.

    **Arguments:**

    - `observations`: Array of observations
    - `obs_labels`: Optional labels for observation features
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    n_steps, n_features = observations.shape

    if obs_labels is None:
        obs_labels = [f"Feature {i}" for i in range(n_features)]

    # Plot as heatmap if many features, otherwise as lines
    if n_features > 10:
        im = ax.imshow(observations.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set_ylabel("Observation Feature")
        ax.set_yticks(range(min(10, n_features)))
        ax.set_yticklabels(obs_labels[:10])
        plt.colorbar(im, ax=ax, label="Value")
    else:
        for i, label in enumerate(obs_labels):
            ax.plot(observations[:, i], label=label, linewidth=2, alpha=0.7)
        ax.set_ylabel("Observation Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel("Time Step")
    ax.set_title("Observation History")
    apply_standard_formatting(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax


def plot_reward_over_time(
    rewards: Float[Array, "n_steps"],
    cumulative: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot reward over time.

    **Arguments:**

    - `rewards`: Array of rewards per timestep
    - `cumulative`: Whether to show cumulative reward
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, axes = create_figure(figsize=figsize or get_config().wide_figsize, nrows=1, ncols=2 if cumulative else 1)

    if not cumulative:
        axes = [axes]

    # Instantaneous reward
    axes[0].plot(rewards, linewidth=2, color=get_config().primary_color)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Instantaneous Reward")
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    apply_standard_formatting(axes[0])

    # Cumulative reward
    if cumulative:
        cumsum = jnp.cumsum(rewards)
        axes[1].plot(cumsum, linewidth=2, color=get_config().accent_color)
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Cumulative Reward")
        axes[1].set_title("Cumulative Reward")
        apply_standard_formatting(axes[1])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, axes if cumulative else (fig, axes[0])


def plot_heatmap_occupancy(
    occupancy: Float[Array, "height width"],
    obstacles: Optional[List[tuple[int, int]]] = None,
    goal_pos: Optional[tuple[int, int]] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot heatmap of state occupancy in grid world.

    **Arguments:**

    - `occupancy`: Occupancy count or probability for each grid cell
    - `obstacles`: Optional obstacle positions
    - `goal_pos`: Optional goal position
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize)

    height, width = occupancy.shape

    # Normalize occupancy
    norm_occupancy = occupancy / jnp.max(occupancy) if jnp.max(occupancy) > 0 else occupancy

    # Plot heatmap
    im = ax.imshow(norm_occupancy, cmap="hot", alpha=0.8, origin="upper")
    plt.colorbar(im, ax=ax, label="Occupancy")

    # Overlay obstacles
    if obstacles:
        for obs_row, obs_col in obstacles:
            ax.add_patch(plt.Rectangle((obs_col - 0.5, obs_row - 0.5), 1, 1, fill=True, color="blue", alpha=0.5))

    # Mark goal
    if goal_pos:
        goal_row, goal_col = goal_pos
        ax.plot(goal_col, goal_row, "*", markersize=20, color="yellow", markeredgecolor="black", markeredgewidth=2)

    # Grid lines
    ax.set_xticks(jnp.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_title("State Occupancy Heatmap")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="plots", close=False)

    return fig, ax
