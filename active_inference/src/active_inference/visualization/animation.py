"""Animation utilities for dynamic visualization."""

from typing import List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from .core import create_figure, ensure_matplotlib, get_config


def create_belief_animation(
    beliefs: Float[Array, "n_steps n_states"],
    true_states: Optional[List[int]] = None,
    interval: int = 100,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Create animation of belief evolution over time.

    **Arguments:**

    - `beliefs`: Array of beliefs (n_steps, n_states)
    - `true_states`: Optional true state sequence
    - `interval`: Milliseconds between frames
    - `figsize`: Figure size
    - `save_path`: If provided, save animation to this path
    - `**kwargs`: Additional animation options

    **Returns:**

    - Animation object
    """
    plt = ensure_matplotlib()

    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib.animation required for animations")

    fig, ax = create_figure(figsize=figsize)

    n_steps, n_states = beliefs.shape

    # Initialize plot
    bars = ax.bar(range(n_states), beliefs[0], color=get_config().primary_color, alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_xlabel("State")
    ax.set_ylabel("Belief Probability")
    ax.set_title("Belief Evolution - Step 0")

    # Add true state marker if available
    true_state_line = None
    if true_states:
        true_state_line = ax.axvline(x=true_states[0], color="red", linestyle="--", linewidth=2, label="True State")
        ax.legend()

    def update(frame):
        # Update bars
        for bar, height in zip(bars, beliefs[frame]):
            bar.set_height(height)

        # Update true state marker
        if true_states and true_state_line:
            true_state_line.set_xdata([true_states[frame], true_states[frame]])

        ax.set_title(f"Belief Evolution - Step {frame}")
        return bars

    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=False)

    if save_path:
        output_path = get_config().get_output_path("animations", save_path)

        try:
            if save_path.endswith(".gif"):
                anim.save(output_path, writer="pillow", fps=1000 // interval)
            else:
                anim.save(output_path, writer="ffmpeg", fps=1000 // interval)
            print(f"Saved animation to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
            print("You may need to install ffmpeg or pillow for animation export")

    return anim


def create_trajectory_animation(
    positions: List[tuple[int, int]],
    grid_shape: tuple[int, int],
    obstacles: Optional[List[tuple[int, int]]] = None,
    goal_pos: Optional[tuple[int, int]] = None,
    interval: int = 200,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Create animation of agent trajectory through environment.

    **Arguments:**

    - `positions`: List of (row, col) positions
    - `grid_shape`: (height, width) of grid
    - `obstacles`: Optional obstacle positions
    - `goal_pos`: Optional goal position
    - `interval`: Milliseconds between frames
    - `figsize`: Figure size
    - `save_path`: If provided, save animation
    - `**kwargs`: Additional animation options

    **Returns:**

    - Animation object
    """
    plt = ensure_matplotlib()

    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib.animation required for animations")

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

    # Initialize agent
    agent_circle = plt.Circle(positions[0][::-1], 0.3, color=get_config().accent_color, alpha=0.8, zorder=5)
    ax.add_patch(agent_circle)

    # Trail
    (trail_line,) = ax.plot([], [], "o-", linewidth=2, markersize=4, color=get_config().primary_color, alpha=0.4)

    # Grid lines
    ax.set_xticks(jnp.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(jnp.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Agent Trajectory - Step 0")

    def update(frame):
        row, col = positions[frame]

        # Update agent position
        agent_circle.center = (col, row)

        # Update trail
        trail_rows, trail_cols = zip(*positions[: frame + 1])
        trail_line.set_data(trail_cols, trail_rows)

        ax.set_title(f"Agent Trajectory - Step {frame}")

        return agent_circle, trail_line

    anim = FuncAnimation(fig, update, frames=len(positions), interval=interval, blit=False)

    if save_path:
        output_path = get_config().get_output_path("animations", save_path)

        try:
            if save_path.endswith(".gif"):
                anim.save(output_path, writer="pillow", fps=1000 // interval)
            else:
                anim.save(output_path, writer="ffmpeg", fps=1000 // interval)
            print(f"Saved animation to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")

    return anim


def create_sampling_animation(
    samples: Float[Array, "n_samples n_dims"],
    dims: tuple[int, int] = (0, 1),
    interval: int = 50,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Create animation of sampling trajectory.

    **Arguments:**

    - `samples`: Array of samples (n_samples, n_dimensions)
    - `dims`: Which dimensions to animate
    - `interval`: Milliseconds between frames
    - `figsize`: Figure size
    - `save_path`: If provided, save animation
    - `**kwargs`: Additional animation options

    **Returns:**

    - Animation object
    """
    plt = ensure_matplotlib()

    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise ImportError("matplotlib.animation required for animations")

    fig, ax = create_figure(figsize=figsize)

    # Set up plot limits
    x_min, x_max = jnp.min(samples[:, dims[0]]), jnp.max(samples[:, dims[0]])
    y_min, y_max = jnp.min(samples[:, dims[1]]), jnp.max(samples[:, dims[1]])

    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Initialize plots
    (trail_line,) = ax.plot([], [], alpha=0.3, linewidth=0.5, color=get_config().primary_color)
    scatter = ax.scatter([], [], s=20, color=get_config().accent_color, alpha=0.7)
    (current_point,) = ax.plot([], [], "ro", markersize=10)

    ax.set_xlabel(f"Dimension {dims[0]}")
    ax.set_ylabel(f"Dimension {dims[1]}")
    ax.set_title("Sampling Trajectory - Sample 0")
    ax.grid(True, alpha=0.3)

    def update(frame):
        # Update trail
        trail_line.set_data(samples[: frame + 1, dims[0]], samples[: frame + 1, dims[1]])

        # Update scatter (show recent samples)
        window = min(100, frame + 1)
        scatter.set_offsets(samples[max(0, frame - window) : frame + 1, dims])

        # Update current point
        current_point.set_data([samples[frame, dims[0]]], [samples[frame, dims[1]]])

        ax.set_title(f"Sampling Trajectory - Sample {frame}")

        return trail_line, scatter, current_point

    anim = FuncAnimation(fig, update, frames=len(samples), interval=interval, blit=False)

    if save_path:
        output_path = get_config().get_output_path("animations", save_path)

        try:
            if save_path.endswith(".gif"):
                anim.save(output_path, writer="pillow", fps=1000 // interval)
            else:
                anim.save(output_path, writer="ffmpeg", fps=1000 // interval)
            print(f"Saved animation to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")

    return anim
