"""Example 15: Grid world navigation with active inference agent (formerly Example 02).

Demonstrates:
- Building a grid world environment
- Creating an active inference agent
- Running perception-action cycles
- Evaluating agent performance
- Visualizing agent behavior and trajectories

**THRML Integration**:
- Agent uses THRML sampling-based inference internally (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for efficient perception
- Energy-efficient sampling for agent state inference
"""

import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

# Import example utilities
from example_utils import ExampleRunner, create_figure, plot_line_series

from active_inference.agents import ActiveInferenceAgent
from active_inference.core import Precision
from active_inference.environments import GridWorld, GridWorldConfig
from active_inference.inference import ThrmlInferenceEngine
from active_inference.models import build_grid_world_model

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "15_grid_world_agent"


def main():
    """Run grid world navigation example."""
    # Initialize example runner
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    grid_size = runner.get_config("grid_size", default=5)
    n_observations = runner.get_config("n_observations", default=10)
    observation_noise = runner.get_config("observation_noise", default=0.1)
    goal_location = tuple(runner.get_config("goal_location", default=[3, 3]))
    obstacle_locations = [tuple(loc) for loc in runner.get_config("obstacle_locations", default=[[1, 1], [1, 2]])]
    max_steps = runner.get_config("max_steps", default=50)
    planning_horizon = runner.get_config("planning_horizon", default=3)
    sensory_precision = runner.get_config("sensory_precision", default=1.0)
    state_precision = runner.get_config("state_precision", default=1.0)
    action_precision = runner.get_config("action_precision", default=2.0)
    goal_preference_strength = runner.get_config("goal_preference_strength", default=3.0)
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  grid_size: {grid_size}")
    runner.logger.info(f"  planning_horizon: {planning_horizon}")

    # === 1. CREATE ENVIRONMENT ===
    runner.logger.info("Creating grid world environment")

    # Create grid world from config
    config = GridWorldConfig(
        size=grid_size,
        n_observations=n_observations,
        observation_noise=observation_noise,
        goal_location=goal_location,
        obstacle_locations=obstacle_locations,
    )

    runner.logger.info(f"Environment: {config.size}x{config.size} grid")
    runner.logger.info(f"Goal: {config.goal_location}")
    runner.logger.info(f"Obstacles: {config.obstacle_locations}")

    # Save environment configuration
    env_config = {
        "seed": seed,
        "grid_size": config.size,
        "n_observations": config.n_observations,
        "observation_noise": config.observation_noise,
        "goal_location": config.goal_location,
        "obstacle_locations": config.obstacle_locations,
    }
    runner.save_config(env_config, "environment_config.json")

    # Create environment
    env = GridWorld(config=config)

    # === 2. BUILD GENERATIVE MODEL ===
    runner.logger.info("Building generative model")

    model = build_grid_world_model(config, goal_preference_strength=goal_preference_strength)
    runner.logger.info(f"Model: {model.n_states} states, {model.n_observations} observations")

    # === 3. CREATE AGENT ===
    runner.logger.info("Creating active inference agent")

    precision = Precision(
        sensory_precision=sensory_precision,
        state_precision=state_precision,
        action_precision=action_precision,
    )

    # Create THRML inference engine
    thrml_engine = ThrmlInferenceEngine(
        model=model,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )

    agent = ActiveInferenceAgent(
        model=model,
        precision=precision,
        planning_horizon=planning_horizon,
        thrml_engine=thrml_engine,
    )

    runner.logger.info(f"Agent: planning horizon = {agent.planning_horizon}")
    runner.logger.info(f"Action precision: {precision.action_precision}")

    # Save agent configuration
    agent_config = {
        "planning_horizon": agent.planning_horizon,
        "sensory_precision": precision.sensory_precision,
        "state_precision": precision.state_precision,
        "action_precision": precision.action_precision,
        "goal_preference_strength": goal_preference_strength,
    }
    runner.save_config(agent_config, "agent_config.json")

    # === 4. RUN EPISODE ===
    runner.logger.info("\nRunning episode")

    agent_state = agent.reset()
    key_reset, key = jax.random.split(key)
    env, obs = env.reset(key_reset)

    runner.logger.info(f"Starting at: {tuple(env.current_state)}")
    runner.logger.info(f"Initial observation: {obs}")
    done = False
    total_reward = 0.0
    trajectory = []

    # Track data for analysis
    states = []
    actions = []
    rewards = []
    observations = []
    free_energies = []
    beliefs = []

    keys = jax.random.split(key, max_steps)

    for step in range(max_steps):
        if done:
            break

        # Agent perceives and acts
        key_step, key_env = jax.random.split(keys[step])
        action, agent_state, fe = agent.step(key_step, obs, agent_state)

        # Map action to direction
        action_names = ["↑ up", "→ right", "↓ down", "← left"]

        # Environment responds
        prev_state = tuple(env.current_state)
        env, obs, reward, done = env.step(key_env, action)
        curr_state = tuple(env.current_state)

        total_reward += reward

        # Store trajectory
        trajectory.append(
            {
                "step": step,
                "state": prev_state,
                "action": action_names[action],
                "next_state": curr_state,
                "reward": reward,
                "obs": obs,
                "free_energy": float(fe),
            }
        )

        # Store data
        states.append(prev_state)
        actions.append(int(action))
        rewards.append(float(reward))
        observations.append(int(obs))
        free_energies.append(float(fe))
        beliefs.append(np.array(agent_state.belief))

        # Log step
        runner.logger.info(
            f"Step {step:2d}: {prev_state} --{action_names[action]}--> {curr_state} "
            f"| obs={obs} reward={reward:.1f} FE={fe:.3f}"
        )

    runner.logger.info("\n=== Episode complete ===")
    runner.logger.info(f"Steps taken: {len(trajectory)}")
    runner.logger.info(f"Total reward: {total_reward:.1f}")
    runner.logger.info(f"Goal reached: {'Yes' if done else 'No'}")

    # === 5. SAVE TRAJECTORY DATA ===
    trajectory_data = {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "observations": np.array(observations),
        "free_energies": np.array(free_energies),
        "beliefs": np.array(beliefs),
        "total_reward": total_reward,
        "steps_taken": len(trajectory),
        "goal_reached": done,
    }
    runner.save_data(trajectory_data, "trajectory")

    # === 6. VISUALIZE RESULTS ===
    runner.logger.info("\nGenerating visualizations")

    # Plot 1: Trajectory in grid
    fig, axes = create_figure(2, 2, figsize=(12, 10))

    # Grid visualization
    ax = axes[0, 0]
    grid = np.zeros((config.size, config.size))

    # Mark obstacles
    for obs_loc in config.obstacle_locations:
        grid[obs_loc] = -1

    # Mark goal
    grid[config.goal_location] = 2

    # Plot trajectory
    trajectory_rows = [s[0] for s in states]
    trajectory_cols = [s[1] for s in states]

    im = ax.imshow(grid, cmap="RdYlGn", alpha=0.3, vmin=-1, vmax=2)
    ax.plot(trajectory_cols, trajectory_rows, "b-", alpha=0.5, linewidth=2)
    ax.plot(trajectory_cols, trajectory_rows, "bo", markersize=8)
    ax.plot(trajectory_cols[0], trajectory_rows[0], "go", markersize=12, label="Start")
    ax.plot(trajectory_cols[-1], trajectory_rows[-1], "r*", markersize=15, label="End")

    ax.set_title("Agent Trajectory in Grid World")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(config.size))
    ax.set_yticks(range(config.size))

    # Free energy over time
    ax = axes[0, 1]
    plot_line_series(
        ax, np.array(free_energies), title="Free Energy Over Episode", xlabel="Time Step", ylabel="Free Energy"
    )

    # Cumulative reward
    ax = axes[1, 0]
    cumulative_rewards = np.cumsum(rewards)
    plot_line_series(ax, cumulative_rewards, title="Cumulative Reward", xlabel="Time Step", ylabel="Cumulative Reward")

    # Actions histogram
    ax = axes[1, 1]
    action_counts = np.bincount(actions, minlength=4)
    action_names_short = ["Up", "Right", "Down", "Left"]
    ax.bar(range(4), action_counts)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_xticks(range(4))
    ax.set_xticklabels(action_names_short)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    runner.save_plot(fig, "trajectory_analysis", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 2: Belief evolution
    if len(beliefs) > 0:
        fig, ax = create_figure(1, 1, figsize=(10, 6))

        # Show top-k most probable states over time
        belief_matrix = np.array(beliefs).T
        im = ax.imshow(belief_matrix[:20, :], aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title("Belief Evolution (Top 20 States)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State Index")
        plt.colorbar(im, ax=ax, label="Probability")

        plt.tight_layout()
        runner.save_plot(fig, "belief_evolution", formats=["png"])
        plt.close(fig)

    # === 7. RECORD METRICS ===
    runner.record_metric("total_reward", float(total_reward))
    runner.record_metric("steps_taken", len(trajectory))
    runner.record_metric("goal_reached", bool(done))
    runner.record_metric("mean_free_energy", float(np.mean(free_energies)))
    runner.record_metric("mean_reward_per_step", float(np.mean(rewards)))

    # Efficiency: optimal path length vs actual
    manhattan_distance = abs(config.goal_location[0] - states[0][0]) + abs(config.goal_location[1] - states[0][1])
    efficiency = manhattan_distance / len(trajectory) if len(trajectory) > 0 else 0
    runner.record_metric("path_efficiency", float(efficiency))

    runner.logger.info(f"\nMean free energy: {np.mean(free_energies):.3f}")
    runner.logger.info(f"Path efficiency: {efficiency:.2%}")

    # === 8. FINISH ===
    runner.end()
    runner.logger.info("✓ Example complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
