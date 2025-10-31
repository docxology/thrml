"""Example 03: Exploration vs exploitation with precision control.

Demonstrates:
- Testing different precision values
- Comparing exploration vs exploitation
- Analyzing policy entropy
- Measuring performance trade-offs
- Visualizing precision effects on behavior

**THRML Integration**:
- Agent uses THRML sampling-based inference internally (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for efficient inference
- Energy-efficient sampling for policy evaluation across precision values
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import example utilities
from example_utils import ExampleRunner, calculate_entropy, create_figure

from active_inference.agents import ActiveInferenceAgent
from active_inference.core import Precision
from active_inference.environments import GridWorld, GridWorldConfig
from active_inference.inference import ThrmlInferenceEngine
from active_inference.models import build_grid_world_model

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "03_precision_control"


def main():
    """Run precision control example."""
    # Initialize example runner
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    grid_size = runner.get_config("grid_size", default=3)
    n_observations = runner.get_config("n_observations", default=5)
    observation_noise = runner.get_config("observation_noise", default=0.2)
    goal_location = tuple(runner.get_config("goal_location", default=[2, 2]))
    planning_horizon = runner.get_config("planning_horizon", default=2)
    precision_values = runner.get_config("precision_values", default=[0.5, 2.0, 10.0])
    precision_labels = runner.get_config(
        "precision_labels", default=["Low (Exploratory)", "Medium (Balanced)", "High (Exploitative)"]
    )
    max_steps = runner.get_config("max_steps", default=10)
    goal_preference_strength = runner.get_config("goal_preference_strength", default=2.0)
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  grid_size: {grid_size}")
    runner.logger.info(f"  precision_values: {precision_values}")

    # === 1. CREATE ENVIRONMENT ===
    runner.logger.info("Creating grid world environment")

    # Simple grid world from config
    config = GridWorldConfig(
        size=grid_size,
        n_observations=n_observations,
        observation_noise=observation_noise,
        goal_location=goal_location,
    )

    runner.logger.info(f"Environment: {config.size}x{config.size} grid")
    runner.logger.info(f"Goal: {config.goal_location}")

    # Save configuration
    env_config = {
        "seed": seed,
        "grid_size": config.size,
        "n_observations": config.n_observations,
        "observation_noise": config.observation_noise,
        "goal_location": config.goal_location,
    }
    runner.save_config(env_config)

    env = GridWorld(config=config)
    model = build_grid_world_model(config, goal_preference_strength=goal_preference_strength)

    runner.logger.info(f"Model: {model.n_states} states, {model.n_observations} observations")

    # Create THRML inference engine (shared across all precision tests)
    thrml_engine = ThrmlInferenceEngine(
        model=model,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )

    # === 2. TEST DIFFERENT PRECISION VALUES ===
    labels = precision_labels

    # Store results for each precision
    all_results = {}

    for prec_value, label in zip(precision_values, labels):
        runner.logger.info(f"\n=== Testing {label}: precision = {prec_value} ===")

        precision = Precision(action_precision=prec_value)
        agent = ActiveInferenceAgent(
            model=model,
            precision=precision,
            planning_horizon=planning_horizon,
            thrml_engine=thrml_engine,
        )

        # Track metrics
        entropies = []
        free_energies = []
        action_probs_list = []
        actions_taken = []
        states_visited = []

        # Run short episode
        agent_state = agent.reset()
        key_reset, key = jax.random.split(key)
        # Reset environment and set to corner (start position)
        import equinox as eqx

        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([0, 0]))
        env, obs = env.reset(key_reset)

        keys = jax.random.split(key, max_steps)

        for step in range(max_steps):
            # Get action distribution
            action_probs = agent.get_action_distribution(agent_state.belief)
            action_probs_list.append(np.array(action_probs))

            # Calculate entropy
            entropy = float(calculate_entropy(action_probs))
            entropies.append(entropy)

            runner.logger.info(f"Step {step}: Action probs: {[f'{p:.3f}' for p in action_probs]}")
            runner.logger.info(f"         Entropy: {entropy:.3f}")

            # Take action
            key_step, key_env = jax.random.split(keys[step])
            action, agent_state, fe = agent.step(key_step, obs, agent_state)

            actions_taken.append(int(action))
            states_visited.append(tuple(env.current_state))
            free_energies.append(float(fe))

            env, obs, _, done = env.step(key_env, action)

            if done:
                runner.logger.info(f"         Goal reached at step {step}!")
                break

        # Store results
        all_results[label] = {
            "precision": prec_value,
            "entropies": np.array(entropies),
            "free_energies": np.array(free_energies),
            "action_probs": np.array(action_probs_list),
            "actions_taken": np.array(actions_taken),
            "states_visited": states_visited,
            "mean_entropy": float(np.mean(entropies)),
            "mean_free_energy": float(np.mean(free_energies)),
            "goal_reached": bool(done),
            "steps_to_goal": step + 1 if done else None,
        }

        runner.logger.info(f"Mean entropy: {np.mean(entropies):.3f}")
        runner.logger.info(f"Mean free energy: {np.mean(free_energies):.3f}")

    # === 3. SAVE RESULTS ===
    runner.save_data(all_results, "precision_comparison", format="npz")

    # === 4. VISUALIZE RESULTS ===
    runner.logger.info("\nGenerating visualizations")

    # Plot 1: Entropy comparison
    fig, axes = create_figure(2, 2, figsize=(14, 10))

    # Entropy over time for each precision
    ax = axes[0, 0]
    for label in labels:
        entropies = all_results[label]["entropies"]
        ax.plot(entropies, marker="o", label=label)
    ax.set_title("Action Policy Entropy Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Entropy (bits)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Free energy over time for each precision
    ax = axes[0, 1]
    for label in labels:
        free_energies = all_results[label]["free_energies"]
        ax.plot(free_energies, marker="o", label=label)
    ax.set_title("Free Energy Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Free Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean entropy comparison
    ax = axes[1, 0]
    mean_entropies = [all_results[label]["mean_entropy"] for label in labels]
    precision_vals = [all_results[label]["precision"] for label in labels]
    ax.bar(range(len(labels)), mean_entropies, tick_label=[f"γ={p}" for p in precision_vals])
    ax.set_title("Mean Policy Entropy by Precision")
    ax.set_xlabel("Precision (γ)")
    ax.set_ylabel("Mean Entropy")
    ax.grid(True, alpha=0.3, axis="y")

    # Mean free energy comparison
    ax = axes[1, 1]
    mean_fes = [all_results[label]["mean_free_energy"] for label in labels]
    ax.bar(range(len(labels)), mean_fes, tick_label=[f"γ={p}" for p in precision_vals])
    ax.set_title("Mean Free Energy by Precision")
    ax.set_xlabel("Precision (γ)")
    ax.set_ylabel("Mean Free Energy")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    runner.save_plot(fig, "precision_comparison", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 2: Action probability distributions
    fig, axes = create_figure(1, 3, figsize=(15, 5))

    action_names = ["Up", "Right", "Down", "Left"]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        action_probs = all_results[label]["action_probs"]

        # Show heatmap of action probabilities over time
        im = ax.imshow(action_probs.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"{label}\n(γ={all_results[label]['precision']})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action")
        ax.set_yticks(range(4))
        ax.set_yticklabels(action_names)
        plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    runner.save_plot(fig, "action_distributions", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 3: Trajectories for each precision
    fig, axes = create_figure(1, 3, figsize=(15, 5))

    for idx, label in enumerate(labels):
        ax = axes[idx]
        states = all_results[label]["states_visited"]

        # Create grid
        grid = np.zeros((config.size, config.size))
        grid[config.goal_location] = 1

        # Plot trajectory
        rows = [s[0] for s in states]
        cols = [s[1] for s in states]

        ax.imshow(grid, cmap="RdYlGn", alpha=0.3, vmin=0, vmax=1)
        ax.plot(cols, rows, "b-", alpha=0.5, linewidth=2)
        ax.plot(cols, rows, "bo", markersize=8)
        ax.plot(cols[0], rows[0], "go", markersize=12, label="Start")
        ax.plot(cols[-1], rows[-1], "r*", markersize=15, label="End")

        ax.set_title(f"{label}\n(γ={all_results[label]['precision']})")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(config.size))
        ax.set_yticks(range(config.size))

    plt.tight_layout()
    runner.save_plot(fig, "trajectories_by_precision", formats=["png", "pdf"])
    plt.close(fig)

    # === 5. ANALYSIS ===
    runner.logger.info("\n=== Analysis ===")
    runner.logger.info("\nLow precision (high temperature):")
    runner.logger.info("  - More uniform action distributions")
    runner.logger.info("  - Higher entropy = more exploration")
    runner.logger.info("  - Good for uncertain/novel situations")
    runner.logger.info(f"  - Mean entropy: {all_results[labels[0]]['mean_entropy']:.3f}")

    runner.logger.info("\nHigh precision (low temperature):")
    runner.logger.info("  - Peaked action distributions")
    runner.logger.info("  - Lower entropy = more exploitation")
    runner.logger.info("  - Good for well-known situations")
    runner.logger.info(f"  - Mean entropy: {all_results[labels[2]]['mean_entropy']:.3f}")

    # === 6. RECORD METRICS ===
    for label in labels:
        prefix = label.replace(" ", "_").replace("(", "").replace(")", "")
        runner.record_metric(f"{prefix}_mean_entropy", all_results[label]["mean_entropy"])
        runner.record_metric(f"{prefix}_mean_free_energy", all_results[label]["mean_free_energy"])
        runner.record_metric(f"{prefix}_goal_reached", all_results[label]["goal_reached"])
        if all_results[label]["steps_to_goal"]:
            runner.record_metric(f"{prefix}_steps_to_goal", all_results[label]["steps_to_goal"])

    # === 7. FINISH ===
    runner.end()
    runner.logger.info("\n✓ Example complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
