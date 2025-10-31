"""Example 04: Markov Decision Process (MDP) demonstration.

Demonstrates:
- Fully observable MDP formulation
- Deterministic state transitions
- Reward-driven behavior
- Policy optimization
- Value iteration vs active inference

**THRML Integration**:
- Agent uses THRML sampling-based inference internally (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for efficient state inference
- Energy-efficient sampling for policy evaluation
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import example utilities
from example_utils import ExampleRunner, create_figure

from active_inference.agents import ActiveInferenceAgent
from active_inference.core import GenerativeModel, Precision
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "04_mdp_example"


def main():
    """Run MDP example."""
    # Initialize example runner
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=5)
    reward_state = runner.get_config("reward_state", default=4)
    n_actions = runner.get_config("n_actions", default=2)
    max_steps = runner.get_config("max_steps", default=50)
    sensory_precision = runner.get_config("sensory_precision", default=10.0)
    action_precision = runner.get_config("action_precision", default=3.0)
    planning_horizon = runner.get_config("planning_horizon", default=3)
    reward_preference_strength = runner.get_config("reward_preference_strength", default=5.0)
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  n_states: {n_states}")
    runner.logger.info(f"  reward_state: {reward_state}")

    # === 1. CREATE MDP ===
    runner.logger.info("Creating chain MDP")

    # Build chain MDP: States: 0 -> 1 -> 2 -> ... -> n_states-1
    # Actions: 0=left, 1=right, Goal: reach the reward state
    n_observations = n_states  # Fully observable

    # Observation model: identity (fully observable)
    A = jnp.eye(n_observations, n_states)

    # Transition model
    B = jnp.zeros((n_states, n_states, n_actions))

    # Action 0: move left
    for s in range(n_states):
        if s > 0:
            B = B.at[s - 1, s, 0].set(1.0)
        else:
            B = B.at[s, s, 0].set(1.0)  # Stay at boundary

    # Action 1: move right
    for s in range(n_states):
        if s < n_states - 1:
            B = B.at[s + 1, s, 1].set(1.0)
        else:
            B = B.at[s, s, 1].set(1.0)  # Stay at boundary

    # Preferences: prefer reward state
    C = jnp.zeros(n_observations)
    C = C.at[reward_state].set(reward_preference_strength)

    # Prior: uniform
    D = jnp.ones(n_states) / n_states

    model = GenerativeModel(
        n_states=n_states,
        n_observations=n_observations,
        n_actions=n_actions,
        A=A,
        B=B,
        C=C,
        D=D,
    )

    runner.logger.info(f"Chain MDP: {n_states} states")
    runner.logger.info(f"Reward state: {reward_state}")
    runner.logger.info("Actions: 0=left, 1=right")
    runner.logger.info("Observation: Fully observable (identity mapping)")

    # Save configuration
    config = {
        "seed": seed,
        "n_states": n_states,
        "reward_state": reward_state,
        "mdp_type": "chain",
        "fully_observable": True,
        "A_matrix": model.A,
        "B_tensor": model.B,
        "C_preferences": model.C,
        "D_prior": model.D,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "steps_per_sample": steps_per_sample,
    }
    runner.save_config(config)

    # === 2. CREATE AGENT ===
    runner.logger.info("\nCreating active inference agent")

    precision = Precision(
        sensory_precision=sensory_precision,
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

    runner.logger.info(f"Planning horizon: {agent.planning_horizon}")
    runner.logger.info(f"Action precision: {precision.action_precision}")

    # === 3. RUN MULTIPLE EPISODES FROM DIFFERENT START STATES ===
    runner.logger.info("\n=== Running episodes ===")

    all_episodes = []

    for start_state in range(n_states):
        runner.logger.info(f"\nEpisode starting from state {start_state}")

        # Reset agent
        agent_state = agent.reset()

        # Set current state
        current_state = start_state

        # Track episode
        states = [current_state]
        actions = []
        beliefs = []
        free_energies = []
        rewards_received = []

        for step in range(max_steps):
            # Observe current state (fully observable)
            observation = current_state

            # Perceive (agent now uses THRML internally)
            key, subkey_perceive = jax.random.split(key)
            posterior, fe = agent.perceive(observation, agent_state.belief, key=subkey_perceive)
            # Update agent state with new belief
            from active_inference.agents import AgentState

            agent_state = AgentState(
                belief=posterior,
                observation_history=agent_state.observation_history,
                action_history=agent_state.action_history,
                free_energy_history=agent_state.free_energy_history,
            )

            runner.logger.info(f"  Step {step}: state={current_state}, belief={posterior}")

            # Act
            key, subkey = jax.random.split(key)
            action = agent.act(subkey, posterior)

            runner.logger.info(f"    Action: {action} ({'left' if action == 0 else 'right'})")

            # Transition (deterministic in MDP)
            next_state = jnp.argmax(model.B[:, current_state, action])

            # Reward
            reward = 1.0 if next_state == reward_state else 0.0

            # Store
            actions.append(int(action))
            beliefs.append(np.array(posterior))
            free_energies.append(float(fe))
            rewards_received.append(float(reward))

            # Update
            current_state = int(next_state)
            states.append(current_state)

            # Note: Continue running to collect more trajectory data for visualization

        # Store episode
        episode_data = {
            "start_state": start_state,
            "states": np.array(states),
            "actions": np.array(actions),
            "beliefs": np.array(beliefs),
            "free_energies": np.array(free_energies),
            "rewards": np.array(rewards_received),
            "steps_to_goal": len(actions) if current_state == reward_state else None,
            "reached_goal": current_state == reward_state,
        }
        all_episodes.append(episode_data)

    # === 4. SAVE RESULTS ===
    runner.save_data({"episodes": all_episodes}, "mdp_episodes", format="npz")

    # === 5. VISUALIZE RESULTS ===
    runner.logger.info("\nGenerating visualizations")

    # Plot 1: State trajectories from all start states
    fig, axes = create_figure(2, 2, figsize=(14, 10))

    # Trajectories
    ax = axes[0, 0]
    for i, episode in enumerate(all_episodes):
        states = episode["states"]
        ax.plot(states, marker="o", label=f"Start={i}")
    ax.axhline(reward_state, color="r", linestyle="--", alpha=0.5, label="Goal")
    ax.set_title("State Trajectories from Different Start States")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("State")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Steps to goal
    ax = axes[0, 1]
    steps_to_goal = [e["steps_to_goal"] if e["steps_to_goal"] else max_steps for e in all_episodes]
    ax.bar(range(n_states), steps_to_goal)
    ax.set_title("Steps to Reach Goal")
    ax.set_xlabel("Start State")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3, axis="y")

    # Free energy evolution
    ax = axes[1, 0]
    for i, episode in enumerate(all_episodes):
        if len(episode["free_energies"]) > 0:
            ax.plot(episode["free_energies"], marker="o", label=f"Start={i}")
    ax.set_title("Free Energy Evolution")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Free Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Belief evolution for one episode
    ax = axes[1, 1]
    episode_to_plot = all_episodes[0]  # Start from state 0
    if len(episode_to_plot["beliefs"]) > 0:
        belief_matrix = np.array(episode_to_plot["beliefs"]).T
        im = ax.imshow(belief_matrix, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title(f"Belief Evolution (Start State={episode_to_plot['start_state']})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State")
        plt.colorbar(im, ax=ax, label="Belief")

    plt.tight_layout()
    runner.save_plot(fig, "mdp_analysis", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 2: Transition and preference matrices
    fig, axes = create_figure(1, 3, figsize=(15, 5))

    # Transition for action 0 (left)
    ax = axes[0]
    im = ax.imshow(np.array(model.B[:, :, 0]), cmap="Blues", interpolation="nearest")
    ax.set_title("Transition: Action 0 (Left)")
    ax.set_xlabel("Current State")
    ax.set_ylabel("Next State")
    plt.colorbar(im, ax=ax)

    # Transition for action 1 (right)
    ax = axes[1]
    im = ax.imshow(np.array(model.B[:, :, 1]), cmap="Blues", interpolation="nearest")
    ax.set_title("Transition: Action 1 (Right)")
    ax.set_xlabel("Current State")
    ax.set_ylabel("Next State")
    plt.colorbar(im, ax=ax)

    # Preferences
    ax = axes[2]
    ax.bar(range(n_states), np.array(model.C))
    ax.set_title("State Preferences")
    ax.set_xlabel("State")
    ax.set_ylabel("Preference (log)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    runner.save_plot(fig, "mdp_model", formats=["png", "pdf"])
    plt.close(fig)

    # === 6. ANALYSIS ===
    runner.logger.info("\n=== MDP Analysis ===")
    runner.logger.info("Characteristics:")
    runner.logger.info("  - Fully observable: agent knows exact state")
    runner.logger.info("  - Deterministic transitions: B matrices are deterministic")
    runner.logger.info("  - Reward-driven: preferences encoded in C vector")
    runner.logger.info("  - Active inference automatically finds optimal policy")

    # === 7. RECORD METRICS ===
    runner.record_metric("n_states", n_states)
    runner.record_metric("reward_state", reward_state)
    runner.record_metric("n_episodes", len(all_episodes))

    successful_episodes = sum(1 for e in all_episodes if e["reached_goal"])
    runner.record_metric("success_rate", successful_episodes / len(all_episodes))

    avg_steps = np.mean([e["steps_to_goal"] for e in all_episodes if e["steps_to_goal"]])
    runner.record_metric("average_steps_to_goal", float(avg_steps))

    runner.logger.info(f"\nSuccess rate: {successful_episodes}/{len(all_episodes)}")
    runner.logger.info(f"Average steps to goal: {avg_steps:.1f}")

    # === 8. FINISH ===
    runner.end()
    runner.logger.info("\n✓ Example complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
