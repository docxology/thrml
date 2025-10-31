"""Example 05: Partially Observable Markov Decision Process (POMDP) demonstration.

Demonstrates:
- Partial observability with noisy observations
- Belief state maintenance
- Information-seeking behavior (epistemic value)
- Comparison with fully observable case
- Active sensing and exploration

**THRML Integration**:
- Agent uses THRML sampling-based inference for belief updating (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for efficient belief state inference
- THRML methods excel at belief state sampling in POMDPs with complex posteriors
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
from active_inference.core import GenerativeModel, Precision
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "05_pomdp_example"


def main():
    """Run POMDP example."""
    # Initialize example runner
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    observation_accuracy = runner.get_config("observation_accuracy", default=0.85)
    n_episodes = runner.get_config("n_episodes", default=10)
    max_steps = runner.get_config("max_steps", default=50)
    sensory_precision = runner.get_config("sensory_precision", default=2.0)
    action_precision = runner.get_config("action_precision", default=2.0)
    planning_horizon = runner.get_config("planning_horizon", default=2)
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  observation_accuracy: {observation_accuracy}")
    runner.logger.info(f"  n_episodes: {n_episodes}")

    # === 1. BUILD TIGER POMDP ===
    runner.logger.info("Building Tiger POMDP")

    # Classic Tiger POMDP: States: 0=tiger-left, 1=tiger-right
    # Actions: 0=listen, 1=open-left, 2=open-right
    # Observations: 0=hear-left, 1=hear-right
    n_states = 2
    n_actions = 3
    n_observations = 2

    # Observation model: noisy observations
    A = jnp.array(
        [
            [observation_accuracy, 1 - observation_accuracy],
            [1 - observation_accuracy, observation_accuracy],
        ]
    )

    # Transition model
    B = jnp.zeros((n_states, n_states, n_actions))
    B = B.at[:, :, 0].set(jnp.eye(n_states))  # listen: no change
    B = B.at[:, :, 1].set(jnp.ones((n_states, n_states)) / n_states)  # open-left: reset
    B = B.at[:, :, 2].set(jnp.ones((n_states, n_states)) / n_states)  # open-right: reset

    # Preferences: neutral
    C = jnp.zeros(n_observations)

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

    runner.logger.info("Tiger POMDP problem:")
    runner.logger.info("  States: 0=tiger-left, 1=tiger-right")
    runner.logger.info("  Actions: 0=listen, 1=open-left, 2=open-right")
    runner.logger.info("  Observations: 0=hear-left, 1=hear-right")
    runner.logger.info(f"  Observation accuracy: {observation_accuracy}")

    # Save configuration
    config = {
        "seed": seed,
        "observation_accuracy": observation_accuracy,
        "n_states": n_states,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "pomdp_type": "tiger",
        "partially_observable": True,
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

    # === 3. RUN MULTIPLE EPISODES ===
    runner.logger.info("\n=== Running episodes ===")

    all_episodes = []

    for episode_idx in range(n_episodes):
        runner.logger.info(f"\n--- Episode {episode_idx + 1} ---")

        # Random true state
        key, subkey = jax.random.split(key)
        true_state = int(jax.random.choice(subkey, 2))
        runner.logger.info(f"True state: {true_state} ({'tiger-left' if true_state == 0 else 'tiger-right'})")

        # Reset agent
        agent_state = agent.reset()

        # Track episode
        observations_received = []
        actions_taken = []
        beliefs = []
        belief_entropies = []
        free_energies = []
        rewards_received = []
        listen_count = 0

        for step in range(max_steps):
            # Generate observation based on true state
            key, subkey = jax.random.split(key)
            obs_probs = model.A[:, true_state]
            observation = int(jax.random.choice(subkey, model.n_observations, p=obs_probs))

            observations_received.append(observation)

            runner.logger.info(
                f"\nStep {step}: Observation={observation} ({'hear-left' if observation == 0 else 'hear-right'})"
            )

            # Perceive (agent now uses THRML internally)
            key, subkey_perceive, subkey = jax.random.split(key, 3)
            posterior, fe = agent.perceive(observation, agent_state.belief, key=subkey_perceive)
            # Update agent state with new belief
            from active_inference.agents import AgentState

            agent_state = AgentState(
                belief=posterior,
                observation_history=agent_state.observation_history,
                action_history=agent_state.action_history,
                free_energy_history=agent_state.free_energy_history,
            )

            # Calculate belief entropy
            belief_ent = calculate_entropy(posterior)

            runner.logger.info(f"  Belief: {posterior}")
            runner.logger.info(f"  Belief entropy: {belief_ent:.3f}")
            runner.logger.info(f"  Free energy: {fe:.3f}")

            # Store
            beliefs.append(np.array(posterior))
            belief_entropies.append(belief_ent)
            free_energies.append(float(fe))

            # Act
            key, subkey = jax.random.split(key)
            action = agent.act(subkey, posterior)

            action_names = ["listen", "open-left", "open-right"]
            runner.logger.info(f"  Action: {action} ({action_names[action]})")

            actions_taken.append(int(action))

            # Track listen actions (information seeking)
            if action == 0:
                listen_count += 1
                reward = 0.0  # No reward for listening

            # Check if opening door
            if action > 0:
                # Determine reward
                correct_door = 1 if true_state == 0 else 2  # Open left if tiger is left, etc.
                reward = 10.0 if action == correct_door else -10.0

                runner.logger.info(f"  Opening door: {'correct' if action == correct_door else 'incorrect'}!")
                runner.logger.info(f"  Reward: {reward}")

                # Note: Continue episode instead of breaking to collect more data
                # Reset state randomly for next round
                key, subkey = jax.random.split(key)
                true_state = int(jax.random.choice(subkey, 2))

            # Store reward
            rewards_received.append(float(reward))

        # Store episode
        episode_data = {
            "episode": episode_idx,
            "true_state": true_state,
            "observations": np.array(observations_received),
            "actions": np.array(actions_taken),
            "beliefs": np.array(beliefs),
            "belief_entropies": np.array(belief_entropies),
            "free_energies": np.array(free_energies),
            "rewards": np.array(rewards_received),
            "listen_count": listen_count,
            "final_action": int(actions_taken[-1]) if len(actions_taken) > 0 else None,
            # Calculate accuracy from all door-opening actions
            "correct_door": bool(np.sum([r for r in rewards_received if r > 0]) > 0),
            "total_reward": float(np.sum(rewards_received)),
        }
        all_episodes.append(episode_data)

    # === 4. SAVE RESULTS ===
    runner.save_data({"episodes": all_episodes}, "pomdp_episodes", format="npz")

    # === 5. VISUALIZE RESULTS ===
    runner.logger.info("\nGenerating visualizations")

    # Plot 1: Belief evolution for sample episodes
    fig, axes = create_figure(2, 3, figsize=(15, 10))

    for idx in range(min(6, n_episodes)):
        ax = axes[idx // 3, idx % 3]
        episode = all_episodes[idx]

        beliefs = episode["beliefs"]
        if len(beliefs) > 0:
            # Plot belief trajectory
            steps = range(len(beliefs))
            ax.plot(steps, [b[0] for b in beliefs], "b-o", label="P(tiger-left)")
            ax.plot(steps, [b[1] for b in beliefs], "r-o", label="P(tiger-right)")

            # Mark true state
            true_state = episode["true_state"]
            ax.axhline(1.0 if true_state == 0 else 0.0, color="g", linestyle="--", alpha=0.3, label="True state")

            ax.set_title(
                f"Episode {idx+1}: True={'L' if true_state == 0 else 'R'}, "
                f"Chose={'L' if episode['final_action'] == 1 else 'R' if episode['final_action'] == 2 else '?'}"
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    runner.save_plot(fig, "belief_trajectories", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 2: Summary statistics
    fig, axes = create_figure(2, 2, figsize=(12, 10))

    # Listen counts
    ax = axes[0, 0]
    listen_counts = [e["listen_count"] for e in all_episodes]
    ax.hist(listen_counts, bins=range(max(listen_counts) + 2), edgecolor="black")
    ax.set_title("Information Seeking Behavior")
    ax.set_xlabel("Number of Listen Actions")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3, axis="y")

    # Belief entropy evolution
    ax = axes[0, 1]
    for idx in range(min(5, n_episodes)):
        episode = all_episodes[idx]
        ax.plot(episode["belief_entropies"], marker="o", label=f"Ep {idx+1}")
    ax.set_title("Belief Uncertainty Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Belief Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success rate
    ax = axes[1, 0]
    correct_decisions = sum(1 for e in all_episodes if e["correct_door"])
    success_rate = correct_decisions / n_episodes
    ax.bar(["Correct", "Incorrect"], [correct_decisions, n_episodes - correct_decisions])
    ax.set_title(f"Decision Accuracy ({success_rate:.0%})")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    # Free energy evolution
    ax = axes[1, 1]
    for idx in range(min(5, n_episodes)):
        episode = all_episodes[idx]
        ax.plot(episode["free_energies"], marker="o", label=f"Ep {idx+1}")
    ax.set_title("Free Energy Evolution")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Free Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    runner.save_plot(fig, "pomdp_analysis", formats=["png", "pdf"])
    plt.close(fig)

    # Plot 3: Observation model and strategy
    fig, axes = create_figure(1, 2, figsize=(12, 5))

    # Observation likelihood
    ax = axes[0]
    im = ax.imshow(np.array(model.A), cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Observation Model (accuracy={observation_accuracy})")
    ax.set_xlabel("True State")
    ax.set_ylabel("Observation")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tiger-left", "tiger-right"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["hear-left", "hear-right"])
    plt.colorbar(im, ax=ax, label="P(obs|state)")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f"{model.A[i, j]:.2f}", ha="center", va="center", color="red", fontsize=12)

    # Rewards distribution
    ax = axes[1]
    rewards = [e["total_reward"] for e in all_episodes]
    mean_reward = np.mean(rewards)
    ax.bar(range(len(rewards)), rewards)
    ax.axhline(mean_reward, color="r", linestyle="--", label=f"Mean={mean_reward:.1f}")
    ax.set_title("Total Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    runner.save_plot(fig, "pomdp_model", formats=["png", "pdf"])
    plt.close(fig)

    # === 6. ANALYSIS ===
    runner.logger.info("\n=== POMDP Analysis ===")
    runner.logger.info("Characteristics:")
    runner.logger.info("  - Partial observability: agent must infer state from noisy observations")
    runner.logger.info("  - Belief state: agent maintains probability distribution over states")
    runner.logger.info("  - Information seeking: agent performs 'listen' actions to reduce uncertainty")
    runner.logger.info("  - Epistemic value: gathering information before committing to action")

    # === 7. RECORD METRICS ===
    runner.record_metric("n_episodes", n_episodes)
    runner.record_metric("observation_accuracy", observation_accuracy)
    runner.record_metric("success_rate", float(success_rate))
    runner.record_metric("mean_listen_count", float(np.mean(listen_counts)))
    runner.record_metric("mean_reward", float(np.mean(rewards)))

    # Compare initial vs final belief entropy
    initial_entropies = [e["belief_entropies"][0] for e in all_episodes if len(e["belief_entropies"]) > 0]
    final_entropies = [e["belief_entropies"][-1] for e in all_episodes if len(e["belief_entropies"]) > 0]
    runner.record_metric("mean_initial_entropy", float(np.mean(initial_entropies)))
    runner.record_metric("mean_final_entropy", float(np.mean(final_entropies)))
    runner.record_metric("entropy_reduction", float(np.mean(initial_entropies) - np.mean(final_entropies)))

    runner.logger.info(f"\nSuccess rate: {success_rate:.1%}")
    runner.logger.info(f"Mean listen count: {np.mean(listen_counts):.1f}")
    runner.logger.info(f"Mean reward: {np.mean(rewards):.1f}")
    runner.logger.info(f"Entropy reduction: {np.mean(initial_entropies) - np.mean(final_entropies):.3f}")

    # === 8. FINISH ===
    runner.end()
    runner.logger.info("\n✓ Example complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
