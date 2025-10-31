"""Example 14: Basic state inference with active inference (formerly Example 01).

This example demonstrates:
- Creating a simple generative model (POMDP)
- Performing THRML-based sampling inference from observations
- Calculating variational free energy
- Saving outputs and visualizations

**Configuration**: All parameters loaded from examples_config.yaml
**Pattern**: Thin orchestrator calling src/active_inference methods

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for energy-efficient inference
- Comprehensive demonstration of THRML inference capabilities
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import example utilities with config support
from example_utils import ExampleRunner, create_figure, plot_distribution, plot_line_series

from active_inference.core import GenerativeModel, variational_free_energy
from active_inference.inference import ThrmlInferenceEngine

# Configuration loaded automatically from examples_config.yaml
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "14_basic_inference"


def main():
    """Run basic inference example with config-driven parameters."""
    # Initialize example runner - automatically loads config
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=4)
    n_observations = runner.get_config("n_observations", default=4)
    n_actions = runner.get_config("n_actions", default=2)
    n_inference_iterations = runner.get_config("n_inference_iterations", default=16)
    observation_noise = runner.get_config("observation_noise", default=0.05)
    test_observations = runner.get_config("test_observations", default=[0, 1, 2, 3, 0, 1])
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  n_states: {n_states}")
    runner.logger.info(f"  n_observations: {n_observations}")
    runner.logger.info(f"  n_actions: {n_actions}")
    runner.logger.info(f"  inference_iterations: {n_inference_iterations}")
    runner.logger.info(f"  observation_noise: {observation_noise}")
    runner.logger.info(f"  THRML: n_samples={n_samples}, n_warmup={n_warmup}, steps_per_sample={steps_per_sample}")

    # === 1. CREATE GENERATIVE MODEL ===
    runner.logger.info("Creating generative model")

    # Observation model: somewhat noisy identity mapping
    A = jnp.eye(n_observations, n_states) * 0.8 + jnp.ones((n_observations, n_states)) * observation_noise
    A = A / A.sum(axis=0, keepdims=True)  # Normalize

    # State transition model: deterministic with action-dependent transitions
    B = jnp.zeros((n_states, n_states, n_actions))
    # Action 0: rotate forward (0->1->2->3->0)
    for s in range(n_states - 1):
        B = B.at[s + 1, s, 0].set(1.0)
    B = B.at[0, n_states - 1, 0].set(1.0)
    # Action 1: rotate backward (0->3->2->1->0)
    for s in range(1, n_states):
        B = B.at[s - 1, s, 1].set(1.0)
    B = B.at[n_states - 1, 0, 1].set(1.0)

    # Preferences: no strong preferences
    C = jnp.zeros(n_observations)

    # Prior: uniform
    D = jnp.ones(n_states) / n_states

    # Create model using src method
    model = GenerativeModel(
        n_states=n_states,
        n_observations=n_observations,
        n_actions=n_actions,
        A=A,
        B=B,
        C=C,
        D=D,
    )

    runner.logger.info(f"Model: {n_states} states, {n_observations} observations, {n_actions} actions")

    # === CREATE THRML INFERENCE ENGINE ===
    runner.logger.info("Creating THRML inference engine")
    thrml_engine = ThrmlInferenceEngine(
        model=model,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )
    runner.logger.info(f"THRML engine: {n_samples} samples, {n_warmup} warmup, {steps_per_sample} steps/sample")

    # Save model configuration
    config = {
        "seed": seed,
        "n_states": n_states,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "n_inference_iterations": n_inference_iterations,
        "observation_noise": observation_noise,
        "A_matrix": A,
        "B_tensor": B,
        "C_preferences": C,
        "D_prior": D,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "steps_per_sample": steps_per_sample,
    }
    runner.save_config(config)

    # === 2. PERFORM INFERENCE ON SEQUENCE OF OBSERVATIONS ===
    runner.logger.info("Performing state inference")

    # Get test observations from config
    observations = test_observations
    true_states = observations.copy()  # Assuming correspondence for this example

    # Track results
    posteriors = []
    free_energies = []
    kl_divergences = []

    prior_belief = D.copy()

    for i, obs in enumerate(observations):
        runner.logger.info(f"\n--- Step {i} ---")
        runner.logger.info(f"Observation: {obs}")
        runner.logger.info(f"Prior belief: {prior_belief}")

        # Perform inference using THRML sampling
        key_infer, key = jax.random.split(key)
        # Update model prior for this inference step
        model_with_prior = eqx.tree_at(lambda m: m.D, model, prior_belief)
        temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)
        posterior = temp_engine.infer_with_sampling(
            key=key_infer,
            observation=obs,
            n_state_samples=n_samples,
        )
        # Calculate free energy for compatibility
        fe = variational_free_energy(obs, posterior, model)

        # Calculate KL divergence
        kl = jnp.sum(posterior * jnp.log(posterior / (prior_belief + 1e-16) + 1e-16))

        runner.logger.info(f"Posterior belief: {posterior}")
        runner.logger.info(f"Free energy: {fe:.4f}")
        runner.logger.info(f"KL[posterior || prior]: {kl:.4f}")
        runner.logger.info(f"MAP state: {jnp.argmax(posterior)}")

        # Store results
        posteriors.append(np.array(posterior))
        free_energies.append(float(fe))
        kl_divergences.append(float(kl))

        # Update prior for next step
        prior_belief = posterior

    # Save inference results
    inference_data = {
        "observations": np.array(observations),
        "true_states": np.array(true_states),
        "posteriors": np.array(posteriors),
        "free_energies": np.array(free_energies),
        "kl_divergences": np.array(kl_divergences),
    }
    runner.save_data(inference_data, "inference_results")

    # === 3. VISUALIZE RESULTS ===
    runner.logger.info("\nGenerating visualizations")

    # Plot 1: Belief trajectory
    fig, axes = create_figure(2, 2, figsize=(12, 10))

    # Belief evolution heatmap
    ax = axes[0, 0]
    belief_matrix = np.array(posteriors).T
    im = ax.imshow(belief_matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title("Belief Evolution Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("State")
    ax.set_yticks(range(n_states))
    plt.colorbar(im, ax=ax, label="Probability")

    # True states overlay
    if true_states:
        ax.plot(range(len(true_states)), true_states, "r*-", label="True State", markersize=10)
        ax.legend()

    # Free energy over time
    ax = axes[0, 1]
    plot_line_series(
        ax, np.array(free_energies), title="Variational Free Energy", xlabel="Time Step", ylabel="Free Energy"
    )

    # KL divergence over time
    ax = axes[1, 0]
    plot_line_series(
        ax,
        np.array(kl_divergences),
        title="KL Divergence (Posterior || Prior)",
        xlabel="Time Step",
        ylabel="KL Divergence",
    )

    # Final posterior distribution
    ax = axes[1, 1]
    plot_distribution(
        ax, posteriors[-1], title=f"Final Posterior (obs={observations[-1]})", xlabel="State", ylabel="Probability"
    )

    plt.tight_layout()
    runner.save_plot(fig, "inference_analysis")
    plt.close(fig)

    # Plot 2: Model matrices
    fig, axes = create_figure(1, 2, figsize=(12, 5))

    # Observation likelihood matrix
    ax = axes[0]
    im = ax.imshow(np.array(A), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Observation Likelihood A[o|s]")
    ax.set_xlabel("State")
    ax.set_ylabel("Observation")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_observations))
    plt.colorbar(im, ax=ax, label="P(o|s)")

    # Transition matrix for action 0
    ax = axes[1]
    im = ax.imshow(np.array(B[:, :, 0]), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("State Transition B[s'|s, action=0]")
    ax.set_xlabel("Current State")
    ax.set_ylabel("Next State")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    plt.colorbar(im, ax=ax, label="P(s'|s,a)")

    plt.tight_layout()
    runner.save_plot(fig, "model_matrices")
    plt.close(fig)

    # === 4. RECORD METRICS ===
    runner.record_metric("n_observations", len(observations))
    runner.record_metric("mean_free_energy", float(np.mean(free_energies)))
    runner.record_metric("final_free_energy", float(free_energies[-1]))
    runner.record_metric("mean_kl_divergence", float(np.mean(kl_divergences)))

    # Inference accuracy (did MAP state match true state?)
    map_states = [np.argmax(p) for p in posteriors]
    accuracy = np.mean([m == t for m, t in zip(map_states, true_states)])
    runner.record_metric("inference_accuracy", float(accuracy))

    runner.logger.info("\n=== Results Summary ===")
    runner.logger.info(f"Inference accuracy: {accuracy:.2%}")
    runner.logger.info(f"Mean free energy: {np.mean(free_energies):.4f}")
    runner.logger.info(f"Mean KL divergence: {np.mean(kl_divergences):.4f}")

    # === 5. FINISH ===
    runner.end()
    runner.logger.info("✓ Example complete")
    runner.logger.info("✓ All parameters from examples_config.yaml")
    runner.logger.info(f"✓ Output saved to: {runner.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
