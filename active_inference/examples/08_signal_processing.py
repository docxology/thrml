"""Example 08: THRML Temporal Inference for Signal Processing.

Demonstrates THRML for time series analysis and signal processing:
- Sequential Bayesian filtering with THRML
- Hidden state estimation in temporal models
- Signal denoising using THRML sampling
- Temporal dependencies via transition models
- Comparison with analytical solutions

**THRML Methods Used**:
- `ThrmlInferenceEngine` - THRML-based inference wrapper
- Sequential state inference for time series
- Categorical state spaces for discretized signals
- Real THRML sampling: `Block`, `BlockGibbsSpec`, `sample_states`

This example demonstrates THRML's effectiveness for temporal inference,
showing how block Gibbs sampling can efficiently track hidden states
in dynamical systems with noisy observations.
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from example_utils import ExampleRunner, create_figure

from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "08_signal_processing"


def create_hmm_model(n_states, observation_noise=0.1):
    """Create a Hidden Markov Model for signal processing.

    Args:
        n_states: Number of discrete signal levels
        observation_noise: Observation noise level

    Returns:
        GenerativeModel configured as HMM

    """
    # Transition matrix: smooth temporal dynamics
    # Prefer staying in current state or moving to adjacent states
    B = jnp.zeros((n_states, n_states, 1))  # Single "action" (time evolution)
    for i in range(n_states):
        for j in range(n_states):
            # Distance-based transition probability
            distance = abs(i - j)
            if distance == 0:
                B = B.at[j, i, 0].set(0.7)  # Stay in current state
            elif distance == 1:
                B = B.at[j, i, 0].set(0.15)  # Move to adjacent state
            else:
                B = B.at[j, i, 0].set(0.15 / (n_states - 3) if n_states > 3 else 0.0)
        # Normalize
        B = B.at[:, i, 0].set(B[:, i, 0] / jnp.sum(B[:, i, 0]))

    # Observation matrix: noisy observations of signal level
    A = jnp.eye(n_states) * (1 - observation_noise)
    # Add uniform noise to all states
    A = A + observation_noise / n_states
    # Normalize
    A = A / jnp.sum(A, axis=0, keepdims=True)

    # Initial state prior: uniform
    D = jnp.ones(n_states) / n_states

    # No preferences (neutral HMM)
    C = jnp.zeros(n_states)

    model = GenerativeModel(n_states=n_states, n_observations=n_states, n_actions=1, A=A, B=B, C=C, D=D)

    return model


def generate_signal_trajectory(key, n_steps, n_states, smoothness=0.9):
    """Generate a smooth signal trajectory as ground truth.

    Args:
        key: JAX random key
        n_steps: Number of time steps
        n_states: Number of discrete signal levels
        smoothness: Smoothness parameter (higher = smoother)

    Returns:
        Array of signal states over time

    """
    states = np.zeros(n_steps, dtype=int)
    states[0] = n_states // 2  # Start in middle

    for t in range(1, n_steps):
        key, subkey = jax.random.split(key)
        # Bias towards staying near current state
        if jax.random.uniform(subkey) < smoothness:
            # Small change
            change = jax.random.choice(subkey, jnp.array([-1, 0, 1]))
            states[t] = np.clip(states[t - 1] + int(change), 0, n_states - 1)
        else:
            # Random jump
            states[t] = int(jax.random.choice(subkey, jnp.arange(n_states)))

    return states


def add_observation_noise(key, true_states, observation_noise, n_states):
    """Add noise to observations.

    Args:
        key: JAX random key
        true_states: True signal states
        observation_noise: Noise probability
        n_states: Number of states

    Returns:
        Noisy observations

    """
    n_steps = len(true_states)
    observations = np.array(true_states).copy()

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        if jax.random.uniform(subkey) < observation_noise:
            # Corrupted observation
            observations[t] = int(jax.random.choice(subkey, jnp.arange(n_states)))

    return observations


def main():
    """Run THRML temporal inference demonstration."""
    # Initialize runner
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=20)
    n_steps = runner.get_config("n_steps", default=200)
    observation_noise = runner.get_config("observation_noise", default=0.2)
    smoothness = runner.get_config("smoothness", default=0.85)

    # THRML parameters
    n_samples = runner.get_config("n_samples", default=100)
    n_warmup = runner.get_config("n_warmup", default=25)
    steps_per_sample = runner.get_config("steps_per_sample", default=3)

    key = jax.random.key(seed)

    runner.logger.info("THRML Temporal Inference Configuration:")
    runner.logger.info(f"  Number of states: {n_states}")
    runner.logger.info(f"  Time steps: {n_steps}")
    runner.logger.info(f"  Observation noise: {observation_noise}")
    runner.logger.info(f"  THRML samples: {n_samples}")

    # === CONFIGURATION ===
    with runner.section("Configuration"):
        config = {
            "seed": seed,
            "n_states": n_states,
            "n_steps": n_steps,
            "observation_noise": observation_noise,
            "smoothness": smoothness,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "steps_per_sample": steps_per_sample,
        }
        runner.save_config(config)

    # === CREATE HMM MODEL ===
    with runner.section("Model Creation"):
        runner.logger.info("Creating Hidden Markov Model...")
        model = create_hmm_model(n_states, observation_noise=0.1)

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=model,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        )

        runner.logger.info(f"✓ HMM created with {n_states} states")
        runner.logger.info("✓ THRML engine initialized")

    # === GENERATE SIGNAL ===
    with runner.section("Signal Generation"):
        runner.logger.info("Generating test signal...")

        key, subkey1, subkey2 = jax.random.split(key, 3)
        true_states = generate_signal_trajectory(subkey1, n_steps, n_states, smoothness)
        observations = add_observation_noise(subkey2, true_states, observation_noise, n_states)

        # Calculate noise statistics
        noise_rate = np.mean(observations != true_states)
        runner.logger.info(f"✓ Generated {n_steps}-step trajectory")
        runner.logger.info(f"  Actual observation corruption rate: {noise_rate*100:.1f}%")

    # === THRML FILTERING ===
    with runner.section("THRML Sequential Inference"):
        runner.logger.info("Running THRML filtering...")

        beliefs = []
        estimated_states = []
        current_prior = model.D.copy()

        for t in range(n_steps):
            obs = int(observations[t])

            # THRML inference
            key, subkey = jax.random.split(key)
            model_with_prior = eqx.tree_at(lambda m: m.D, model, current_prior)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)

            posterior = temp_engine.infer_with_sampling(
                key=subkey,
                observation=obs,
                n_state_samples=n_samples,
            )
            beliefs.append(posterior)

            # Estimate state (MAP)
            est_state = int(jnp.argmax(posterior))
            estimated_states.append(est_state)

            # Predict next prior using transition model
            # current_prior[s_{t+1}] = sum_s_t B[s_{t+1}, s_t, a=0] * posterior[s_t]
            current_prior = jnp.einsum("ij,j->i", model.B[:, :, 0], posterior)
            current_prior = current_prior / (jnp.sum(current_prior) + 1e-16)

            if t % 50 == 0:
                runner.logger.info(f"  Step {t}/{n_steps}: obs={obs}, true={true_states[t]}, est={est_state}")

        beliefs = np.array(beliefs)
        estimated_states = np.array(estimated_states)

        runner.logger.info(f"✓ Completed {n_steps} steps of THRML filtering")

    # === ANALYSIS ===
    with runner.section("Performance Analysis"):
        # Estimation accuracy
        estimation_accuracy = np.mean(estimated_states == true_states)
        observation_accuracy = np.mean(observations == true_states)

        # Estimation error
        estimation_errors = np.abs(estimated_states - true_states)
        observation_errors = np.abs(observations - true_states)

        # Mean absolute error
        mae_estimation = np.mean(estimation_errors)
        mae_observation = np.mean(observation_errors)

        runner.logger.info("\nPerformance Metrics:")
        runner.logger.info(f"  Raw Observation Accuracy: {observation_accuracy*100:.1f}%")
        runner.logger.info(f"  THRML Estimation Accuracy: {estimation_accuracy*100:.1f}%")
        runner.logger.info(f"  Improvement: {(estimation_accuracy - observation_accuracy)*100:+.1f}%")
        runner.logger.info(f"  Raw Observation MAE: {mae_observation:.2f} states")
        runner.logger.info(f"  THRML Estimation MAE: {mae_estimation:.2f} states")

        # Save results
        results = {
            "true_states": true_states,
            "observations": observations,
            "estimated_states": estimated_states,
            "beliefs": beliefs,
            "estimation_accuracy": estimation_accuracy,
            "observation_accuracy": observation_accuracy,
            "mae_estimation": mae_estimation,
            "mae_observation": mae_observation,
        }
        runner.save_data(results, "filtering_results")

    # === VISUALIZATION ===
    with runner.section("Visualization"):
        runner.logger.info("Generating visualizations...")

        fig, axes = create_figure(2, 2, figsize=(14, 10))
        time = np.arange(n_steps)

        # Signal trajectories
        ax = axes[0, 0]
        ax.plot(time, true_states, "g-", label="True Signal", linewidth=2, alpha=0.7)
        ax.plot(time, observations, "r.", label="Noisy Observations", markersize=2, alpha=0.5)
        ax.plot(time, estimated_states, "b-", label="THRML Estimate", linewidth=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal State")
        ax.set_title("THRML Signal Denoising")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Belief heatmap
        ax = axes[0, 1]
        im = ax.imshow(beliefs.T, aspect="auto", cmap="viridis", origin="lower", extent=[0, n_steps, 0, n_states])
        ax.plot(time, true_states, "r-", linewidth=1, alpha=0.5, label="True State")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal State")
        ax.set_title("THRML Belief Evolution")
        plt.colorbar(im, ax=ax, label="Belief Probability")
        ax.legend()

        # Estimation errors
        ax = axes[1, 0]
        ax.plot(time, estimation_errors, "b-", label="THRML Error", linewidth=2, alpha=0.7)
        ax.plot(time, observation_errors, "r-", label="Observation Error", linewidth=1, alpha=0.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Absolute Error (states)")
        ax.set_title("Estimation Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1, 1]
        ax.hist(observation_errors, bins=20, alpha=0.5, label="Observation", density=True, color="red")
        ax.hist(estimation_errors, bins=20, alpha=0.5, label="THRML", density=True, color="blue")
        ax.set_xlabel("Absolute Error (states)")
        ax.set_ylabel("Density")
        ax.set_title("Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig, "thrml_signal_processing", formats=["png", "pdf"])
        plt.close(fig)

        # Create zoom plot for detailed view
        fig_zoom, ax = create_figure(1, 1, figsize=(12, 6))
        zoom_start = 50
        zoom_end = 100
        zoom_time = time[zoom_start:zoom_end]

        ax.plot(zoom_time, true_states[zoom_start:zoom_end], "g-", label="True Signal", linewidth=3, alpha=0.7)
        ax.plot(zoom_time, observations[zoom_start:zoom_end], "ro", label="Noisy Observations", markersize=6, alpha=0.6)
        ax.plot(zoom_time, estimated_states[zoom_start:zoom_end], "b-", label="THRML Estimate", linewidth=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal State")
        ax.set_title(f"THRML Signal Denoising (Detail: steps {zoom_start}-{zoom_end})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig_zoom, "thrml_signal_detail", formats=["png", "pdf"])
        plt.close(fig_zoom)

    # === METRICS ===
    with runner.section("Summary Metrics"):
        runner.record_metric("estimation_accuracy", float(estimation_accuracy))
        runner.record_metric("observation_accuracy", float(observation_accuracy))
        runner.record_metric("improvement_pct", float((estimation_accuracy - observation_accuracy) * 100))
        runner.record_metric("mae_estimation", float(mae_estimation))
        runner.record_metric("mae_observation", float(mae_observation))
        runner.record_metric("mae_reduction_pct", float((mae_observation - mae_estimation) / mae_observation * 100))
        runner.record_metric("n_steps", n_steps)
        runner.record_metric("n_states", n_states)
        runner.record_metric("thrml_n_samples", n_samples)

        runner.logger.info(f"\n{'='*70}")
        runner.logger.info("THRML SIGNAL PROCESSING SUMMARY")
        runner.logger.info(f"{'='*70}")
        runner.logger.info(f"Time Steps: {n_steps}")
        runner.logger.info(f"Signal States: {n_states}")
        runner.logger.info(f"Observation Noise: {noise_rate*100:.1f}%")
        runner.logger.info(f"Raw Accuracy: {observation_accuracy*100:.1f}%")
        runner.logger.info(f"THRML Accuracy: {estimation_accuracy*100:.1f}%")
        runner.logger.info(f"Improvement: {(estimation_accuracy - observation_accuracy)*100:+.1f}%")
        runner.logger.info(f"MAE Reduction: {(mae_observation - mae_estimation)/mae_observation*100:.1f}%")
        runner.logger.info(f"{'='*70}\n")

    runner.end()
    runner.logger.info("✓ THRML signal processing demonstration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
