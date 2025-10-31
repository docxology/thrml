"""Example 10: Active Inference Fundamentals.

Demonstrates:
- Free energy minimization
- Belief updating mechanics
- Precision control effects
- Exploration vs exploitation
- Information gain
- Expected free energy components

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`)
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for energy-efficient inference
- Comprehensive demonstration of THRML inference in active inference fundamentals
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from example_utils import ExampleRunner, calculate_entropy, calculate_kl_divergence, create_figure, plot_distribution

from active_inference.core import GenerativeModel, variational_free_energy
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "10_active_inference_fundamentals"


def expected_free_energy(posterior, model, action):
    """Calculate expected free energy for an action (epistemic and pragmatic value)."""
    # Predict next state distribution
    predicted_state = jnp.dot(model.B[:, :, action].T, posterior)

    # Predict observation distribution
    predicted_obs = jnp.dot(model.A.T, predicted_state)

    # Epistemic value (information gain) - entropy of predicted observation
    epistemic_value = -calculate_entropy(predicted_obs)

    # Pragmatic value (expected utility) - dot product with preferences
    pragmatic_value = jnp.dot(predicted_obs, jnp.exp(model.C))

    # Expected free energy (negative because we minimize)
    efe = -pragmatic_value + epistemic_value

    return efe, epistemic_value, pragmatic_value


def main():
    """Run active inference fundamentals example."""
    runner = ExampleRunner(
        EXAMPLE_NAME,
        OUTPUT_BASE,
        enable_profiling=True,
        enable_validation=True,
    )
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=4)
    n_observations = runner.get_config("n_observations", default=4)
    n_actions = runner.get_config("n_actions", default=2)
    n_inference_iterations = runner.get_config("n_inference_iterations", default=16)
    observation_noise = runner.get_config("observation_noise", default=0.05)
    goal_state = runner.get_config("goal_state", default=3)
    goal_preference_strength = runner.get_config("goal_preference_strength", default=2.0)
    precision_values = runner.get_config("precision_values", default=[0.5, 1.0, 2.0, 5.0])
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

    # === 1. CONFIGURATION ===
    with runner.section("Configuration"):
        runner.logger.info(f"States: {n_states}")
        runner.logger.info(f"Observations: {n_observations}")
        runner.logger.info(f"Actions: {n_actions}")

        config = {
            "seed": seed,
            "n_states": n_states,
            "n_observations": n_observations,
            "n_actions": n_actions,
            "n_inference_iterations": n_inference_iterations,
            "observation_noise": observation_noise,
            "goal_state": goal_state,
            "goal_preference_strength": goal_preference_strength,
            "precision_values": precision_values,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "steps_per_sample": steps_per_sample,
        }
        runner.save_config(config)

    # === 2. CREATE GENERATIVE MODEL ===
    with runner.section("Generative Model"):
        # Observation model - identity with noise (using config)
        A = jnp.eye(n_observations, n_states) * (1.0 - observation_noise) + observation_noise / n_observations
        A = A / A.sum(axis=0, keepdims=True)

        # Transition model - cyclic dynamics
        B = jnp.zeros((n_states, n_states, n_actions))
        # Action 0: forward cycle
        B = B.at[1, 0, 0].set(1.0)
        B = B.at[2, 1, 0].set(1.0)
        B = B.at[3, 2, 0].set(1.0)
        B = B.at[0, 3, 0].set(1.0)
        # Action 1: backward cycle
        B = B.at[3, 0, 1].set(1.0)
        B = B.at[0, 1, 1].set(1.0)
        B = B.at[1, 2, 1].set(1.0)
        B = B.at[2, 3, 1].set(1.0)

        # Preferences - prefer goal_state (using config)
        C = jnp.zeros(n_states)
        C = C.at[goal_state].set(goal_preference_strength)

        # Prior - uniform
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

        runner.logger.info("Observation model (A): partial observability")
        runner.logger.info("Transition model (B): cyclic dynamics")
        runner.logger.info("Preferences (C): favor state 3")
        runner.logger.info("Prior (D): uniform")

        runner.validate_data(A, "observation_likelihood")
        runner.validate_data(B, "transition_model")
        runner.validate_data(C, "preferences")
        runner.validate_data(D, "prior")

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=model,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        )
        runner.logger.info(f"THRML engine: {n_samples} samples, {n_warmup} warmup, {steps_per_sample} steps/sample")

    # === 3. FREE ENERGY MINIMIZATION ===
    with runner.section("Free Energy Minimization"):
        # Start with uniform prior
        prior_belief = D.copy()

        # Observe state 2
        observation = 2

        runner.logger.info(f"\nObservation: {observation}")
        runner.logger.info(f"Prior belief: {prior_belief}")
        runner.logger.info(f"Prior entropy: {calculate_entropy(prior_belief):.4f} bits")

        # Perform inference using THRML sampling
        with runner.profile("State Inference"):
            key_infer, key = jax.random.split(key)
            # Update model prior for this inference step
            model_with_prior = eqx.tree_at(lambda m: m.D, model, prior_belief)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)
            posterior = temp_engine.infer_with_sampling(
                key=key_infer,
                observation=observation,
                n_state_samples=n_samples,
            )
            # Calculate free energy for compatibility
            vfe = variational_free_energy(observation, posterior, model)

        runner.logger.info(f"\nPosterior belief: {posterior}")
        runner.logger.info(f"Posterior entropy: {calculate_entropy(posterior):.4f} bits")
        runner.logger.info(f"Variational free energy: {vfe:.4f}")

        # Calculate KL divergence
        kl = calculate_kl_divergence(posterior, prior_belief)
        runner.logger.info(f"KL[posterior || prior]: {kl:.4f} nats")
        runner.logger.info(
            f"Information gain: {calculate_entropy(prior_belief) - calculate_entropy(posterior):.4f} bits"
        )

        runner.validate_data(
            posterior,
            "posterior_belief",
            checks={
                "valid_distribution": lambda d: abs(np.sum(d) - 1.0) < 1e-6,
            },
        )

    # === 4. PRECISION CONTROL ===
    with runner.section("Precision Control"):
        precisions = precision_values

        action_dists = []
        entropies = []

        for prec in precisions:
            # Calculate action distribution using softmax with precision
            # Simple policy: choose action based on expected free energy
            efes = []
            for action in range(n_actions):
                efe, _, _ = expected_free_energy(posterior, model, action)
                efes.append(efe)

            efes = jnp.array(efes)

            # Apply softmax with precision
            action_probs = jnp.exp(-prec * efes)
            action_probs = action_probs / jnp.sum(action_probs)

            action_dists.append(action_probs)
            entropy = calculate_entropy(action_probs)
            entropies.append(entropy)

            runner.logger.info(f"\nPrecision γ={prec}:")
            runner.logger.info(f"  Action probabilities: {action_probs}")
            runner.logger.info(f"  Policy entropy: {entropy:.4f} bits")
            runner.logger.info(f"  EFE values: {efes}")

        runner.logger.info("\n=== Precision Effects ===")
        runner.logger.info(f"Low precision (γ={precisions[0]}): High entropy = Exploratory")
        runner.logger.info(f"High precision (γ={precisions[-1]}): Low entropy = Exploitative")

    # === 5. EXPECTED FREE ENERGY COMPONENTS ===
    with runner.section("Expected Free Energy Analysis"):
        runner.logger.info("\nEFE components for each action:")

        efe_data = []
        for action in range(n_actions):
            efe, epistemic, pragmatic = expected_free_energy(posterior, model, action)

            runner.logger.info(f"\nAction {action}:")
            runner.logger.info(f"  Expected Free Energy: {float(efe):.4f}")
            runner.logger.info(f"  Epistemic value (info gain): {float(epistemic):.4f}")
            runner.logger.info(f"  Pragmatic value (utility): {float(pragmatic):.4f}")

            efe_data.append(
                {
                    "action": action,
                    "efe": float(efe),
                    "epistemic": float(epistemic),
                    "pragmatic": float(pragmatic),
                }
            )

    # === 6. SAVE RESULTS ===
    with runner.section("Data Saving"):
        results = {
            "prior_belief": np.array(prior_belief),
            "posterior_belief": np.array(posterior),
            "vfe": float(vfe),
            "kl_divergence": float(kl),
            "observation": observation,
            "precisions": np.array(precisions),
            "action_distributions": np.array(action_dists),
            "policy_entropies": np.array(entropies),
            "efe_components": efe_data,
        }
        runner.save_data(results, "fundamentals_results", format="json")

    # === 7. VISUALIZATIONS ===
    with runner.section("Visualization"):
        # Create main analysis figure
        fig, axes = create_figure(2, 3, figsize=(15, 10))

        # Prior and posterior
        ax = axes[0, 0]
        x = np.arange(n_states)
        width = 0.35
        ax.bar(x - width / 2, prior_belief, width, label="Prior", alpha=0.7)
        ax.bar(x + width / 2, posterior, width, label="Posterior", alpha=0.7)
        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        ax.set_title("Belief Update")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Observation likelihood
        ax = axes[0, 1]
        im = ax.imshow(np.array(A), cmap="Blues", aspect="auto")
        ax.set_xlabel("State")
        ax.set_ylabel("Observation")
        ax.set_title("Observation Model A[o|s]")
        plt.colorbar(im, ax=ax, label="P(o|s)")

        # Preferences
        ax = axes[0, 2]
        plot_distribution(
            ax,
            np.exp(C) / np.sum(np.exp(C)),
            title="Preferred Observations",
            xlabel="Observation",
            ylabel="Preference (normalized)",
        )

        # Precision effects on policy
        ax = axes[1, 0]
        for i, prec in enumerate(precisions):
            ax.bar(np.arange(n_actions) + i * 0.2, action_dists[i], width=0.2, label=f"γ={prec}")
        ax.set_xlabel("Action")
        ax.set_ylabel("Probability")
        ax.set_title("Precision Effects on Action Selection")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Policy entropy vs precision
        ax = axes[1, 1]
        ax.plot(precisions, entropies, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Precision γ")
        ax.set_ylabel("Policy Entropy (bits)")
        ax.set_title("Exploration-Exploitation Trade-off")
        ax.grid(True, alpha=0.3)

        # EFE components
        ax = axes[1, 2]
        actions = [d["action"] for d in efe_data]
        epistemic_vals = [d["epistemic"] for d in efe_data]
        pragmatic_vals = [d["pragmatic"] for d in efe_data]

        x = np.arange(len(actions))
        width = 0.35
        ax.bar(x - width / 2, epistemic_vals, width, label="Epistemic", alpha=0.7)
        ax.bar(x + width / 2, pragmatic_vals, width, label="Pragmatic", alpha=0.7)
        ax.set_xlabel("Action")
        ax.set_ylabel("Value")
        ax.set_title("EFE Components")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        runner.save_plot(fig, "fundamentals_analysis", formats=["png", "pdf"])
        plt.close(fig)

        # === ADDITIONAL VISUALIZATIONS ===

        # Transition matrices for both actions
        fig2, axes2 = create_figure(1, 2, figsize=(12, 5))

        for action in range(n_actions):
            ax = axes2[action]
            im = ax.imshow(np.array(B[:, :, action]).T, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
            ax.set_xlabel("Current State")
            ax.set_ylabel("Next State")
            ax.set_title(f"Transition Model B[s'|s,a={action}]")
            ax.set_xticks(range(n_states))
            ax.set_yticks(range(n_states))
            plt.colorbar(im, ax=ax, label="P(s'|s,a)")

            # Add text annotations
            for i in range(n_states):
                for j in range(n_states):
                    val = B[j, i, action]
                    if val > 0.01:
                        text = ax.text(
                            i,
                            j,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="white" if val > 0.5 else "black",
                            fontsize=10,
                        )

        plt.tight_layout()
        runner.save_plot(fig2, "transition_matrices", formats=["png", "pdf"])
        plt.close(fig2)

        # Information theory visualization
        fig3, axes3 = create_figure(2, 2, figsize=(12, 10))

        # 1. KL divergence decomposition
        ax = axes3[0, 0]
        kl_terms = []
        for s in range(n_states):
            if posterior[s] > 1e-10 and prior_belief[s] > 1e-10:
                kl_term = posterior[s] * (np.log(posterior[s]) - np.log(prior_belief[s]))
                kl_terms.append(kl_term)
            else:
                kl_terms.append(0)

        ax.bar(range(n_states), kl_terms, alpha=0.7, color="steelblue")
        ax.set_xlabel("State")
        ax.set_ylabel("KL Contribution (nats)")
        ax.set_title(f"KL Divergence Decomposition (Total: {kl:.4f} nats)")
        ax.grid(True, alpha=0.3, axis="y")

        # 2. Entropy comparison
        ax = axes3[0, 1]
        entropies_comp = [
            calculate_entropy(prior_belief),
            calculate_entropy(posterior),
            calculate_entropy(np.array([1 / n_states] * n_states)),  # max entropy
        ]
        labels_comp = ["Prior", "Posterior", "Max\nEntropy"]
        colors_comp = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        ax.bar(labels_comp, entropies_comp, alpha=0.7, color=colors_comp)
        ax.set_ylabel("Entropy (bits)")
        ax.set_title("Belief Entropy Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Precision vs determinism
        ax = axes3[1, 0]
        determinism_scores = []
        for i, action_dist in enumerate(action_dists):
            # Determinism = 1 - normalized entropy
            max_ent = np.log2(n_actions)
            ent = calculate_entropy(action_dist)
            determinism = 1 - (ent / max_ent) if max_ent > 0 else 1.0
            determinism_scores.append(determinism)

        ax.plot(precisions, determinism_scores, "ro-", linewidth=2, markersize=10, label="Determinism")
        ax.plot(
            precisions,
            np.array(entropies) / np.log2(n_actions),
            "bo-",
            linewidth=2,
            markersize=10,
            label="Normalized Entropy",
        )
        ax.set_xlabel("Precision γ")
        ax.set_ylabel("Score")
        ax.set_title("Precision Effects on Policy Determinism")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

        # 4. Expected outcomes under each action
        ax = axes3[1, 1]
        expected_states = []
        for action in range(n_actions):
            # Expected next state under this action
            next_state_dist = jnp.dot(B[:, :, action].T, posterior)
            expected_states.append(next_state_dist)

        x_pos = np.arange(n_states)
        width = 0.35
        for i, exp_state in enumerate(expected_states):
            offset = (i - n_actions / 2 + 0.5) * width
            ax.bar(x_pos + offset, exp_state, width, label=f"Action {i}", alpha=0.7)

        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        ax.set_title("Expected State Distribution by Action")
        ax.set_xticks(x_pos)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        runner.save_plot(fig3, "information_theory_analysis", formats=["png", "pdf"])
        plt.close(fig3)

    # === 8. METRICS ===
    with runner.section("Metrics"):
        runner.record_metric("observation", observation)
        runner.record_metric("prior_entropy_bits", float(calculate_entropy(prior_belief)))
        runner.record_metric("posterior_entropy_bits", float(calculate_entropy(posterior)))
        runner.record_metric(
            "information_gain_bits", float(calculate_entropy(prior_belief) - calculate_entropy(posterior))
        )
        runner.record_metric("kl_divergence_nats", float(kl))
        runner.record_metric("variational_free_energy", float(vfe))
        runner.record_metric("map_state", int(jnp.argmax(posterior)))
        runner.record_metric("posterior_confidence", float(jnp.max(posterior)))

        # Precision effects
        for i, prec in enumerate(precisions):
            runner.record_metric(f"policy_entropy_precision_{prec}", float(entropies[i]), log=False)

    runner.end()
    runner.logger.info("✓ Example complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
