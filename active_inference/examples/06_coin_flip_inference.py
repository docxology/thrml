"""Example 06: Bayesian Coin Flip Inference with THRML Sampling.

Demonstrates:
- Bayesian parameter estimation with THRML sampling
- THRML-based posterior sampling for bias inference
- Comparison of THRML sampling vs analytical Beta-Binomial updates
- Sequential belief updating using THRML
- Convergence to true parameter
- Credible intervals from THRML samples
- Predictive distributions

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`) for posterior estimation
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- Comprehensive demonstration of THRML for discrete parameter estimation
- THRML sampling validated against analytical Beta-Binomial results
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import example utilities
from example_utils import ExampleRunner, calculate_entropy, create_figure, plot_line_series
from scipy import stats

from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "06_coin_flip_inference"


def beta_posterior(prior_alpha, prior_beta, n_heads, n_tails):
    """Calculate Beta posterior parameters."""
    posterior_alpha = prior_alpha + n_heads
    posterior_beta = prior_beta + n_tails
    return posterior_alpha, posterior_beta


def beta_mean(alpha, beta):
    """Calculate mean of Beta distribution."""
    return alpha / (alpha + beta)


def beta_mode(alpha, beta):
    """Calculate mode of Beta distribution."""
    if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
    return None


def beta_credible_interval(alpha, beta, confidence=0.95):
    """Calculate credible interval for Beta distribution."""
    lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
    upper = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
    return lower, upper


def main():
    """Run coin flip inference example."""
    # Initialize example runner with profiling
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
    true_bias = runner.get_config("true_bias", default=0.7)
    prior_alpha = runner.get_config("prior_alpha", default=1.0)
    prior_beta = runner.get_config("prior_beta", default=1.0)
    n_flips = runner.get_config("n_flips", default=1000)
    checkpoints = runner.get_config("checkpoints", default=[1, 5, 10, 20, 50, 100])
    n_iterations = runner.get_config("n_iterations", default=50)
    n_warmup = runner.get_config("n_warmup", default=10)
    # THRML sampling parameters for bias inference
    n_samples = runner.get_config("n_samples", default=200)
    thrml_n_warmup = runner.get_config("thrml_n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)
    n_bias_levels = runner.get_config("n_bias_levels", default=100)  # Discretize bias [0,1]

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  true_bias: {true_bias}")
    runner.logger.info(f"  n_flips: {n_flips}")

    # === 1. CONFIGURATION ===
    with runner.section("Configuration"):
        runner.logger.info(f"True coin bias: {true_bias}")
        runner.logger.info(f"Prior: Beta({prior_alpha}, {prior_beta})")
        runner.logger.info(f"Number of flips: {n_flips}")

        # Save configuration
        config = {
            "seed": seed,
            "true_bias": true_bias,
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "n_flips": n_flips,
            "checkpoints": checkpoints,
            "n_samples": n_samples,
            "thrml_n_warmup": thrml_n_warmup,
            "steps_per_sample": steps_per_sample,
            "n_bias_levels": n_bias_levels,
        }
        runner.save_config(config)

    # === 2. GENERATE DATA ===
    with runner.section("Data Generation"):
        # Generate coin flips from true bias
        key, subkey = jax.random.split(key)
        flips = jax.random.bernoulli(subkey, true_bias, shape=(n_flips,))
        flips = np.array(flips, dtype=int)

        n_heads = np.sum(flips)
        n_tails = n_flips - n_heads
        observed_frequency = n_heads / n_flips

        runner.logger.info(f"Generated {n_flips} flips")
        runner.logger.info(f"Observed: {n_heads} heads, {n_tails} tails")
        runner.logger.info(f"Observed frequency: {observed_frequency:.3f}")
        runner.logger.info(f"True bias: {true_bias:.3f}")
        runner.logger.info(f"Estimation error: {abs(observed_frequency - true_bias):.3f}")

        # Validate data
        runner.validate_data(
            flips,
            "coin_flips",
            checks={
                "binary": lambda d: np.all((d == 0) | (d == 1)),
                "correct_length": lambda d: len(d) == n_flips,
            },
        )

    # === 2.5. CREATE THRML MODEL FOR BIAS INFERENCE ===
    with runner.section("THRML Model Setup"):
        runner.logger.info("Creating THRML generative model for bias inference")
        runner.logger.info(f"Discretizing bias space into {n_bias_levels} levels")

        # Create discrete bias model: states = bias levels, observations = heads/tails
        bias_grid = jnp.linspace(0.0, 1.0, n_bias_levels)

        # Observation model: P(heads|bias) = bias, P(tails|bias) = 1-bias
        A = jnp.zeros((2, n_bias_levels))  # 2 obs (heads=0, tails=1), n_bias_levels states
        A = A.at[0, :].set(bias_grid)  # P(heads | bias_i) = bias_i
        A = A.at[1, :].set(1.0 - bias_grid)  # P(tails | bias_i) = 1 - bias_i

        # Prior: Beta distribution discretized over bias grid
        prior_bias_probs = jnp.array([stats.beta.pdf(b, prior_alpha, prior_beta) for b in bias_grid])
        prior_bias_probs = prior_bias_probs / (jnp.sum(prior_bias_probs) + 1e-16)

        # Transition: no dynamics (static parameter)
        B = jnp.eye(n_bias_levels)[jnp.newaxis, :, :]  # Identity for single action

        # Preferences: neutral
        C = jnp.zeros(2)

        # Create generative model
        bias_model = GenerativeModel(
            n_states=n_bias_levels,
            n_observations=2,
            n_actions=1,
            A=A,
            B=B,
            C=C,
            D=prior_bias_probs,
        )

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=bias_model,
            n_samples=n_samples,
            n_warmup=thrml_n_warmup,
            steps_per_sample=steps_per_sample,
        )

        runner.logger.info(f"THRML model: {n_bias_levels} bias levels, 2 observations")
        runner.logger.info(
            f"THRML engine: {n_samples} samples, {thrml_n_warmup} warmup, {steps_per_sample} steps/sample"
        )
        runner.logger.info("THRML methods: CategoricalNode, Block, BlockGibbsSpec, CategoricalEBMFactor")

    # === 3. SEQUENTIAL INFERENCE ===
    with runner.section("Sequential Bayesian Inference"):
        # Track posterior evolution (analytical)
        alphas = [prior_alpha]
        betas = [prior_beta]
        means = [beta_mean(prior_alpha, prior_beta)]
        modes = []
        credible_intervals = []
        entropies = []

        # Track THRML posterior evolution
        thrml_posteriors = []
        thrml_means = []
        thrml_kl_divs = []

        current_alpha = prior_alpha
        current_beta = prior_beta
        current_prior = prior_bias_probs.copy()

        runner.logger.info("\nSequential updates (analytical + THRML):")
        for i, flip in enumerate(flips, 1):
            # === ANALYTICAL UPDATE ===
            if flip == 1:  # Heads
                current_alpha += 1
            else:  # Tails
                current_beta += 1

            alphas.append(current_alpha)
            betas.append(current_beta)

            # Calculate statistics
            mean = beta_mean(current_alpha, current_beta)
            means.append(mean)

            mode = beta_mode(current_alpha, current_beta)
            modes.append(mode if mode is not None else mean)

            lower, upper = beta_credible_interval(current_alpha, current_beta)
            credible_intervals.append((lower, upper))

            # Calculate entropy (using scipy)
            entropy = stats.beta.entropy(current_alpha, current_beta)
            entropies.append(entropy)

            # === THRML INFERENCE ===
            observation = int(flip)  # 0=tails, 1=heads

            # Update prior for THRML
            key, subkey_thrml = jax.random.split(key)
            model_with_prior = eqx.tree_at(lambda m: m.D, bias_model, current_prior)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)

            # THRML sampling inference
            runner.logger.debug(f"  THRML: Running block Gibbs sampling (n_samples={n_samples})...")
            thrml_posterior = temp_engine.infer_with_sampling(
                key=subkey_thrml,
                observation=observation,
                n_state_samples=n_samples,
            )
            thrml_posteriors.append(thrml_posterior)

            # Calculate THRML mean (weighted average of bias grid)
            thrml_mean = float(jnp.sum(thrml_posterior * bias_grid))
            thrml_means.append(thrml_mean)

            # Compare THRML to analytical
            analytical_posterior = jnp.array([stats.beta.pdf(b, current_alpha, current_beta) for b in bias_grid])
            analytical_posterior = analytical_posterior / (jnp.sum(analytical_posterior) + 1e-16)
            kl_div = calculate_entropy(thrml_posterior) - jnp.sum(
                thrml_posterior * jnp.log(analytical_posterior + 1e-16)
            )
            thrml_kl_divs.append(float(kl_div))

            # Update prior for next step
            current_prior = thrml_posterior

            # Log checkpoints
            if i in checkpoints:
                runner.logger.info(f"\nAfter {i} flips:")
                runner.logger.info(f"  Analytical: Beta({current_alpha:.1f}, {current_beta:.1f}), mean={mean:.4f}")
                runner.logger.info(f"  THRML: mean={thrml_mean:.4f}, KL[THRML||Analytical]={kl_div:.6f} nats")
                runner.logger.info(
                    f"  THRML samples: {n_samples}, warmup: {thrml_n_warmup}, steps/sample: {steps_per_sample}"
                )
                runner.logger.info(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
                runner.logger.info(f"  Entropy: {entropy:.4f}")
                runner.logger.info(f"  Error: {abs(mean - true_bias):.4f}")

        # Final posterior
        final_alpha, final_beta = alphas[-1], betas[-1]
        final_mean = means[-1]
        final_lower, final_upper = credible_intervals[-1]

        runner.logger.info("\n=== Final Posterior ===")
        runner.logger.info(f"Beta({final_alpha:.1f}, {final_beta:.1f})")
        runner.logger.info(f"Mean: {final_mean:.4f}")
        runner.logger.info(f"95% CI: [{final_lower:.4f}, {final_upper:.4f}]")
        runner.logger.info(f"True bias in CI: {final_lower <= true_bias <= final_upper}")

    # === 4. SAVE RESULTS ===
    with runner.section("Data Saving"):
        results = {
            "flips": flips,
            "alphas": np.array(alphas),
            "betas": np.array(betas),
            "means": np.array(means),
            "modes": np.array(modes),
            "credible_intervals": np.array(credible_intervals),
            "entropies": np.array(entropies),
            "true_bias": true_bias,
            "observed_frequency": observed_frequency,
            # THRML results
            "thrml_posteriors": np.array(thrml_posteriors),
            "thrml_means": np.array(thrml_means),
            "thrml_kl_divs": np.array(thrml_kl_divs),
            "bias_grid": np.array(bias_grid),
            "n_bias_levels": n_bias_levels,
            "thrml_n_samples": n_samples,
            "thrml_n_warmup": thrml_n_warmup,
            "thrml_steps_per_sample": steps_per_sample,
        }
        runner.save_data(results, "inference_results")

    # === 5. VISUALIZATIONS ===
    with runner.section("Visualization"):
        # Plot 1: Posterior evolution
        fig, axes = create_figure(2, 2, figsize=(14, 10))

        # Mean estimate over time (Analytical vs THRML)
        ax = axes[0, 0]
        steps = np.arange(len(means))
        ax.plot(steps, means, "b-", label="Analytical mean", linewidth=2)
        ax.plot(steps[1:], thrml_means, "coral", "--", label="THRML mean", linewidth=2, alpha=0.8)
        ax.axhline(true_bias, color="r", linestyle="--", label="True bias", linewidth=2)
        ax.axhline(observed_frequency, color="g", linestyle=":", label="Observed frequency")
        ax.fill_between(
            steps[1:],
            [ci[0] for ci in credible_intervals],
            [ci[1] for ci in credible_intervals],
            alpha=0.3,
            label="95% CI",
        )
        ax.set_xlabel("Number of Flips")
        ax.set_ylabel("Estimated Bias")
        ax.set_title("Posterior Mean Evolution (Analytical vs THRML)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Posterior entropy over time
        ax = axes[0, 1]
        plot_line_series(
            ax,
            np.array(entropies),
            title="Posterior Uncertainty (Entropy)",
            xlabel="Number of Flips",
            ylabel="Entropy (nats)",
        )

        # Estimation error over time (Analytical vs THRML)
        ax = axes[1, 0]
        errors_analytical = np.abs(np.array(means) - true_bias)
        errors_thrml = np.abs(np.array(thrml_means) - true_bias)
        ax.semilogy(steps, errors_analytical, "b-", linewidth=2, label="Analytical")
        ax.semilogy(steps[1:], errors_thrml, "coral", "--", linewidth=2, label="THRML", alpha=0.8)
        ax.set_xlabel("Number of Flips")
        ax.set_ylabel("Absolute Error (log scale)")
        ax.set_title("Estimation Error: Analytical vs THRML")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # THRML vs Analytical comparison (KL divergence)
        ax = axes[1, 1]
        ax.semilogy(steps[1:], thrml_kl_divs, "purple", "-", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Number of Flips")
        ax.set_ylabel("KL Divergence (nats, log scale)")
        ax.set_title("THRML vs Analytical: KL[THRML||Analytical]")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig, "inference_evolution", formats=["png", "pdf"])
        plt.close(fig)

        # Plot 1b: Final posterior comparison (Analytical vs THRML)
        fig_final, ax_final = create_figure(1, 1, figsize=(10, 6))
        theta_grid = np.linspace(0, 1, 1000)
        final_pdf = stats.beta.pdf(theta_grid, final_alpha, final_beta)
        ax_final.plot(theta_grid, final_pdf, "b-", linewidth=2, label="Analytical Posterior")

        # THRML posterior (discretized)
        final_thrml = thrml_posteriors[-1] if thrml_posteriors else None
        if final_thrml is not None:
            # Interpolate THRML posterior to continuous grid for comparison
            from scipy.interpolate import interp1d

            interp_fn = interp1d(bias_grid, final_thrml, kind="linear", bounds_error=False, fill_value=0.0)
            thrml_pdf_interp = interp_fn(theta_grid)
            # Normalize
            thrml_pdf_interp = thrml_pdf_interp / (jnp.sum(thrml_pdf_interp) * (theta_grid[1] - theta_grid[0]))
            ax_final.plot(theta_grid, thrml_pdf_interp, "coral", "--", linewidth=2, label="THRML Posterior", alpha=0.8)

        ax_final.axvline(true_bias, color="r", linestyle="--", label="True bias", linewidth=2)
        ax_final.axvline(final_mean, color="g", linestyle=":", label="Analytical mean")
        if final_thrml is not None:
            ax_final.axvline(thrml_means[-1], color="orange", linestyle=":", label="THRML mean")
        ax_final.axvline(final_lower, color="gray", linestyle=":", alpha=0.5)
        ax_final.axvline(final_upper, color="gray", linestyle=":", alpha=0.5, label="95% CI")
        ax_final.fill_between(
            theta_grid,
            0,
            final_pdf,
            where=(theta_grid >= final_lower) & (theta_grid <= final_upper),
            alpha=0.2,
            color="orange",
        )
        ax_final.set_xlabel("Bias θ")
        ax_final.set_ylabel("Probability Density")
        ax_final.set_title("Final Posterior Comparison: Analytical vs THRML")
        ax_final.legend()
        ax_final.grid(True, alpha=0.3)
        plt.tight_layout()
        runner.save_plot(fig_final, "final_posterior_comparison", formats=["png", "pdf"])
        plt.close(fig_final)

        # Plot 2: Posterior evolution at checkpoints
        fig, axes = create_figure(2, 3, figsize=(15, 10))

        for idx, checkpoint in enumerate(checkpoints):
            ax = axes[idx // 3, idx % 3]
            alpha_cp = alphas[checkpoint]
            beta_cp = betas[checkpoint]

            pdf = stats.beta.pdf(theta_grid, alpha_cp, beta_cp)
            ax.plot(theta_grid, pdf, "b-", linewidth=2)
            ax.axvline(true_bias, color="r", linestyle="--", linewidth=2)
            ax.set_title(f"After {checkpoint} flips\nBeta({alpha_cp:.1f}, {beta_cp:.1f})")
            ax.set_xlabel("Bias θ")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig, "posterior_checkpoints", formats=["png", "pdf"])
        plt.close(fig)

    # === 6. METRICS ===
    with runner.section("Metrics"):
        runner.record_metric("n_flips", n_flips)
        runner.record_metric("n_heads", int(n_heads))
        runner.record_metric("n_tails", int(n_tails))
        runner.record_metric("observed_frequency", float(observed_frequency))
        runner.record_metric("true_bias", true_bias)
        runner.record_metric("final_mean_estimate", float(final_mean))
        runner.record_metric("final_estimation_error", float(abs(final_mean - true_bias)))
        runner.record_metric("final_entropy", float(entropies[-1]))
        runner.record_metric("true_in_credible_interval", bool(final_lower <= true_bias <= final_upper))
        runner.record_metric("credible_interval_width", float(final_upper - final_lower))

        # THRML metrics
        runner.record_metric("thrml_n_samples", n_samples)
        runner.record_metric("thrml_n_warmup", thrml_n_warmup)
        runner.record_metric("thrml_steps_per_sample", steps_per_sample)
        runner.record_metric("thrml_n_bias_levels", n_bias_levels)
        if thrml_means:
            runner.record_metric("thrml_final_mean", float(thrml_means[-1]))
            runner.record_metric("thrml_final_error", float(abs(thrml_means[-1] - true_bias)))
            runner.record_metric("thrml_final_kl_div", float(thrml_kl_divs[-1]))
            runner.record_metric("thrml_mean_kl_div", float(np.mean(thrml_kl_divs)))
            runner.record_metric("analytical_vs_thrml_mean_diff", float(abs(final_mean - thrml_means[-1])))

        # Convergence rate (error at different points)
        convergence_errors = {f"error_at_{cp}_flips": float(abs(means[cp] - true_bias)) for cp in checkpoints}
        for key, value in convergence_errors.items():
            runner.record_metric(key, value, log=False)

        # THRML convergence errors
        if thrml_means:
            thrml_convergence = {
                f"thrml_error_at_{cp}_flips": float(abs(thrml_means[cp - 1] - true_bias))
                for cp in checkpoints
                if cp > 0 and cp - 1 < len(thrml_means)
            }
            for key, value in thrml_convergence.items():
                runner.record_metric(key, value, log=False)

    # === 7. FINISH ===
    runner.end()
    runner.logger.info("✓ Example complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
