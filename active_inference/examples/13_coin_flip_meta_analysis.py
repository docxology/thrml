"""Example 13: Coin Flip Meta-Analysis

Comprehensive parameter sweep and meta-analysis of Bayesian coin flip inference.

**Parameters Swept:**
- Coin bias: 10 levels from 0.1 to 0.9
- Number of flips: 10 levels from 10 to 10,000 (log scale)
- Prior concentration: 3 levels (weak, medium, strong)

**Analyses:**
- Posterior accuracy vs ground truth
- Confidence (posterior concentration) evolution
- Sample efficiency
- Bias estimation error
- Convergence rates

**Total Conditions:** 10 × 10 × 3 = 300 parameter combinations

This demonstrates comprehensive experimental design and meta-analysis using
the active_inference framework's statistical analysis and validation tools.

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`) alongside analytical updates
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- Compares THRML sampling vs analytical Beta-Binomial for validation
- GPU-accelerated block Gibbs sampling for energy-efficient inference
- THRML sampling run on representative subset for performance/comparison
- Comprehensive demonstration of THRML for meta-analysis with statistical validation
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback: simple progress indicator
    class tqdm:
        """Fallback progress indicator when tqdm is not available."""

        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            if self.desc:
                print(f"{self.desc}: 0/{self.total if self.total else '?'}")

        def update(self, n=1):
            self.n += n
            if self.n % max(1, (self.total or 100) // 10) == 0:
                print(f"{self.desc}: {self.n}/{self.total if self.total else '?'}")

        def close(self):
            if self.desc:
                print(f"{self.desc}: Complete ({self.n}/{self.total if self.total else self.n})")

        def __iter__(self):
            if self.iterable is None:
                raise TypeError("Cannot iterate over non-iterable tqdm object")
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # For example_utils

# Import example utilities
from example_utils import ExampleRunner

from active_inference import visualization as viz
from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine
from active_inference.utils import (
    ResourceTracker,
    compute_summary_statistics,
    generate_statistical_report,
    linear_regression,
    pearson_correlation,
)


def beta_mean_var(alpha, beta):
    """Compute mean and variance of Beta distribution."""
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return mean, var


def update_beta_posterior(prior_alpha, prior_beta, n_heads, n_tails):
    """Update Beta posterior given observations."""
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    return post_alpha, post_beta


def run_coin_flip_trial(
    true_bias, n_flips, prior_alpha, prior_beta, key, use_thrml=False, thrml_engine=None, n_bias_levels=100
):
    """Run a single coin flip inference trial.

    Returns:
        Dictionary with results (includes THRML comparison if use_thrml=True)

    """
    # Generate coin flips
    flips = jax.random.bernoulli(key, p=true_bias, shape=(n_flips,))
    n_heads = int(jnp.sum(flips))
    n_tails = n_flips - n_heads

    # Compute analytical posterior
    post_alpha, post_beta = update_beta_posterior(prior_alpha, prior_beta, n_heads, n_tails)

    # Posterior statistics
    post_mean, post_var = beta_mean_var(post_alpha, post_beta)
    post_std = np.sqrt(post_var)

    # Prior statistics
    prior_mean, prior_var = beta_mean_var(prior_alpha, prior_beta)

    # Compute metrics
    bias_error = abs(post_mean - true_bias)
    confidence = 1.0 / post_std  # Inverse of std as confidence
    relative_error = bias_error / true_bias if true_bias > 0 else bias_error

    # 95% credible interval
    ci_lower = np.percentile(np.random.beta(post_alpha, post_beta, 10000), 2.5)
    ci_upper = np.percentile(np.random.beta(post_alpha, post_beta, 10000), 97.5)
    ci_width = ci_upper - ci_lower
    ci_contains_true = ci_lower <= true_bias <= ci_upper

    result = {
        "true_bias": true_bias,
        "n_flips": n_flips,
        "n_heads": n_heads,
        "n_tails": n_tails,
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "prior_mean": prior_mean,
        "post_alpha": post_alpha,
        "post_beta": post_beta,
        "post_mean": post_mean,
        "post_std": post_std,
        "bias_error": bias_error,
        "relative_error": relative_error,
        "confidence": confidence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_width,
        "ci_contains_true": ci_contains_true,
    }

    # Add THRML comparison if requested
    if use_thrml and thrml_engine is not None:
        import scipy.stats as stats

        # Discretize bias space
        bias_grid = jnp.linspace(0.0, 1.0, n_bias_levels)

        # Create prior over discretized bias
        prior_bias_probs = jnp.array([stats.beta.pdf(b, prior_alpha, prior_beta) for b in bias_grid])
        prior_bias_probs = prior_bias_probs / (jnp.sum(prior_bias_probs) + 1e-16)

        # Update prior sequentially for each flip
        current_prior = prior_bias_probs.copy()
        for flip in flips:
            observation = int(flip)
            key, subkey = jax.random.split(key)
            model_with_prior = eqx.tree_at(lambda m: m.D, thrml_engine.model, current_prior)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)
            posterior = temp_engine.infer_with_sampling(key=subkey, observation=observation)
            current_prior = posterior

        # Get THRML posterior statistics
        thrml_mean = float(jnp.sum(posterior * bias_grid))
        thrml_error = abs(thrml_mean - true_bias)

        # Compare to analytical (discretized)
        analytical_posterior = jnp.array([stats.beta.pdf(b, post_alpha, post_beta) for b in bias_grid])
        analytical_posterior = analytical_posterior / (jnp.sum(analytical_posterior) + 1e-16)

        # KL divergence
        kl_div = float(jnp.sum(posterior * jnp.log((posterior + 1e-16) / (analytical_posterior + 1e-16))))

        result.update(
            {
                "thrml_mean": thrml_mean,
                "thrml_error": thrml_error,
                "thrml_kl_div": kl_div,
            }
        )

    return result


def main():
    """Run comprehensive coin flip meta-analysis."""
    # Initialize runner
    output_base = Path(__file__).parent.parent / "output"
    runner = ExampleRunner(example_name="13_coin_flip_meta_analysis", output_base=output_base)

    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    true_biases_range = runner.get_config("true_biases_range", default=[0.1, 0.9])
    true_biases_n = runner.get_config("true_biases_n", default=10)
    n_flips_min = runner.get_config("n_flips_min", default=10)
    n_flips_max = runner.get_config("n_flips_max", default=100000)
    n_flips_levels_count = runner.get_config("n_flips_levels", default=10)
    n_repeats = runner.get_config("n_repeats", default=5)
    prior_configs_dict = runner.get_config(
        "prior_configs", default={"weak": [1.0, 1.0], "medium": [5.0, 5.0], "strong": [20.0, 20.0]}
    )
    ci_confidence = runner.get_config("ci_confidence", default=0.95)
    n_bootstrap_samples = runner.get_config("n_bootstrap_samples", default=10000)

    # THRML parameters
    use_thrml_comparison = runner.get_config("use_thrml_comparison", default=True)
    thrml_n_samples = runner.get_config("n_samples", default=200)
    thrml_n_warmup = runner.get_config("n_warmup", default=50)
    thrml_steps_per_sample = runner.get_config("steps_per_sample", default=5)
    n_bias_levels = runner.get_config("n_bias_levels", default=100)
    thrml_fraction = runner.get_config("thrml_fraction", default=0.1)  # Run THRML on 10% of trials

    # Convert to numpy arrays and tuples
    true_biases = np.linspace(true_biases_range[0], true_biases_range[1], true_biases_n)
    n_flips_levels = np.logspace(np.log10(n_flips_min), np.log10(n_flips_max), n_flips_levels_count, dtype=int)
    prior_configs = {k: tuple(v) for k, v in prior_configs_dict.items()}

    # Set random seed from config
    key = jax.random.PRNGKey(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  true_biases_range: {true_biases_range}")
    runner.logger.info(f"  n_flips_range: [{n_flips_min}, {n_flips_max}]")
    runner.logger.info(f"  n_repeats: {n_repeats}")

    print("\n" + "=" * 70)
    print("Example 13: Coin Flip Meta-Analysis")
    print("=" * 70 + "\n")

    # Initialize resource tracker
    tracker = ResourceTracker()
    tracker.start()

    # ========================================================================
    # Configuration
    # ========================================================================
    with runner.section("Configuration"):
        # Use config-loaded values

        config = {
            "true_biases": true_biases.tolist(),
            "n_flips_levels": n_flips_levels.tolist(),
            "prior_configs": {k: list(v) for k, v in prior_configs.items()},
            "n_repeats": n_repeats,
            "total_conditions": len(true_biases) * len(n_flips_levels) * len(prior_configs),
            "total_trials": len(true_biases) * len(n_flips_levels) * len(prior_configs) * n_repeats,
        }

        print(f"True biases: {len(true_biases)} levels")
        print(f"N flips: {len(n_flips_levels)} levels ({n_flips_levels[0]} to {n_flips_levels[-1]})")
        print(f"Prior types: {len(prior_configs)} types")
        print(f"Repeats per condition: {n_repeats}")
        print(f"Total conditions: {config['total_conditions']}")
        print(f"Total trials: {config['total_trials']}")

        # THRML setup
        thrml_engine = None
        if use_thrml_comparison:
            runner.logger.info("Setting up THRML inference engine for comparison")
            import scipy.stats as stats

            bias_grid = jnp.linspace(0.0, 1.0, n_bias_levels)
            A = jnp.zeros((2, n_bias_levels))
            A = A.at[0, :].set(bias_grid)
            A = A.at[1, :].set(1.0 - bias_grid)
            prior_bias_probs = jnp.ones(n_bias_levels) / n_bias_levels
            B = jnp.eye(n_bias_levels)[jnp.newaxis, :, :]
            C = jnp.zeros(2)
            bias_model = GenerativeModel(
                n_states=n_bias_levels, n_observations=2, n_actions=1, A=A, B=B, C=C, D=prior_bias_probs
            )
            thrml_engine = ThrmlInferenceEngine(
                model=bias_model,
                n_samples=thrml_n_samples,
                n_warmup=thrml_n_warmup,
                steps_per_sample=thrml_steps_per_sample,
            )
            config.update(
                {
                    "thrml_n_samples": thrml_n_samples,
                    "thrml_n_warmup": thrml_n_warmup,
                    "thrml_steps_per_sample": thrml_steps_per_sample,
                    "n_bias_levels": n_bias_levels,
                    "thrml_fraction": thrml_fraction,
                }
            )
            runner.logger.info(f"THRML will run on ~{thrml_fraction*100:.0f}% of trials")

        runner.save_config(config)

    # ========================================================================
    # Run Parameter Sweep
    # ========================================================================
    with runner.section("Parameter Sweep"):
        print("\nRunning comprehensive parameter sweep...")

        # Use config-loaded key
        results = []

        # Total iterations
        total_iters = len(true_biases) * len(n_flips_levels) * len(prior_configs) * n_repeats

        pbar = tqdm(total=total_iters, desc="Running trials")

        thrml_count = 0
        total_count = 0

        for bias in true_biases:
            for n_flips in n_flips_levels:
                for prior_name, (prior_alpha, prior_beta) in prior_configs.items():
                    for repeat in range(n_repeats):
                        key, subkey = jax.random.split(key)
                        total_count += 1

                        # Decide if we should run THRML (on subset)
                        should_use_thrml = (
                            use_thrml_comparison
                            and thrml_engine is not None
                            and (total_count - 1) % int(1.0 / thrml_fraction) == 0
                        )

                        result = run_coin_flip_trial(
                            true_bias=bias,
                            n_flips=n_flips,
                            prior_alpha=prior_alpha,
                            prior_beta=prior_beta,
                            key=subkey,
                            use_thrml=should_use_thrml,
                            thrml_engine=thrml_engine if should_use_thrml else None,
                            n_bias_levels=n_bias_levels,
                        )
                        result["prior_type"] = prior_name
                        result["repeat"] = repeat
                        result["used_thrml"] = should_use_thrml
                        results.append(result)

                        if should_use_thrml:
                            thrml_count += 1

                        pbar.update(1)

        pbar.close()

        print(f"✓ Completed {len(results)} trials")
        if use_thrml_comparison:
            print(f"✓ THRML comparison run on {thrml_count} trials ({thrml_count/len(results)*100:.1f}%)")

        # Save raw results
        runner.save_data({"trials": results}, "all_trials.json")

    # ========================================================================
    # Data Aggregation and Analysis
    # ========================================================================
    with runner.section("Data Aggregation"):
        print("\nAggregating results...")

        # Convert to arrays for analysis
        bias_errors = np.array([r["bias_error"] for r in results])
        confidences = np.array([r["confidence"] for r in results])
        n_flips_arr = np.array([r["n_flips"] for r in results])
        true_biases_arr = np.array([r["true_bias"] for r in results])
        ci_widths = np.array([r["ci_width"] for r in results])
        ci_coverage = np.array([r["ci_contains_true"] for r in results], dtype=float)

        # Aggregate by condition
        aggregated = {}

        for bias in true_biases:
            for n_flips in n_flips_levels:
                for prior_name in prior_configs.keys():
                    # Filter results for this condition
                    condition_results = [
                        r
                        for r in results
                        if r["true_bias"] == bias and r["n_flips"] == n_flips and r["prior_type"] == prior_name
                    ]

                    if condition_results:
                        key_tuple = (bias, n_flips, prior_name)
                        aggregated[key_tuple] = {
                            "mean_bias_error": np.mean([r["bias_error"] for r in condition_results]),
                            "std_bias_error": np.std([r["bias_error"] for r in condition_results]),
                            "mean_confidence": np.mean([r["confidence"] for r in condition_results]),
                            "mean_ci_width": np.mean([r["ci_width"] for r in condition_results]),
                            "ci_coverage_rate": np.mean([r["ci_contains_true"] for r in condition_results]),
                        }

        print(f"✓ Aggregated {len(aggregated)} conditions")

        # Save aggregated results
        agg_data = {f"{b}_{n}_{p}": v for (b, n, p), v in aggregated.items()}
        runner.save_data(agg_data, "aggregated_results.json")

    # ========================================================================
    # Statistical Analysis
    # ========================================================================
    with runner.section("Statistical Analysis"):
        print("\nPerforming statistical analysis...")

        # 1. Regression: Bias error vs N flips (log scale)
        log_n_flips = np.log10(n_flips_arr)
        reg_result = linear_regression(log_n_flips, bias_errors)
        print("\n1. Regression: Bias Error vs log10(N Flips)")
        print(reg_result)

        # 2. Correlation: Confidence vs N flips
        corr_result = pearson_correlation(n_flips_arr, confidences)
        print("\n2. Correlation: Confidence vs N Flips")
        print(corr_result)

        # 3. Summary statistics by prior type
        print("\n3. Summary Statistics by Prior Type")
        print("-" * 50)

        prior_stats = {}
        for prior_name in prior_configs.keys():
            prior_results = [r for r in results if r["prior_type"] == prior_name]
            prior_errors = np.array([r["bias_error"] for r in prior_results])
            stats = compute_summary_statistics(prior_errors)
            prior_stats[prior_name] = stats

            print(f"\n{prior_name.capitalize()} Prior:")
            print(f"  Mean error: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  Median error: {stats['median']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

        # 4. CI coverage analysis
        print("\n4. Credible Interval Coverage")
        print("-" * 50)
        overall_coverage = np.mean(ci_coverage)
        expected_coverage = ci_confidence
        print(f"Overall {ci_confidence*100:.0f}% CI coverage: {overall_coverage:.1%}")
        print(f"Expected coverage: {expected_coverage*100:.1f}%")
        print(f"Difference: {(overall_coverage - expected_coverage)*100:.1f} percentage points")

        # THRML comparison analysis (if used)
        thrml_analysis = {}
        if use_thrml_comparison:
            thrml_trials = [r for r in results if r.get("used_thrml", False)]
            if thrml_trials:
                thrml_errors = np.array([r["thrml_error"] for r in thrml_trials])
                analytical_errors_thrml = np.array([r["bias_error"] for r in thrml_trials])
                thrml_kl_divs = np.array([r["thrml_kl_div"] for r in thrml_trials])

                # Compare THRML vs analytical
                mean_thrml_error = np.mean(thrml_errors)
                mean_analytical_error = np.mean(analytical_errors_thrml)
                mean_kl_div = np.mean(thrml_kl_divs)

                print("\n5. THRML vs Analytical Comparison")
                print("-" * 50)
                print(f"Trials with THRML: {len(thrml_trials)}")
                print(f"Mean THRML error: {mean_thrml_error:.6f}")
                print(f"Mean analytical error: {mean_analytical_error:.6f}")
                print(f"Mean KL divergence: {mean_kl_div:.6f}")

                thrml_analysis = {
                    "n_thrml_trials": len(thrml_trials),
                    "mean_thrml_error": float(mean_thrml_error),
                    "mean_analytical_error": float(mean_analytical_error),
                    "mean_kl_divergence": float(mean_kl_div),
                    "error_ratio": (
                        float(mean_thrml_error / mean_analytical_error) if mean_analytical_error > 0 else 1.0
                    ),
                }

        # Save analysis results
        analysis_results = {
            "regression": {
                "slope": float(reg_result.slope),
                "intercept": float(reg_result.intercept),
                "r_squared": float(reg_result.r_squared),
                "p_value": float(reg_result.p_value),
            },
            "correlation": {
                "r": float(corr_result.correlation),
                "p_value": float(corr_result.p_value),
            },
            "prior_stats": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in prior_stats.items()},
            "ci_coverage": float(overall_coverage),
            "thrml_analysis": thrml_analysis,
        }
        runner.save_data(analysis_results, "statistical_analysis.json")

        # Generate comprehensive statistical report
        # Only include prior types that have data
        stat_report_data = {}
        for prior_name in prior_configs.keys():
            prior_data = np.array([r["bias_error"] for r in results if r["prior_type"] == prior_name])
            if len(prior_data) > 0:
                stat_report_data[f"{prior_name}_prior"] = prior_data

        # Only generate report if we have data for multiple groups
        if len(stat_report_data) >= 2:
            stat_report = generate_statistical_report(stat_report_data, compare_groups=True)
            runner.save_report(stat_report, "comprehensive_statistical_report.txt")
        else:
            runner.logger.warning("⚠️ Skipping statistical report: need at least 2 prior groups")

    # ========================================================================
    # Visualizations
    # ========================================================================
    with runner.section("Visualization"):
        print("\nGenerating visualizations...")

        # 1. Heatmap: Bias error vs true bias and N flips (weak prior)
        weak_results = [r for r in results if r["prior_type"] == "weak"]

        error_matrix = np.zeros((len(true_biases), len(n_flips_levels)))
        for i, bias in enumerate(true_biases):
            for j, n_flips in enumerate(n_flips_levels):
                cond_results = [r for r in weak_results if r["true_bias"] == bias and r["n_flips"] == n_flips]
                if cond_results:
                    error_matrix[i, j] = np.mean([r["bias_error"] for r in cond_results])

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            error_matrix,
            aspect="auto",
            cmap="hot_r",
            norm=LogNorm(vmin=max(error_matrix.min(), 1e-4), vmax=error_matrix.max()),
        )
        ax.set_xlabel("Number of Flips (log scale)", fontsize=12)
        ax.set_ylabel("True Coin Bias", fontsize=12)
        ax.set_title("Bias Estimation Error Heatmap (Weak Prior)", fontsize=14, fontweight="bold")

        # Set ticks
        ax.set_xticks(range(len(n_flips_levels)))
        ax.set_xticklabels([f"{int(n)}" for n in n_flips_levels], rotation=45)
        ax.set_yticks(range(len(true_biases)))
        ax.set_yticklabels([f"{b:.1f}" for b in true_biases])

        plt.colorbar(im, ax=ax, label="Mean Bias Error")
        plt.tight_layout()
        runner.save_plot(fig, "bias_error_heatmap", formats=["png", "pdf"])
        plt.close(fig)

        # 2. Scatter with regression: Bias error vs log(N flips)
        viz.plot_scatter_with_regression(
            np.log10(n_flips_arr),
            bias_errors,
            x_label="log10(Number of Flips)",
            y_label="Bias Estimation Error",
            title="Learning Curve: Error vs Sample Size",
            save_path=str(runner.get_plot_path("learning_curve_regression.png")),
        )

        # 3. Confidence evolution
        # Find the bias closest to 0.5 for plotting
        closest_bias_to_half = true_biases[np.argmin(np.abs(true_biases - 0.5))]

        fig, ax = plt.subplots(figsize=(10, 6))
        for prior_name in prior_configs.keys():
            prior_results = [
                r for r in results if r["prior_type"] == prior_name and np.isclose(r["true_bias"], closest_bias_to_half)
            ]
            n_flips_sorted = sorted(set([r["n_flips"] for r in prior_results]))
            mean_conf = []
            for n in n_flips_sorted:
                conf_vals = [r["confidence"] for r in prior_results if r["n_flips"] == n]
                mean_conf.append(np.mean(conf_vals))

            ax.plot(
                n_flips_sorted,
                mean_conf,
                "o-",
                label=f"{prior_name.capitalize()} Prior",
                linewidth=2,
                markersize=6,
                alpha=0.7,
            )

        ax.set_xlabel("Number of Flips", fontsize=12)
        ax.set_ylabel("Confidence (1/σ)", fontsize=12)
        ax.set_title(
            f"Confidence Evolution with Sample Size (Bias≈{closest_bias_to_half:.2f})", fontsize=14, fontweight="bold"
        )
        ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        runner.save_plot(fig, "confidence_evolution", formats=["png", "pdf"])
        plt.close(fig)

        # 4. Prior comparison boxplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bias error by prior
        prior_errors = [
            [r["bias_error"] for r in results if r["prior_type"] == "weak"],
            [r["bias_error"] for r in results if r["prior_type"] == "medium"],
            [r["bias_error"] for r in results if r["prior_type"] == "strong"],
        ]

        bp1 = axes[0].boxplot(prior_errors, tick_labels=["Weak", "Medium", "Strong"], patch_artist=True)
        for patch, color in zip(bp1["boxes"], ["lightblue", "lightgreen", "lightcoral"]):
            patch.set_facecolor(color)
        axes[0].set_ylabel("Bias Estimation Error", fontsize=12)
        axes[0].set_xlabel("Prior Type", fontsize=12)
        axes[0].set_title("Error Distribution by Prior Type", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")

        # CI width by prior
        prior_ci_widths = [
            [r["ci_width"] for r in results if r["prior_type"] == "weak"],
            [r["ci_width"] for r in results if r["prior_type"] == "medium"],
            [r["ci_width"] for r in results if r["prior_type"] == "strong"],
        ]

        bp2 = axes[1].boxplot(prior_ci_widths, tick_labels=["Weak", "Medium", "Strong"], patch_artist=True)
        for patch, color in zip(bp2["boxes"], ["lightblue", "lightgreen", "lightcoral"]):
            patch.set_facecolor(color)
        axes[1].set_ylabel("95% CI Width", fontsize=12)
        axes[1].set_xlabel("Prior Type", fontsize=12)
        axes[1].set_title("Credible Interval Width by Prior Type", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        runner.save_plot(fig, "prior_comparison", formats=["png", "pdf"])
        plt.close(fig)

        # 5. Correlation matrix
        analysis_data = {
            "log(N Flips)": np.log10(n_flips_arr),
            "Bias Error": bias_errors,
            "Confidence": confidences,
            "CI Width": ci_widths,
            "True Bias": true_biases_arr,
        }
        viz.plot_correlation_matrix(analysis_data, save_path=str(runner.get_plot_path("variable_correlations.png")))

        print("✓ Generated 5 comprehensive visualizations")

    # ========================================================================
    # Summary Metrics
    # ========================================================================
    with runner.section("Summary Metrics"):
        tracker.stop()

        # Calculate key metrics
        mean_bias_error = np.mean(bias_errors)
        median_bias_error = np.median(bias_errors)
        best_condition_error = np.min([agg["mean_bias_error"] for agg in aggregated.values()])
        worst_condition_error = np.max([agg["mean_bias_error"] for agg in aggregated.values()])

        # Find optimal condition
        optimal_condition = min(aggregated.items(), key=lambda x: x[1]["mean_bias_error"])
        opt_bias, opt_n, opt_prior = optimal_condition[0]

        print("\n" + "=" * 70)
        print("META-ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total trials run: {len(results)}")
        print(f"Conditions tested: {len(aggregated)}")
        print("\nOverall Performance:")
        print(f"  Mean bias error: {mean_bias_error:.6f}")
        print(f"  Median bias error: {median_bias_error:.6f}")
        print(f"  Best condition error: {best_condition_error:.6f}")
        print(f"  Worst condition error: {worst_condition_error:.6f}")
        print("\nOptimal Condition:")
        print(f"  True bias: {opt_bias:.1f}")
        print(f"  N flips: {opt_n}")
        print(f"  Prior type: {opt_prior}")
        print(f"  Mean error: {optimal_condition[1]['mean_bias_error']:.6f}")
        print(f"\n95% CI Coverage: {overall_coverage:.1%} (target: 95%)")
        print("\nRegression (Error vs log N):")
        print(f"  Slope: {reg_result.slope:.6f} (negative = learning)")
        print(f"  R² = {reg_result.r_squared:.4f}")
        print("=" * 70)

        # Record metrics
        runner.record_metric("total_trials", len(results))
        runner.record_metric("mean_bias_error", float(mean_bias_error))
        runner.record_metric("median_bias_error", float(median_bias_error))
        runner.record_metric("ci_coverage", float(overall_coverage))
        runner.record_metric("regression_r_squared", float(reg_result.r_squared))
        runner.record_metric("regression_slope", float(reg_result.slope))

        # Save resource report
        resource_report = tracker.generate_report()
        runner.save_report(resource_report, "resource_usage.txt")

    runner.end()
    print("\n✓ Meta-analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
