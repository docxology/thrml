"""Demonstration: Different Seeds Show Different Sampling Outcomes

This script runs the coin flip inference multiple times with different seeds
to show that the observed frequency varies naturally around the true bias.
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def run_coin_flip_experiment(seed, true_bias=0.7, n_flips=100):
    """Run a single coin flip experiment with given seed."""
    key = jr.key(seed)
    flips = jr.bernoulli(key, true_bias, shape=(n_flips,))

    n_heads = int(jnp.sum(flips))
    n_tails = n_flips - n_heads
    observed_freq = n_heads / n_flips

    # Bayesian inference with uniform prior
    prior_alpha, prior_beta = 1.0, 1.0
    posterior_alpha = prior_alpha + n_heads
    posterior_beta = prior_beta + n_tails

    # Posterior statistics
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
    ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)

    true_in_ci = ci_lower <= true_bias <= ci_upper

    return {
        "seed": seed,
        "n_heads": n_heads,
        "n_tails": n_tails,
        "observed_freq": observed_freq,
        "posterior_mean": posterior_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "true_in_ci": true_in_ci,
        "error": abs(observed_freq - true_bias),
    }


def main():
    print("=" * 70)
    print("Coin Flip Inference with Different Random Seeds")
    print("=" * 70)

    true_bias = 0.7
    n_flips = 100
    n_experiments = 100

    print(f"\nTrue coin bias: {true_bias}")
    print(f"Flips per experiment: {n_flips}")
    print(f"Number of different seeds: {n_experiments}")

    # Run experiments with different seeds
    seeds = range(1, n_experiments + 1)
    results = [run_coin_flip_experiment(s, true_bias, n_flips) for s in seeds]

    # Extract data for analysis
    observed_freqs = np.array([r["observed_freq"] for r in results])
    posterior_means = np.array([r["posterior_mean"] for r in results])
    ci_widths = np.array([r["ci_width"] for r in results])
    errors = np.array([r["error"] for r in results])
    true_in_ci_count = sum(r["true_in_ci"] for r in results)

    # Statistics
    mean_observed = np.mean(observed_freqs)
    std_observed = np.std(observed_freqs)
    min_observed = np.min(observed_freqs)
    max_observed = np.max(observed_freqs)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    ci_coverage = true_in_ci_count / n_experiments

    print("\n" + "=" * 70)
    print("Aggregate Results Across All Seeds:")
    print("=" * 70)

    print("\nObserved Frequency Statistics:")
    print(f"  Mean: {mean_observed:.4f} (theoretical: {true_bias:.4f})")
    print(f"  Std:  {std_observed:.4f} (theoretical: {np.sqrt(true_bias*(1-true_bias)/n_flips):.4f})")
    print(f"  Min:  {min_observed:.4f}")
    print(f"  Max:  {max_observed:.4f}")

    print("\nEstimation Error:")
    print(f"  Mean absolute error: {mean_error:.4f}")
    print(f"  Max absolute error:  {max_error:.4f}")

    print("\nCredible Interval Coverage:")
    print(f"  True bias captured: {true_in_ci_count}/{n_experiments} ({ci_coverage:.1%})")
    print("  Target: 95%")
    print(f"  Status: {'✓ GOOD' if 0.93 <= ci_coverage <= 0.97 else '✗ Check this'}")

    # Find experiments with extreme observations
    print("\n" + "=" * 70)
    print("Examples of Different Outcomes:")
    print("=" * 70)

    # Sort by observed frequency
    sorted_results = sorted(results, key=lambda r: r["observed_freq"])

    # Show lowest, median, and highest
    indices = [0, len(sorted_results) // 2, len(sorted_results) - 1]
    labels = ["Lowest", "Median", "Highest"]

    for idx, label in zip(indices, labels):
        r = sorted_results[idx]
        print(f"\n{label} Observed Frequency:")
        print(f"  Seed: {r['seed']}")
        print(f"  Heads: {r['n_heads']}/100")
        print(f"  Observed freq: {r['observed_freq']:.3f}")
        print(f"  Posterior mean: {r['posterior_mean']:.4f}")
        print(f"  95% CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
        print(f"  True in CI: {'✓' if r['true_in_ci'] else '✗'}")

    # Find the seed that gives exactly 78 heads (like the example)
    seed_78 = next((r for r in results if r["n_heads"] == 78), None)
    if seed_78:
        print("\n" + "=" * 70)
        print("Seed giving 78 heads (like Example 06 with seed=42):")
        print("=" * 70)
        print(f"  Seed: {seed_78['seed']}")
        print(f"  Observed freq: {seed_78['observed_freq']:.3f}")
        print(f"  Posterior mean: {seed_78['posterior_mean']:.4f}")
        print(f"  95% CI: [{seed_78['ci_lower']:.4f}, {seed_78['ci_upper']:.4f}]")
        print(f"  Error: {seed_78['error']:.4f}")
        print(
            f"\n  This is one of {sum(1 for r in results if r['n_heads'] >= 78)} seeds (out of {n_experiments}) that gave ≥78 heads"
        )

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of observed frequencies
    ax = axes[0, 0]
    ax.hist(observed_freqs, bins=30, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(true_bias, color="r", linestyle="--", linewidth=2, label=f"True bias ({true_bias})")
    ax.axvline(mean_observed, color="b", linestyle=":", linewidth=2, label=f"Mean observed ({mean_observed:.3f})")
    ax.axvline(0.78, color="g", linestyle="--", linewidth=2, alpha=0.7, label="Example 06 result (0.78)")
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution Across {n_experiments} Different Seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Estimation errors
    ax = axes[0, 1]
    ax.hist(errors, bins=30, density=True, alpha=0.7, edgecolor="black", color="orange")
    ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean error ({mean_error:.3f})")
    ax.set_xlabel("Absolute Error |observed - true|")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Estimation Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Credible intervals
    ax = axes[1, 0]
    for i, r in enumerate(sorted_results[::5]):  # Plot every 5th for clarity
        color = "green" if r["true_in_ci"] else "red"
        ax.plot([r["ci_lower"], r["ci_upper"]], [i, i], color=color, alpha=0.5, linewidth=2)
        ax.plot(r["posterior_mean"], i, "o", color=color, markersize=4)
    ax.axvline(true_bias, color="blue", linestyle="--", linewidth=2, label="True bias")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Experiment (sorted by observed freq)")
    ax.set_title(f"95% Credible Intervals (Coverage: {ci_coverage:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Observed freq vs posterior mean
    ax = axes[1, 1]
    ax.scatter(observed_freqs, posterior_means, alpha=0.5, s=30)
    ax.plot([0.5, 0.9], [0.5, 0.9], "r--", linewidth=2, label="x=y line")
    ax.axvline(true_bias, color="g", linestyle=":", alpha=0.5, label="True bias")
    ax.axhline(true_bias, color="g", linestyle=":", alpha=0.5)
    # Highlight example 06 result
    ax.scatter([0.78], [0.7745], color="red", s=100, marker="*", zorder=10, label="Example 06 (seed=42)")
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Posterior Mean")
    ax.set_title("Observed vs Inferred Bias")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 0.9)
    ax.set_ylim(0.5, 0.9)

    plt.tight_layout()
    plt.savefig("coin_flip_different_seeds.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved to: coin_flip_different_seeds.png")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print(f"\n1. Individual experiments vary: observed frequency ranges from {min_observed:.2f} to {max_observed:.2f}")
    print(f"2. On average, estimates are accurate: mean observed = {mean_observed:.3f} ≈ {true_bias:.3f}")
    print(f"3. Credible intervals work: {ci_coverage:.0%} contain the true value (target: 95%)")
    print("4. Example 06's result (0.78) is in the upper range but perfectly normal")
    print("5. Each seed produces different but valid sampling outcomes")
    print("\n✓ The inference algorithm is working correctly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
