"""Demonstration: Coin Flip Sampling Variance

This script shows why getting 78/100 heads from a p=0.7 coin is perfectly normal.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt


def simulate_many_experiments(true_bias=0.7, n_flips=100, n_experiments=1000, seed=42):
    """Run many coin flip experiments to show sampling distribution."""
    key = jr.key(seed)
    keys = jr.split(key, n_experiments)

    # Vectorized: generate all experiments at once
    all_flips = jax.vmap(lambda k: jr.bernoulli(k, true_bias, shape=(n_flips,)))(keys)

    # Count heads in each experiment
    heads_counts = jnp.sum(all_flips, axis=1)
    observed_frequencies = heads_counts / n_flips

    return heads_counts, observed_frequencies


def main():
    print("=" * 70)
    print("Coin Flip Sampling Variance Demonstration")
    print("=" * 70)

    true_bias = 0.7
    n_flips = 100
    n_experiments = 10000

    print(f"\nTrue coin bias: {true_bias}")
    print(f"Flips per experiment: {n_flips}")
    print(f"Number of experiments: {n_experiments}")

    # Run simulations
    heads_counts, observed_freqs = simulate_many_experiments(true_bias, n_flips, n_experiments)

    # Calculate statistics
    mean_heads = float(jnp.mean(heads_counts))
    std_heads = float(jnp.std(heads_counts))

    mean_freq = float(jnp.mean(observed_freqs))
    std_freq = float(jnp.std(observed_freqs))

    # Theoretical values
    theoretical_mean = n_flips * true_bias
    theoretical_std = jnp.sqrt(n_flips * true_bias * (1 - true_bias))

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)

    print("\nNumber of Heads:")
    print(f"  Theoretical mean: {theoretical_mean:.1f}")
    print(f"  Observed mean:    {mean_heads:.1f}")
    print(f"  Theoretical std:  {float(theoretical_std):.2f}")
    print(f"  Observed std:     {std_heads:.2f}")

    print("\nObserved Frequency:")
    print(f"  Theoretical mean: {true_bias:.3f}")
    print(f"  Observed mean:    {mean_freq:.3f}")
    print(f"  Observed std:     {std_freq:.3f}")

    # How often do we get 78 or more heads?
    n_at_least_78 = int(jnp.sum(heads_counts >= 78))
    prob_at_least_78 = n_at_least_78 / n_experiments

    print("\n" + "=" * 70)
    print("Analysis of the '78 heads' observation:")
    print("=" * 70)
    print(f"  Experiments with ≥78 heads: {n_at_least_78} / {n_experiments}")
    print(f"  Probability: {prob_at_least_78:.1%}")
    print(f"  Z-score for 78 heads: {(78 - theoretical_mean) / theoretical_std:.2f}")
    print(f"\n  CONCLUSION: Getting 78/100 heads happens ~{prob_at_least_78:.0%} of the time!")
    print("  This is NORMAL sampling variation, NOT an error.")

    # Confidence intervals
    percentiles = jnp.percentile(heads_counts, jnp.array([2.5, 25, 50, 75, 97.5]))
    print("\n" + "=" * 70)
    print("Distribution of heads counts:")
    print("=" * 70)
    print(f"  2.5th percentile:  {int(percentiles[0])} heads")
    print(f"  25th percentile:   {int(percentiles[1])} heads")
    print(f"  Median:            {int(percentiles[2])} heads")
    print(f"  75th percentile:   {int(percentiles[3])} heads")
    print(f"  97.5th percentile: {int(percentiles[4])} heads")
    print(f"\n  78 heads is between {int(percentiles[3])}th and {int(percentiles[4])}th percentile")
    print("  i.e., in the upper quartile but well within 95% range")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of heads counts
    ax = axes[0, 0]
    ax.hist(heads_counts, bins=50, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(theoretical_mean, color="r", linestyle="--", linewidth=2, label=f"True mean ({theoretical_mean:.0f})")
    ax.axvline(78, color="g", linestyle="--", linewidth=2, label="Observed (78)")
    ax.axvline(mean_heads, color="b", linestyle=":", linewidth=2, label=f"Sample mean ({mean_heads:.1f})")
    ax.set_xlabel("Number of Heads")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Heads from {n_experiments} experiments")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of frequencies
    ax = axes[0, 1]
    ax.hist(observed_freqs, bins=50, density=True, alpha=0.7, edgecolor="black")
    ax.axvline(true_bias, color="r", linestyle="--", linewidth=2, label=f"True bias ({true_bias})")
    ax.axvline(0.78, color="g", linestyle="--", linewidth=2, label="Observed (0.78)")
    ax.axvline(mean_freq, color="b", linestyle=":", linewidth=2, label=f"Sample mean ({mean_freq:.3f})")
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Observed Frequencies")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CDF of heads counts
    ax = axes[1, 0]
    sorted_heads = jnp.sort(heads_counts)
    cdf = jnp.arange(1, len(sorted_heads) + 1) / len(sorted_heads)
    ax.plot(sorted_heads, cdf, linewidth=2)
    ax.axvline(70, color="r", linestyle="--", linewidth=2, label="Expected (70)")
    ax.axvline(78, color="g", linestyle="--", linewidth=2, label="Observed (78)")
    ax.axhline(prob_at_least_78, color="g", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Heads")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 1]
    from scipy import stats as sp_stats

    theoretical_quantiles = sp_stats.binom.ppf(jnp.linspace(0.01, 0.99, 100), n_flips, true_bias)
    observed_quantiles = jnp.percentile(heads_counts, jnp.linspace(1, 99, 100))
    ax.plot(theoretical_quantiles, observed_quantiles, "o", alpha=0.5)
    ax.plot([50, 90], [50, 90], "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    ax.set_title("Q-Q Plot (Binomial Distribution)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coin_flip_sampling_variance.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved to: coin_flip_sampling_variance.png")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThe example 06 result (78/100 heads) is CORRECT behavior:")
    print("  1. True bias is 0.7, so we expect ~70 heads on average")
    print(
        f"  2. Standard deviation is ~{float(theoretical_std):.1f}, so 78 is only {(78-70)/float(theoretical_std):.1f} σ away"
    )
    print(f"  3. This happens {prob_at_least_78:.0%} of the time with a fair coin of bias 0.7")
    print("  4. Bayesian inference CORRECTLY estimates 0.7745 from the data")
    print("  5. The 95% CI [0.689, 0.850] DOES contain the true bias 0.7")
    print("\nThis is sampling variance, not a bug. With more flips, it would converge closer to 0.7.")
    print("=" * 70)


if __name__ == "__main__":
    main()
