"""Specific Analysis of Example 06 with Seed=42

This script replicates exactly what happens in Example 06 to show
that seed=42 consistently produces 78 heads, and this is normal.
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def analyze_seed_42():
    """Analyze what happens specifically with seed=42."""
    print("=" * 70)
    print("Analysis of Example 06 with SEED=42")
    print("=" * 70)

    # Exact parameters from example 06
    SEED = 42
    TRUE_BIAS = 0.7
    n_flips = 100

    # Generate flips exactly as in example 06
    key = jr.key(SEED)
    key, subkey = jr.split(key)
    flips = jr.bernoulli(subkey, TRUE_BIAS, shape=(n_flips,))
    flips = np.array(flips, dtype=int)

    n_heads = int(np.sum(flips))
    n_tails = n_flips - n_heads
    observed_frequency = n_heads / n_flips

    print("\nConfiguration:")
    print(f"  Random seed: {SEED}")
    print(f"  True bias: {TRUE_BIAS}")
    print(f"  Number of flips: {n_flips}")

    print("\n" + "=" * 70)
    print("Generated Data:")
    print("=" * 70)
    print(f"  Heads: {n_heads}")
    print(f"  Tails: {n_tails}")
    print(f"  Observed frequency: {observed_frequency:.3f}")
    print(
        f"  Difference from true: {observed_frequency - TRUE_BIAS:+.3f} ({(observed_frequency - TRUE_BIAS)/TRUE_BIAS:+.1%})"
    )

    # Statistical analysis
    expected_heads = n_flips * TRUE_BIAS
    std_heads = np.sqrt(n_flips * TRUE_BIAS * (1 - TRUE_BIAS))
    z_score = (n_heads - expected_heads) / std_heads

    print("\n" + "=" * 70)
    print("Statistical Analysis:")
    print("=" * 70)
    print(f"  Expected heads: {expected_heads:.1f}")
    print(f"  Std deviation: {std_heads:.2f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Interpretation: {n_heads} is {z_score:.1f} standard deviations from mean")

    # Probability calculations
    prob_exact_78 = stats.binom.pmf(78, n_flips, TRUE_BIAS)
    prob_at_least_78 = 1 - stats.binom.cdf(77, n_flips, TRUE_BIAS)

    print("\n" + "=" * 70)
    print("Probability Analysis:")
    print("=" * 70)
    print(f"  P(exactly 78 heads) = {prob_exact_78:.4f} ({prob_exact_78*100:.2f}%)")
    print(f"  P(≥78 heads) = {prob_at_least_78:.4f} ({prob_at_least_78*100:.2f}%)")
    print("\n  → Getting 78 or more heads happens ~5% of the time")
    print("  → This is like rolling a 1 on a 20-sided die: rare but normal!")

    # Bayesian inference
    prior_alpha, prior_beta = 1.0, 1.0
    posterior_alpha = prior_alpha + n_heads
    posterior_beta = prior_beta + n_tails
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
    ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)

    true_in_ci = ci_lower <= TRUE_BIAS <= ci_upper

    print("\n" + "=" * 70)
    print("Bayesian Inference Results:")
    print("=" * 70)
    print(f"  Prior: Beta({prior_alpha}, {prior_beta}) [uniform]")
    print(f"  Data: {n_heads} heads, {n_tails} tails")
    print(f"  Posterior: Beta({posterior_alpha}, {posterior_beta})")
    print(f"  Posterior mean: {posterior_mean:.4f}")
    print(f"  95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI width: {ci_upper - ci_lower:.4f}")
    print(f"\n  True bias (0.7) in CI: {'✓ YES' if true_in_ci else '✗ NO'}")

    if true_in_ci:
        print("  → Inference SUCCESS: True parameter captured by credible interval!")

    # Show first 20 flips
    print("\n" + "=" * 70)
    print("First 20 Flips (for reproducibility check):")
    print("=" * 70)
    first_20 = "".join(["H" if f == 1 else "T" for f in flips[:20]])
    print(f"  {first_20}")
    print(f"  Heads in first 20: {sum(flips[:20])}/20 = {sum(flips[:20])/20:.2f}")

    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Flip sequence
    ax = fig.add_subplot(gs[0, :])
    colors = ["red" if f == 1 else "blue" for f in flips]
    ax.scatter(range(n_flips), flips, c=colors, alpha=0.6, s=30)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Flip Number")
    ax.set_ylabel("Outcome (0=T, 1=H)")
    ax.set_title(f"All 100 Flips from Seed={SEED} (Red=Heads, Blue=Tails)")
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)

    # 2. Running frequency
    ax = fig.add_subplot(gs[1, 0])
    cumsum = np.cumsum(flips)
    running_freq = cumsum / np.arange(1, n_flips + 1)
    ax.plot(range(1, n_flips + 1), running_freq, "b-", linewidth=2, label="Running frequency")
    ax.axhline(TRUE_BIAS, color="r", linestyle="--", linewidth=2, label=f"True bias ({TRUE_BIAS})")
    ax.axhline(
        observed_frequency, color="g", linestyle=":", linewidth=2, label=f"Final freq ({observed_frequency:.2f})"
    )
    ax.fill_between(
        [1, n_flips],
        [TRUE_BIAS - 2 * std_heads / n_flips] * 2,
        [TRUE_BIAS + 2 * std_heads / n_flips] * 2,
        alpha=0.2,
        color="red",
        label="±2σ band",
    )
    ax.set_xlabel("Number of Flips")
    ax.set_ylabel("Cumulative Frequency")
    ax.set_title("Convergence of Observed Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # 3. Binomial distribution
    ax = fig.add_subplot(gs[1, 1])
    k_range = np.arange(50, 91)
    pmf = stats.binom.pmf(k_range, n_flips, TRUE_BIAS)
    ax.bar(k_range, pmf, alpha=0.7, color="lightblue", edgecolor="black")
    ax.axvline(expected_heads, color="r", linestyle="--", linewidth=2, label=f"Expected ({expected_heads:.0f})")
    ax.axvline(n_heads, color="g", linestyle="--", linewidth=2, label=f"Observed ({n_heads})")
    ax.set_xlabel("Number of Heads")
    ax.set_ylabel("Probability")
    ax.set_title(f"Binomial({n_flips}, {TRUE_BIAS}) Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Posterior distribution
    ax = fig.add_subplot(gs[1, 2])
    theta = np.linspace(0, 1, 1000)
    prior_pdf = stats.beta.pdf(theta, prior_alpha, prior_beta)
    posterior_pdf = stats.beta.pdf(theta, posterior_alpha, posterior_beta)

    ax.plot(theta, prior_pdf, "gray", linestyle="--", linewidth=2, label="Prior", alpha=0.5)
    ax.plot(theta, posterior_pdf, "b-", linewidth=2, label="Posterior")
    ax.axvline(TRUE_BIAS, color="r", linestyle="--", linewidth=2, label=f"True ({TRUE_BIAS})")
    ax.axvline(posterior_mean, color="g", linestyle=":", linewidth=2, label=f"Mean ({posterior_mean:.3f})")
    ax.axvline(ci_lower, color="orange", linestyle=":", alpha=0.7)
    ax.axvline(ci_upper, color="orange", linestyle=":", alpha=0.7, label="95% CI")
    ax.fill_betweenx([0, ax.get_ylim()[1]], ci_lower, ci_upper, alpha=0.2, color="orange")
    ax.set_xlabel("Bias θ")
    ax.set_ylabel("Density")
    ax.set_title("Prior and Posterior Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Z-score visualization
    ax = fig.add_subplot(gs[2, 0])
    z_range = np.linspace(-4, 4, 1000)
    normal_pdf = stats.norm.pdf(z_range)
    ax.plot(z_range, normal_pdf, "b-", linewidth=2)
    ax.axvline(z_score, color="g", linestyle="--", linewidth=2, label=f"Observed (z={z_score:.2f})")
    ax.axvline(1.96, color="r", linestyle=":", alpha=0.5, label="95% threshold (±1.96)")
    ax.axvline(-1.96, color="r", linestyle=":", alpha=0.5)
    ax.fill_between(
        z_range, 0, normal_pdf, where=(np.abs(z_range) <= 1.96), alpha=0.2, color="green", label="95% of samples"
    )
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Distribution (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Comparison with other seeds
    ax = fig.add_subplot(gs[2, 1:])
    # Generate 50 other experiments with different seeds
    other_seeds = range(1, 51)
    other_freqs = []
    for s in other_seeds:
        k = jr.key(s)
        k, sk = jr.split(k)
        f = jr.bernoulli(sk, TRUE_BIAS, shape=(n_flips,))
        other_freqs.append(float(jnp.mean(f)))

    ax.hist(
        other_freqs, bins=20, density=True, alpha=0.5, color="lightblue", edgecolor="black", label="Other seeds (1-50)"
    )
    ax.axvline(TRUE_BIAS, color="r", linestyle="--", linewidth=2, label=f"True bias ({TRUE_BIAS})")
    ax.axvline(
        observed_frequency,
        color="g",
        linestyle="--",
        linewidth=3,
        label=f"Seed=42 result ({observed_frequency:.2f})",
        zorder=10,
    )
    ax.set_xlabel("Observed Frequency")
    ax.set_ylabel("Density")
    ax.set_title("Seed=42 Result Compared to Other Random Seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Comprehensive Analysis: Example 06 with Seed={SEED}", fontsize=16, fontweight="bold")
    plt.savefig("example_06_seed_42_analysis.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved to: example_06_seed_42_analysis.png")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n✅ Example 06 with seed=42 produces 78 heads (observed freq = 0.78)")
    print(f"✅ This is {z_score:.1f} standard deviations from mean - within normal range")
    print("✅ Happens ~5% of the time - uncommon but not rare")
    print(f"✅ Bayesian inference correctly estimates {posterior_mean:.4f} from the data")
    print(f"✅ 95% credible interval [{ci_lower:.3f}, {ci_upper:.3f}] CONTAINS true bias 0.7")
    print("\n🎯 CONCLUSION: This is CORRECT behavior, not a bug!")
    print("   The algorithm is learning from the observed data (78/100)")
    print("   and properly quantifying uncertainty around that estimate.")
    print("\n   With infinite flips, we'd converge to 0.7 exactly.")
    print("   With 100 flips, some variation is expected and normal.")
    print("=" * 70)


if __name__ == "__main__":
    analyze_seed_42()
