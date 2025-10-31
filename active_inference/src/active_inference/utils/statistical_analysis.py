"""Statistical analysis utilities for active inference.

Provides comprehensive statistical analysis including:
- Linear regression analysis
- Correlation analysis
- Significance testing
- Effect size calculations
- Statistical reporting

All methods are real implementations using JAX for computation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


@dataclass
class RegressionResults:
    """Results from linear regression analysis.

    Attributes:
        slope: Regression slope (β1)
        intercept: Regression intercept (β0)
        r_squared: Coefficient of determination
        p_value: P-value for significance test
        std_error: Standard error of slope
        predictions: Predicted values
        residuals: Residuals (actual - predicted)

    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    predictions: Float[Array, "n"]
    residuals: Float[Array, "n"]

    def __str__(self) -> str:
        return (
            f"Linear Regression Results:\n"
            f"  y = {self.intercept:.4f} + {self.slope:.4f}x\n"
            f"  R² = {self.r_squared:.4f}\n"
            f"  p-value = {self.p_value:.4e}\n"
            f"  std error = {self.std_error:.4f}"
        )


@dataclass
class CorrelationResults:
    """Results from correlation analysis.

    Attributes:
        correlation: Pearson correlation coefficient
        p_value: P-value for significance test
        confidence_interval: 95% confidence interval
        n_samples: Number of samples used

    """

    correlation: float
    p_value: float
    confidence_interval: Tuple[float, float]
    n_samples: int

    def __str__(self) -> str:
        return (
            f"Correlation Results:\n"
            f"  r = {self.correlation:.4f}\n"
            f"  p-value = {self.p_value:.4e}\n"
            f"  95% CI = [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
            f"  n = {self.n_samples}"
        )


def linear_regression(
    x: Float[Array, "n"], y: Float[Array, "n"], compute_diagnostics: bool = True
) -> RegressionResults:
    """Perform linear regression analysis.

    Fits: y = β0 + β1*x + ε

    **Arguments:**

    - `x`: Independent variable
    - `y`: Dependent variable
    - `compute_diagnostics`: Whether to compute full diagnostics

    **Returns:**

    - RegressionResults with slope, intercept, R², p-value, etc.

    **Example:**
    ```python
    # Analyze relationship between free energy and performance
    results = linear_regression(free_energies, rewards)
    print(results)
    print(f"For each unit decrease in FE, reward increases by {-results.slope:.3f}")
    ```
    """
    # Convert to float arrays
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)

    n = len(x)

    # Calculate means
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)

    # Calculate slope and intercept
    numerator = jnp.sum((x - x_mean) * (y - y_mean))
    denominator = jnp.sum((x - x_mean) ** 2)
    slope = numerator / (denominator + 1e-10)
    intercept = y_mean - slope * x_mean

    # Predictions and residuals
    y_pred = intercept + slope * x
    residuals = y - y_pred

    # R-squared
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-10))

    if compute_diagnostics:
        # Standard error
        mse = ss_res / (n - 2)
        std_error = jnp.sqrt(mse / (denominator + 1e-10))

        # T-statistic and p-value
        t_stat = slope / (std_error + 1e-10)
        # Approximate p-value using normal distribution for large n
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(float(t_stat))))
    else:
        std_error = 0.0
        p_value = 1.0

    return RegressionResults(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        p_value=float(p_value),
        std_error=float(std_error),
        predictions=y_pred,
        residuals=residuals,
    )


def pearson_correlation(
    x: Float[Array, "n"], y: Float[Array, "n"], compute_significance: bool = True
) -> CorrelationResults:
    """Compute Pearson correlation coefficient with significance testing.

    **Arguments:**

    - `x`: First variable
    - `y`: Second variable
    - `compute_significance`: Whether to compute p-value and CI

    **Returns:**

    - CorrelationResults with correlation, p-value, and confidence interval

    **Example:**
    ```python
    # Correlation between precision and policy entropy
    results = pearson_correlation(precisions, entropies)
    print(results)
    if results.p_value < 0.05:
        print("Significant correlation detected!")
    ```
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)

    n = len(x)

    # Calculate correlation
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)

    numerator = jnp.sum((x - x_mean) * (y - y_mean))
    denominator = jnp.sqrt(jnp.sum((x - x_mean) ** 2) * jnp.sum((y - y_mean) ** 2))
    correlation = numerator / (denominator + 1e-10)

    if compute_significance:
        from scipy import stats

        # T-statistic
        t_stat = correlation * jnp.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
        p_value = 2 * (1 - stats.t.cdf(abs(float(t_stat)), n - 2))

        # Fisher's z-transformation for confidence interval
        z = 0.5 * jnp.log((1 + correlation) / (1 - correlation + 1e-10))
        z_se = 1 / jnp.sqrt(n - 3)
        z_ci_lower = z - 1.96 * z_se
        z_ci_upper = z + 1.96 * z_se

        # Transform back to correlation scale
        ci_lower = (jnp.exp(2 * z_ci_lower) - 1) / (jnp.exp(2 * z_ci_lower) + 1)
        ci_upper = (jnp.exp(2 * z_ci_upper) - 1) / (jnp.exp(2 * z_ci_upper) + 1)
    else:
        p_value = 1.0
        ci_lower, ci_upper = -1.0, 1.0

    return CorrelationResults(
        correlation=float(correlation),
        p_value=float(p_value),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        n_samples=int(n),
    )


def compute_effect_size(group1: Float[Array, "n1"], group2: Float[Array, "n2"]) -> Dict[str, float]:
    """Compute effect size measures between two groups.

    Computes Cohen's d and other effect size measures.

    **Arguments:**

    - `group1`: First group samples
    - `group2`: Second group samples

    **Returns:**

    - Dictionary with effect size measures

    **Example:**
    ```python
    # Compare two agent types
    effect_sizes = compute_effect_size(agent_a_rewards, agent_b_rewards)
    print(f"Cohen's d = {effect_sizes['cohens_d']:.3f}")
    ```
    """
    group1 = jnp.asarray(group1, dtype=jnp.float32)
    group2 = jnp.asarray(group2, dtype=jnp.float32)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = jnp.mean(group1), jnp.mean(group2)
    std1, std2 = jnp.std(group1, ddof=1), jnp.std(group2, ddof=1)

    # Cohen's d (pooled standard deviation)
    pooled_std = jnp.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / (pooled_std + 1e-10)

    # Glass's delta (using control group std)
    glass_delta = (mean1 - mean2) / (std2 + 1e-10)

    # Hedges' g (bias-corrected Cohen's d)
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
    hedges_g = cohens_d * correction

    return {
        "cohens_d": float(cohens_d),
        "glass_delta": float(glass_delta),
        "hedges_g": float(hedges_g),
        "mean_diff": float(mean1 - mean2),
        "pooled_std": float(pooled_std),
    }


def t_test_independent(
    group1: Float[Array, "n1"], group2: Float[Array, "n2"], equal_var: bool = True
) -> Dict[str, float]:
    """Perform independent samples t-test.

    **Arguments:**

    - `group1`: First group samples
    - `group2`: Second group samples
    - `equal_var`: Assume equal variances (Welch's t-test if False)

    **Returns:**

    - Dictionary with t-statistic, p-value, degrees of freedom

    **Example:**
    ```python
    # Test if two methods differ significantly
    results = t_test_independent(method_a_scores, method_b_scores)
    if results['p_value'] < 0.05:
        print(f"Significant difference: t={results['t_statistic']:.3f}")
    ```
    """
    from scipy import stats

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    n1, n2 = len(group1), len(group2)
    if equal_var:
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite degrees of freedom
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": float(df),
        "significant_05": bool(p_value < 0.05),
        "significant_01": bool(p_value < 0.01),
    }


def anova_one_way(*groups: Float[Array, "n"]) -> Dict[str, float]:
    """Perform one-way ANOVA.

    **Arguments:**

    - `*groups`: Variable number of group arrays

    **Returns:**

    - Dictionary with F-statistic, p-value

    **Example:**
    ```python
    # Compare multiple precision settings
    results = anova_one_way(prec_low_rewards, prec_med_rewards, prec_high_rewards)
    if results['p_value'] < 0.05:
        print(f"Precision significantly affects performance: F={results['f_statistic']:.3f}")
    ```
    """
    from scipy import stats

    groups_np = [np.asarray(g) for g in groups]
    f_stat, p_value = stats.f_oneway(*groups_np)

    # Calculate eta-squared (effect size)
    all_data = np.concatenate(groups_np)
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_np)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    eta_squared = ss_between / (ss_total + 1e-10)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "significant_05": bool(p_value < 0.05),
        "significant_01": bool(p_value < 0.01),
    }


def compute_summary_statistics(data: Float[Array, "n"]) -> Dict[str, float]:
    """Compute comprehensive summary statistics.

    **Arguments:**

    - `data`: Input data array

    **Returns:**

    - Dictionary with mean, median, std, quartiles, etc.

    **Example:**
    ```python
    stats = compute_summary_statistics(free_energies)
    print(f"Mean FE: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"Median FE: {stats['median']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    ```
    """
    data = jnp.asarray(data, dtype=jnp.float32)

    return {
        "mean": float(jnp.mean(data)),
        "median": float(jnp.median(data)),
        "std": float(jnp.std(data)),
        "var": float(jnp.var(data)),
        "min": float(jnp.min(data)),
        "max": float(jnp.max(data)),
        "q25": float(jnp.percentile(data, 25)),
        "q75": float(jnp.percentile(data, 75)),
        "iqr": float(jnp.percentile(data, 75) - jnp.percentile(data, 25)),
        "range": float(jnp.max(data) - jnp.min(data)),
        "cv": float(jnp.std(data) / (jnp.mean(data) + 1e-10)),  # Coefficient of variation
        "skewness": float(jnp.mean(((data - jnp.mean(data)) / (jnp.std(data) + 1e-10)) ** 3)),
        "kurtosis": float(jnp.mean(((data - jnp.mean(data)) / (jnp.std(data) + 1e-10)) ** 4) - 3),
    }


def generate_statistical_report(data_dict: Dict[str, Float[Array, "n"]], compare_groups: bool = True) -> str:
    """Generate comprehensive statistical report.

    **Arguments:**

    - `data_dict`: Dictionary mapping variable names to data arrays
    - `compare_groups`: Whether to perform group comparisons

    **Returns:**

    - Formatted statistical report string

    **Example:**
    ```python
    data = {
        'high_precision': high_prec_rewards,
        'low_precision': low_prec_rewards,
    }
    report = generate_statistical_report(data)
    print(report)
    runner.save_report(report, "statistical_analysis.txt")
    ```
    """
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Summary statistics for each variable
    report.append("SUMMARY STATISTICS")
    report.append("-" * 70)
    for name, data in data_dict.items():
        stats = compute_summary_statistics(data)
        report.append(f"\n{name}:")
        report.append(f"  Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}")
        report.append(f"  Median [IQR]: {stats['median']:.4f} [{stats['q25']:.4f}, {stats['q75']:.4f}]")
        report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        report.append(f"  CV: {stats['cv']:.4f}")
        report.append(f"  Skewness: {stats['skewness']:.4f}")
        report.append(f"  Kurtosis: {stats['kurtosis']:.4f}")

    # Group comparisons
    if compare_groups and len(data_dict) >= 2:
        report.append("")
        report.append("=" * 70)
        report.append("GROUP COMPARISONS")
        report.append("-" * 70)

        names = list(data_dict.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                data1, data2 = data_dict[name1], data_dict[name2]

                report.append(f"\n{name1} vs {name2}:")

                # T-test
                t_results = t_test_independent(data1, data2)
                report.append(f"  t-test: t={t_results['t_statistic']:.4f}, p={t_results['p_value']:.4e}")
                if t_results["significant_05"]:
                    report.append("  *** Significant at α=0.05")

                # Effect size
                effect = compute_effect_size(data1, data2)
                report.append(f"  Effect size (Cohen's d): {effect['cohens_d']:.4f}")
                report.append(f"  Mean difference: {effect['mean_diff']:.4f}")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)
