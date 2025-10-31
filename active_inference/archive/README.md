# Archive

## Overview

The `archive/` directory contains historical analysis scripts and demonstrations that were used during development to validate the active inference implementation. These scripts provide detailed analysis of specific behaviors and statistical properties.

## Contents

### `analyze_seed_42.py`
**Purpose**: Detailed analysis of Example 06 behavior with seed 42.

**Analysis**:
- Demonstrates correct Bayesian inference behavior
- Shows that observed frequency (0.78 from 78/100 heads) correctly differs from true bias (0.7)
- Validates that sampling variance is expected and normal
- Confirms credible interval contains true parameter

**Key Finding**: With seed 42, observing 78 heads in 100 flips (1.75 σ from expected 70) is within normal sampling variance and happens ~5% of the time. The Bayesian estimate correctly reflects observed data.

**Usage**:
```bash
python3 archive/analyze_seed_42.py
```

**Output**: Detailed statistical analysis confirming correctness of Example 06.

---

### `coin_flip_seed_comparison.py`
**Purpose**: Compare inference results across 100 different random seeds.

**Analysis**:
- Runs coin flip inference with 100 different seeds
- Collects final posteriors, estimates, credible intervals
- Validates that true bias (0.7) is contained in credible intervals
- Shows distribution of estimates across seeds

**Key Finding**: Across 100 seeds, ~95% of credible intervals contain true bias (0.7), validating correct Bayesian inference. Estimates vary naturally due to sampling variance.

**Usage**:
```bash
python3 archive/coin_flip_seed_comparison.py
```

**Output**:
- Summary statistics across seeds
- Coverage probability of credible intervals
- Distribution of posterior estimates
- Validation that inference is statistically correct

---

### `coin_flip_variance_demo.py`
**Purpose**: Demonstrate sampling variance across 10,000 experiments.

**Analysis**:
- Runs 10,000 independent coin flip experiments
- True bias = 0.7, each experiment = 100 flips
- Shows distribution of observed heads counts
- Calculates theoretical vs empirical statistics

**Key Finding**:
- Expected heads: 70 (mean of distribution)
- Standard deviation: ~4.6 heads
- 68% of experiments: 65-75 heads
- 95% of experiments: 61-79 heads
- Seed 42 result (78 heads): 1.75 σ from mean - uncommon but not rare

**Usage**:
```bash
python3 archive/coin_flip_variance_demo.py
```

**Output**:
- Histogram of heads counts across 10,000 experiments
- Theoretical vs empirical statistics
- Validation that seed 42 result is within normal range

---

## Purpose

These scripts were created to:

1. **Validate Correctness**: Confirm that Example 06 (coin flip inference) exhibits correct Bayesian behavior
2. **Explain Variance**: Demonstrate that sampling variance is expected and normal
3. **Statistical Validation**: Show that credible intervals have correct coverage
4. **Educational**: Help users understand probabilistic behavior and sampling variance

## Key Insights

### Sampling Variance is Normal

When flipping a coin 100 times with true bias 0.7:
- **Expected outcome**: 70 heads (mean)
- **Standard deviation**: 4.6 heads
- **Normal range**: 61-79 heads (95% confidence)

The observed result in Example 06 (78 heads with seed 42) is:
- 1.75 standard deviations from the mean
- Happens ~5% of the time
- **Within normal range** - not an error

### Bayesian Inference is Correct

The Bayesian estimate (0.7745) after observing 78 heads:
- **Correctly** reflects the observed data
- Does **not** equal the true bias (0.7) because we only have finite data
- The 95% credible interval [0.689, 0.850] **does contain** true bias (0.7) ✓

Across multiple seeds:
- ~95% of 95% credible intervals contain the true parameter
- This confirms **correct Bayesian inference**

### This is a Feature, Not a Bug

The "discrepancy" between estimated (0.7745) and true (0.7) bias demonstrates:
- **Correct** probabilistic behavior
- **Proper** uncertainty quantification
- **Expected** sampling variance
- **Valid** Bayesian updating

## Integration with Examples

These analysis scripts support [Example 06](../examples/06_coin_flip_inference.py):
- Provide detailed statistical validation
- Explain observed behavior
- Demonstrate correctness across multiple runs

See also:
- [EXAMPLE_06_ANALYSIS.md](../examples/EXAMPLE_06_ANALYSIS.md) - Detailed analysis
- [SAMPLING_VARIANCE_EXPLANATION.md](../examples/SAMPLING_VARIANCE_EXPLANATION.md) - Statistical explanation
- [INVESTIGATION_SUMMARY.md](../examples/INVESTIGATION_SUMMARY.md) - Investigation summary

## When to Use

Run these scripts when:
- Validating statistical correctness
- Understanding sampling variance
- Teaching probabilistic concepts
- Investigating "unexpected" results

## Output

All scripts:
- Print detailed statistical analysis to console
- Save plots to `output/` directory
- Provide clear interpretation of results
- Include validation metrics

## Dependencies

- `jax` and `jax.numpy`
- `matplotlib` for visualization
- `numpy` for numerical operations
- `scipy` for statistical functions
- `active_inference` package

## Running All Scripts

```bash
cd active_inference

# Analyze seed 42
python3 archive/analyze_seed_42.py

# Compare across seeds
python3 archive/coin_flip_seed_comparison.py

# Variance demonstration
python3 archive/coin_flip_variance_demo.py
```

## Historical Context

These scripts were developed during investigation of Example 06 behavior. Initial observations of seed 42 producing 78 heads (vs expected 70) led to detailed statistical investigation that confirmed correct Bayesian behavior.

The investigation demonstrated that:
1. Sampling variance is fundamental to probabilistic inference
2. Bayesian estimates correctly reflect observed data
3. Credible intervals provide proper uncertainty quantification
4. Statistical validation across multiple seeds confirms correctness

## References

- **Bayesian Inference**: Gelman et al., "Bayesian Data Analysis" (2013)
- **Sampling Variance**: Rice, "Mathematical Statistics and Data Analysis" (2006)
- **Beta-Binomial Model**: Bishop, "Pattern Recognition and Machine Learning" (2006)

## Maintenance

These scripts are archived for reference and validation. They demonstrate correct behavior and provide educational value but are not part of the main active inference workflow.

For current examples and usage, see [examples/README.md](../examples/README.md).
