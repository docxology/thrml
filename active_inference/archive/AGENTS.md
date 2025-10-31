# Archive: Analysis Scripts

## Analysis Scripts

### `analyze_seed_42.py`
**Purpose**: Detailed statistical analysis of Example 06 with seed 42.

**Algorithm**:
1. Reproduce Example 06 coin flip inference with seed 42
2. Analyze observed frequency vs true bias
3. Calculate z-score and p-value
4. Validate credible interval coverage
5. Provide statistical interpretation

**Outputs**:
- Statistical summary of seed 42 behavior
- Z-score: 1.75 (within 2σ)
- P-value: ~0.08 (not rare)
- Interpretation: Result is within normal sampling variance

**Key Insight**: Observing 78 heads in 100 flips (when true bias is 0.7) is uncommon but expected ~5% of the time.

**Usage**:
```bash
python3 archive/analyze_seed_42.py
```

---

### `coin_flip_seed_comparison.py`
**Purpose**: Compare inference across 100 different random seeds.

**Algorithm**:
1. Run coin flip inference with 100 different seeds
2. Collect final estimates and credible intervals
3. Calculate coverage probability (% intervals containing true bias)
4. Plot distribution of estimates

**Analysis**:
- Mean estimate across seeds
- Standard deviation of estimates
- Coverage probability of 95% credible intervals
- Distribution of observed heads counts

**Expected Results**:
- Mean estimate ≈ 0.7 (true bias)
- Standard deviation ≈ 0.046
- Coverage probability ≈ 0.95 (target)
- Distribution of heads: Normal(70, 4.6²)

**Validation**: Coverage probability ~95% confirms correct Bayesian inference.

**Usage**:
```bash
python3 archive/coin_flip_seed_comparison.py
```

**Outputs**:
- Summary statistics across 100 seeds
- Coverage probability report
- Distribution plots
- Validation metrics

---

### `coin_flip_variance_demo.py`
**Purpose**: Demonstrate sampling variance across 10,000 experiments.

**Algorithm**:
1. Run 10,000 independent experiments
2. Each experiment: 100 coin flips with true bias 0.7
3. Record observed heads count for each
4. Compare empirical distribution to theoretical

**Statistical Analysis**:
- Theoretical mean: 70 heads
- Theoretical std: 4.58 heads
- Empirical verification of distribution
- Comparison with seed 42 result (78 heads)

**Key Results**:
- 68% of experiments: 65-75 heads (within 1σ)
- 95% of experiments: 61-79 heads (within 2σ)
- 99.7% of experiments: 56-84 heads (within 3σ)
- Seed 42 (78 heads): 1.75σ from mean - normal range

**Visualization**:
- Histogram of heads counts (10,000 experiments)
- Normal distribution overlay
- Marked position of seed 42 result
- Standard deviation ranges

**Usage**:
```bash
python3 archive/coin_flip_variance_demo.py
```

**Outputs**:
- Distribution histogram
- Statistical summary
- Seed 42 position in distribution
- Validation that result is normal

---

## Purpose and Context

These scripts were developed to validate and explain the behavior of Example 06 (coin flip inference). They demonstrate that:

1. **Sampling Variance is Fundamental**: Random variation in observed data is expected
2. **Bayesian Inference is Correct**: Estimates properly reflect observed data
3. **Credible Intervals Work**: Coverage probability matches theoretical value
4. **Seed 42 is Normal**: The specific result is within expected range

## Statistical Concepts

### Sampling Variance

**Definition**: Natural variation in samples drawn from a distribution.

**For Coin Flips**:
- True bias: p = 0.7
- Number of flips: n = 100
- Expected heads: E[X] = np = 70
- Standard deviation: σ = √(np(1-p)) = 4.58

**Implication**: Observing 65-75 heads is common, 61-79 is within 2σ (95% range).

### Bayesian Estimation

**Approach**: Update beliefs using observed data via Bayes' rule.

**For Beta-Binomial**:
- Prior: Beta(α₀, β₀)
- Observed: k heads in n flips
- Posterior: Beta(α₀ + k, β₀ + n - k)

**Property**: Posterior mean = (α₀ + k) / (α₀ + β₀ + n)

**With Weak Prior** (α₀ = β₀ = 1):
- Posterior mean ≈ (k + 1) / (n + 2)
- Dominated by observed data

### Credible Intervals

**Definition**: Bayesian interval containing parameter with specified probability.

**95% Credible Interval**: [q₀.₀₂₅, q₀.₉₇₅] where q are posterior quantiles.

**Correct Behavior**: Across many experiments, ~95% of 95% intervals contain true parameter.

**Validation**: `coin_flip_seed_comparison.py` verifies this property.

## Integration with Examples

These scripts support:
- [Example 06](../examples/06_coin_flip_inference.py): Main coin flip inference demo
- [EXAMPLE_06_ANALYSIS.md](../examples/EXAMPLE_06_ANALYSIS.md): Detailed analysis
- [SAMPLING_VARIANCE_EXPLANATION.md](../examples/SAMPLING_VARIANCE_EXPLANATION.md): Statistical explanation

## When to Run

Run these scripts to:
- **Validate Correctness**: Confirm Bayesian inference is working properly
- **Understand Results**: Interpret "unexpected" observed frequencies
- **Teach Concepts**: Demonstrate sampling variance and Bayesian inference
- **Debug**: Investigate potential issues with inference

## Expected Runtime

- `analyze_seed_42.py`: ~5 seconds
- `coin_flip_seed_comparison.py`: ~1-2 minutes (100 runs)
- `coin_flip_variance_demo.py`: ~2-3 minutes (10,000 runs)

## Dependencies

All scripts require:
- `jax` and `jax.numpy`
- `matplotlib`
- `numpy`
- `scipy`
- `active_inference` package

Install with:
```bash
uv pip install -e ".[all]"
```

## Historical Note

These scripts were created during development to validate Example 06 behavior when seed 42 produced 78 heads (vs expected 70). Investigation confirmed this is correct Bayesian behavior demonstrating proper sampling variance.

The analysis shows:
1. ✅ Bayesian inference is working correctly
2. ✅ Credible intervals have proper coverage
3. ✅ Observed variance matches theory
4. ✅ Seed 42 result is within normal range

## References

**Bayesian Inference**:
- Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

**Sampling Variance**:
- Rice, J. A. (2006). *Mathematical Statistics and Data Analysis*. Duxbury Press.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Duxbury.

**Beta-Binomial Model**:
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

## Maintenance

These scripts are maintained for:
- Validation of correctness
- Educational purposes
- Reference implementation

For current active inference examples, see [examples/README.md](../examples/README.md).
