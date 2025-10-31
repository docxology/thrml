# Analysis and Validation Features

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [API Reference](api.md) | [Module Index](module_index.md) | [Workflows](workflows_patterns.md)

Comprehensive analysis, validation, and resource tracking capabilities.

## Overview

The active_inference framework includes comprehensive analysis, validation, and resource tracking capabilities.

## Statistical Analysis

### Linear Regression

Perform linear regression with full diagnostics:

```python
from active_inference.utils import linear_regression

results = linear_regression(x, y)
print(f"y = {results.intercept:.3f} + {results.slope:.3f}x")
print(f"R² = {results.r_squared:.3f}, p = {results.p_value:.4f}")
```

### Correlation Analysis

Compute Pearson correlation with significance testing:

```python
from active_inference.utils import pearson_correlation

corr = pearson_correlation(x, y)
print(f"r = {corr.correlation:.3f}, p = {corr.p_value:.4f}")
```

### Statistical Tests

```python
from active_inference.utils import t_test_independent, anova_one_way

# Independent t-test
t_results = t_test_independent(group1, group2)
print(f"t = {t_results['t_statistic']:.3f}, p = {t_results['p_value']:.4f}")

# One-way ANOVA
f_results = anova_one_way(group1, group2, group3)
print(f"F = {f_results['f_statistic']:.3f}, p = {f_results['p_value']:.4f}")
```

## Data Validation

### Basic Validation

```python
from active_inference.utils import DataValidator

validator = DataValidator()

# Validate distribution
result = validator.validate_distribution(belief, "belief")
if not result.passed:
    print(f"Validation failed: {result.message}")

# Validate model
results = validator.validate_generative_model(model)
validator.print_report()

# Generate HTML report
validator.generate_html_report(output_path)
```

## Resource Tracking

### Track Resources

```python
from active_inference.utils import ResourceTracker

tracker = ResourceTracker()
tracker.start()

# ... your code ...
tracker.snapshot("checkpoint")

tracker.stop()
report = tracker.generate_report()
print(report)
```

### Estimate Resources

```python
from active_inference.utils import estimate_resources

estimates = estimate_resources(
    n_states=10, n_observations=10, n_actions=4, n_steps=100
)
print(f"Estimated time: {estimates['time_seconds']:.1f}s")
print(f"Estimated memory: {estimates['memory_mb']:.1f}MB")
```

## Enhanced Visualization

### Regression Plots

```python
from active_inference import visualization as viz

# Scatter with regression line
viz.plot_scatter_with_regression(
    x, y,
    x_label="Free Energy",
    y_label="Reward",
    save_path="regression.png"
)
```

### Correlation Matrix

```python
data = {
    'Free Energy': fe_values,
    'Reward': rewards,
    'Entropy': entropies,
}
viz.plot_correlation_matrix(data, save_path="correlation.png")
```

### Residual Diagnostics

```python
viz.plot_residuals(x, y, save_path="residuals.png")
```

## Available Functions

### Statistical Analysis (`active_inference.utils`)
- `linear_regression(x, y)` - Linear regression with diagnostics
- `pearson_correlation(x, y)` - Correlation with significance testing
- `compute_effect_size(group1, group2)` - Effect size measures
- `t_test_independent(group1, group2)` - Independent t-test
- `anova_one_way(*groups)` - One-way ANOVA
- `compute_summary_statistics(data)` - Summary statistics
- `generate_statistical_report(data_dict)` - Comprehensive report

### Validation (`active_inference.utils`)
- `DataValidator` - Main validation class
  - `validate_array(array, name, ...)` - Array validation
  - `validate_distribution(dist, name)` - Distribution validation
  - `validate_generative_model(model)` - Model validation
  - `validate_trajectory(beliefs, actions, observations)` - Trajectory validation
  - `print_report()` - Print validation report
  - `generate_html_report(path)` - Generate HTML report

### Resource Tracking (`active_inference.utils`)
- `ResourceTracker` - Main tracking class
  - `start()`, `stop()` - Start/stop tracking
  - `take_snapshot(name)` - Take resource snapshot
  - `profile_section(name, start_snapshot)` - Profile code section
  - `generate_report()` - Generate text report
  - `save_report(path)` - Save report to file
- `estimate_resources(...)` - Estimate resource requirements
- `print_resource_estimates(estimates)` - Print estimates

### Enhanced Visualization (`active_inference.visualization`)
- `plot_scatter_with_regression(x, y, ...)` - Scatter with regression
- `plot_correlation_matrix(data, ...)` - Correlation heatmap
- `plot_residuals(x, y, ...)` - Residual diagnostics
- `plot_pairwise_relationships(data, ...)` - Pairwise scatter matrix

## Examples

See `examples/12_statistical_validation_demo.py` for a comprehensive demonstration of all analysis and validation features.

## Implementation Details

All methods use real calculations:
- Statistical tests use scipy for p-values and confidence intervals
- Validation performs actual mathematical checks
- Resource tracking uses psutil for system measurements
- All functions are JAX-compatible

---

## Cross-References

### Module Documentation
- [Utils Module](module_utils.md) - Complete utils API
- [Visualization Module](module_visualization.md) - Visualization tools

### Related Guides
- [Workflows & Patterns](workflows_patterns.md#validation-workflow) - Validation patterns
- [Performance Guide](performance.md) - Resource optimization
- [Getting Started](getting_started.md) - Basic usage

### Examples
- [Example 12: Statistical Validation Demo](../examples/12_statistical_validation_demo.py)
- [Example 13: Coin Flip Meta Analysis](../examples/13_coin_flip_meta_analysis.py)

---

> **Next**: [Utils Module](module_utils.md) | [Visualization Module](module_visualization.md) | [Workflows](workflows_patterns.md)
