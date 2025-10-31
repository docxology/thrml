# Utility Functions

## Overview

The `utils` module provides utility functions for evaluating, visualizing, and analyzing active inference agents and their behavior. These utilities support both development and research workflows.

## Modules

### `metrics.py`
Implements evaluation metrics:
- **calculate_kl_divergence**: KL divergence between distributions
- **calculate_prediction_accuracy**: Accuracy of observation predictions
- **calculate_policy_entropy**: Entropy of action distributions (available but not exported in __init__)
- **calculate_expected_utility**: Expected utility calculation (available but not exported in __init__)

### `visualization.py`
Implements plotting functions:
- **plot_belief_trajectory**: Plot belief evolution over time (heatmap + entropy)
- **plot_free_energy**: Plot free energy over time (with moving average)
- **plot_action_distribution**: Plot action probability distribution (bar chart)

## THRML Integration

### Current Usage
- Utilities work with standard arrays and distributions
- Compatible with THRML-generated data

### Integration Opportunities
1. **THRML Observers**: Use THRML observers for metrics collection
2. **Sampling Visualization**: Visualize THRML sampling trajectories
3. **Factor Visualization**: Visualize THRML factor graphs
4. **Performance Metrics**: Use THRML observers for performance tracking

## Dependencies

### Internal Dependencies
- `core`: For belief distributions and free energy
- `agents`: For agent state and behavior

### External Dependencies
- `jax.numpy`: Core numerical operations
- `matplotlib`: For visualization (optional)

### THRML Integration Points
- `thrml.observers`: For metrics collection (future)
- `thrml.factor`: For factor visualization (future)

## Usage

### Metrics
```python
from active_inference.utils import (
    calculate_kl_divergence,
    calculate_prediction_accuracy,
)
from active_inference.utils.metrics import (
    calculate_policy_entropy,
    calculate_expected_utility,
)

# KL divergence
kl = calculate_kl_divergence(posterior, prior)

# Prediction accuracy
accuracy = calculate_prediction_accuracy(
    predicted_observations,
    actual_observations,
)

# Policy entropy (direct import required)
entropy = calculate_policy_entropy(action_probs)
```

### Visualization
```python
from active_inference.utils import (
    plot_belief_trajectory,
    plot_free_energy,
    plot_action_distribution,
)

# Plot belief trajectory
fig, axes = plot_belief_trajectory(beliefs, true_states=true_states)

# Plot free energy
fig, ax = plot_free_energy(free_energies)

# Plot action distribution
fig, ax = plot_action_distribution(action_probs, action_names=["Up", "Right", "Down", "Left"])
```

## Design Principles

1. **Real Metrics**: Actual calculations, no approximations
2. **Modular**: Each function is independently usable
3. **JAX Compatible**: All functions work with JAX arrays
4. **Optional Visualization**: Matplotlib optional (graceful degradation)

## Testing

Comprehensive tests in `tests/test_integration.py`:
- Metric calculations (correctness)
- Visualization (no errors)
- Edge cases (empty data, single values)

## Future Enhancements

1. THRML observer integration
2. Interactive visualizations
3. More evaluation metrics
4. Performance profiling utilities
5. Factor graph visualization
