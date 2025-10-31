# Core Active Inference Components

## Overview

The `core` module provides the fundamental mathematical components for active inference, including generative models, free energy calculations, and precision weighting. These components implement the theoretical foundation of the Free Energy Principle.

## Modules

### `generative_model.py`
Implements discrete generative models for POMDP-style active inference:
- **GenerativeModel**: Base class with A (observation likelihood), B (state transitions), C (preferences), D (state prior)
- **HierarchicalGenerativeModel**: Multi-level generative models for hierarchical inference
- **normalize_distribution**: Helper function to normalize arrays to sum to 1
- **softmax_stable**: Numerically stable softmax function

### `free_energy.py`
Implements free energy calculations:
- **variational_free_energy**: Calculates VFE for perception (inference quality)
- **expected_free_energy**: Calculates EFE for action selection (pragmatic + epistemic value)
- **batch_expected_free_energy**: Batch computation of EFE for all actions (not exported in __init__.py, import directly)

### `precision.py`
Implements precision weighting and message passing:
- **Precision**: Precision parameters (sensory, state, action)
- **PrecisionWeighting**: Utility functions for precision-weighted operations
- **Message**: Message passing infrastructure for hierarchical inference
- **MessageType**: Enum for message types (BOTTOM_UP, TOP_DOWN, LATERAL)

## Dependencies

### THRML Integration
- Currently uses JAX operations directly
- Future integration points for THRML:
  - Use `thrml.CategoricalNode` for state representations
  - Use `thrml.factor.AbstractFactor` for generative model factors
  - Use `thrml.block_sampling` for sampling-based inference

### External Dependencies
- `jax`: Core array operations
- `jax.numpy`: Numerical operations
- `equinox`: Module system
- `jaxtyping`: Type annotations

## Usage

```python
import jax.numpy as jnp
from active_inference.core import GenerativeModel, variational_free_energy, Precision
from active_inference.core.free_energy import batch_expected_free_energy

# Create a generative model
model = GenerativeModel(
    n_states=4,
    n_observations=4,
    n_actions=2,
)

# Calculate free energy
belief = jnp.array([0.25, 0.25, 0.25, 0.25])
fe = variational_free_energy(observation=0, state_belief=belief, model=model)

# Calculate EFE for all actions (direct import required)
efe_values = batch_expected_free_energy(belief, model, planning_horizon=1)

# Set precision parameters
precision = Precision(action_precision=2.0)
```

## Design Principles

1. **Modularity**: Each component is independently usable
2. **JAX Compatibility**: All functions are JIT-compatible and vectorizable
3. **No Mocks**: Only real mathematical operations
4. **Type Safety**: Full type annotations with jaxtyping
5. **Numerical Stability**: All operations include numerical safeguards

## Testing

All components have comprehensive unit tests in `tests/test_core.py`:
- Generative model initialization and normalization
- Free energy calculations (mathematical correctness)
- Precision weighting operations
- Edge cases and numerical stability

## Future Enhancements

1. Full THRML integration for sampling-based inference
2. Continuous state spaces
3. Parameter learning via gradient descent
4. Hierarchical model support expansion
