# Inference Engines

## Overview

The `inference` module provides variational inference engines for active inference, including both traditional variational message passing and THRML-based sampling inference.

## Modules

### `state_inference.py`
Implements variational inference using iterative belief updating:
- **infer_states**: Fixed-point iteration for posterior inference (exported)
- **variational_message_passing**: Forward-backward message passing over sequences (exported)
- **update_belief_batch**: Batch inference over multiple observations (not exported, import directly)

### `thrml_inference.py`
Implements THRML-based sampling inference (template for full integration):
- **ThrmlInferenceEngine**: Sampling-based inference using THRML's block Gibbs sampling
- **infer_with_sampling**: State inference via sampling
- **sample_trajectory**: Sample state trajectories given actions

## THRML Integration

### Current Usage
- Uses `thrml.Block`, `thrml.BlockGibbsSpec`, `thrml.CategoricalNode`
- Uses `thrml.sample_states`, `thrml.SamplingSchedule`
- Uses `thrml.factor.FactorSamplingProgram` for factor-based sampling
- Uses `thrml.models.discrete_ebm.CategoricalEBMFactor` and `CategoricalGibbsConditional`

### Integration Status
- ✅ Basic THRML types imported and used
- ✅ Block structure created for states
- ⚠️ Currently uses direct Bayes rule computation (needs full factor integration)
- 🔄 Template ready for full THRML factor-based inference

### Target Integration
1. Convert generative models to THRML factors
2. Use `thrml.factor.AbstractFactor` for observation likelihood factors
3. Use `thrml.block_sampling.BlockSamplingProgram` for full sampling pipeline
4. Use `thrml.observers` for monitoring inference quality

## Dependencies

### THRML Components
- `thrml.Block`: Block management for state nodes
- `thrml.BlockGibbsSpec`: Specification of free/clamped blocks
- `thrml.CategoricalNode`: Discrete state nodes
- `thrml.sample_states`: Core sampling function
- `thrml.SamplingSchedule`: Sampling configuration
- `thrml.factor.FactorSamplingProgram`: Factor-based sampling
- `thrml.models.discrete_ebm`: Categorical EBM factors and conditionals

### External Dependencies
- `jax`: Core operations
- `jax.numpy`: Numerical operations
- `equinox`: Module system

## Usage

### Variational Inference
```python
from active_inference.inference import infer_states
from active_inference.core import GenerativeModel

model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
prior = jnp.array([0.25, 0.25, 0.25, 0.25])

posterior, free_energy = infer_states(
    observation=0,
    prior_belief=prior,
    model=model,
    n_iterations=16,
)
```

### THRML Sampling Inference
```python
from active_inference.inference import ThrmlInferenceEngine

engine = ThrmlInferenceEngine(
    model=model,
    n_samples=1000,
    n_warmup=100,
    steps_per_sample=5,
)

key = jax.random.key(42)
posterior = engine.infer_with_sampling(key, observation=0)
```

### Sequence Inference
```python
from active_inference.inference import variational_message_passing

observations = [0, 1, 2, 0]
actions = [1, 0, 1]
beliefs = variational_message_passing(observations, actions, model)
```

### Batch Inference
```python
import jax.numpy as jnp
from active_inference.inference.state_inference import update_belief_batch

observations = jnp.array([0, 1, 0, 1])
priors = jnp.tile(model.D, (4, 1))
posteriors = update_belief_batch(observations, priors, model)
```

## Design Principles

1. **Real Inference**: Actual variational inference, not approximations
2. **THRML Integration**: Maximum use of THRML components
3. **Modularity**: Each inference method is independently usable
4. **Batch Support**: All functions support `jax.vmap` for batching
5. **Convergence**: Iterative methods include convergence checking

## Performance Considerations

- Variational inference: Fast, deterministic, suitable for real-time
- THRML sampling: Slower but more flexible, GPU-accelerated
- Batch operations: Use `jax.vmap` for parallel inference

## Testing

Comprehensive tests in `tests/test_inference.py`:
- Single observation inference
- Sequence inference
- Batch inference
- Convergence behavior
- THRML integration (template tests)

## Future Enhancements

1. Full THRML factor integration for `ThrmlInferenceEngine`
2. Hierarchical inference across model levels
3. Adaptive convergence criteria
4. GPU-optimized batch inference
5. Continuous state space inference
