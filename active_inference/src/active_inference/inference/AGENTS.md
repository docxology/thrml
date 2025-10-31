# Inference Module Agents and Functions

## Variational Inference (`state_inference.py`)

### `infer_states(observation, prior_belief, model, n_iterations, convergence_threshold)`
**Purpose**: Infer hidden states from observation using variational inference.

**Algorithm**: Fixed-point iteration minimizing variational free energy.

**Arguments**:
- `observation`: Observed index (int)
- `prior_belief`: Prior over states [n_states]
- `model`: GenerativeModel instance
- `n_iterations`: Maximum iterations (default: 16)
- `convergence_threshold`: Convergence criterion (default: 1e-4)

**Returns**: Tuple of (posterior_belief, final_free_energy)

**THRML Integration**: Can be replaced with THRML sampling for complex models.

### `variational_message_passing(observations, actions, model, n_iterations)`
**Purpose**: Perform forward-backward message passing over observation sequences.

**Algorithm**: Forward filtering with optional backward smoothing.

**Arguments**:
- `observations`: List of observation indices
- `actions`: List of action indices (one fewer than observations)
- `model`: GenerativeModel instance
- `n_iterations`: Inference iterations per step

**Returns**: List of posterior beliefs [n_steps]

**THRML Integration**: Use `thrml.sample_states` for sampling-based sequence inference.

### `update_belief_batch(observations, prior_beliefs, model, n_iterations)`
**Purpose**: Batch inference over multiple observations.

**Arguments**:
- `observations`: Batch of observation indices [batch] (as Float[Array])
- `prior_beliefs`: Batch of prior beliefs [batch, n_states]
- `model`: GenerativeModel instance
- `n_iterations`: Inference iterations (default: 16)

**Returns**: Batch of posterior beliefs [batch, n_states]

**THRML Integration**: Use `jax.vmap` with THRML sampling for GPU-accelerated batch inference.

**Note**: Not exported in `inference/__init__.py` - import directly: `from active_inference.inference.state_inference import update_belief_batch`

## THRML-Based Inference (`thrml_inference.py`)

### `ThrmlInferenceEngine`
**Purpose**: Sampling-based inference using THRML's block Gibbs sampling.

**Attributes**:
- `model`: GenerativeModel instance
- `n_samples`: Number of samples for inference
- `n_warmup`: Warmup steps before sampling
- `steps_per_sample`: Gibbs steps between samples

**THRML Components Used**:
- `thrml.Block`: State node blocks
- `thrml.BlockGibbsSpec`: Block specification
- `thrml.CategoricalNode`: Discrete state nodes
- `thrml.sample_states`: Core sampling
- `thrml.SamplingSchedule`: Sampling configuration
- `thrml.factor.FactorSamplingProgram`: Factor-based program (future)
- `thrml.models.discrete_ebm.CategoricalEBMFactor`: Categorical factors (future)
- `thrml.models.discrete_ebm.CategoricalGibbsConditional`: Categorical sampling (future)

**Methods**:

#### `infer_with_sampling(key, observation, n_state_samples)`
**Purpose**: Infer state distribution using THRML sampling.

**Current Implementation**:
- Creates THRML block structure
- Sets up energy function (placeholder)
- Currently uses direct Bayes rule computation (not full THRML sampling yet)
- Creates blocks and schedules but computes posterior directly

**Target Implementation**:
- Convert generative model to THRML factors
- Use `FactorSamplingProgram` for factor-based sampling
- Use `CategoricalEBMFactor` for observation likelihood
- Use `sample_states` for actual sampling

**Returns**: Approximate posterior distribution [n_states]

#### `sample_trajectory(key, actions, initial_state_belief)`
**Purpose**: Sample state trajectory given actions.

**Algorithm**: Sequential state prediction using transition model.

**Returns**: List of state samples [n_steps]

**THRML Integration**: Can use THRML for trajectory sampling with full uncertainty propagation.

## Usage Patterns

### Basic Variational Inference
```python
from active_inference.inference import infer_states

posterior, fe = infer_states(
    observation=0,
    prior_belief=model.D,
    model=model,
    n_iterations=16,
)
```

### THRML Sampling Inference
```python
from active_inference.inference import ThrmlInferenceEngine

engine = ThrmlInferenceEngine(model=model, n_samples=1000)
posterior = engine.infer_with_sampling(key, observation=0)
```

### Batch Inference
```python
from active_inference.inference import update_belief_batch
import jax

observations = jnp.array([0, 1, 0, 1])
priors = jnp.tile(model.D, (4, 1))
posteriors = update_belief_batch(observations, priors, model)
```

### Sequence Inference
```python
from active_inference.inference import variational_message_passing

observations = [0, 1, 2, 0]
actions = [1, 0, 1]
beliefs = variational_message_passing(observations, actions, model)
```

## THRML Integration Roadmap

### Phase 1: Current (Template)
- ✅ Basic THRML imports
- ✅ Block structure creation
- ⚠️ Direct computation (needs factor integration)

### Phase 2: Factor Integration
- [ ] Convert `GenerativeModel` to `AbstractFactor`
- [ ] Implement observation likelihood factor
- [ ] Implement state transition factor
- [ ] Use `FactorSamplingProgram` for inference

### Phase 3: Full Sampling
- [ ] Replace direct computation with `sample_states`
- [ ] Use `CategoricalEBMFactor` for categorical states
- [ ] Use `CategoricalGibbsConditional` for sampling
- [ ] Add `StateObserver` for monitoring

### Phase 4: Optimization
- [ ] GPU-accelerated batch inference
- [ ] Adaptive sampling schedules
- [ ] Convergence monitoring via observers
