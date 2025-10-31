# Building Custom Models

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [Model Module](module_models.md) | [Core Module](module_core.md)

Guide to creating custom generative models for active inference.

## Overview

Custom generative models allow you to define specific task dynamics, observation structures, and preferences beyond the pre-built models. This guide covers the complete process from specification to validation.

## Generative Model Components

### POMDP Structure

A generative model encodes beliefs about the world as a Partially Observable Markov Decision Process (POMDP):

```python
from active_inference.core import GenerativeModel
import jax.numpy as jnp

model = GenerativeModel(
    n_states=S,           # Hidden states
    n_observations=O,     # Observations
    n_actions=A,          # Actions
    A=A_matrix,           # P(o|s)
    B=B_tensor,           # P(s'|s,a)
    C=C_vector,           # Preferences
    D=D_vector            # Initial belief
)
```

### Matrix Dimensions

- **A matrix**: `[n_observations, n_states]` - Observation likelihood
- **B tensor**: `[n_states, n_states, n_actions]` - State transitions
- **C vector**: `[n_observations]` - Preferences (goals)
- **D vector**: `[n_states]` - Prior belief

---

## Step-by-Step Model Building

### Step 1: Define State Space

**Example**: Simple 2-room navigation

```python
# States
states = {
    0: "room_1",
    1: "room_2",
    2: "hallway"
}
n_states = 3
```

---

### Step 2: Define Observation Space

**Example**: Noisy room sensors

```python
# Observations
observations = {
    0: "sense_room_1",
    1: "sense_room_2",
    2: "sense_hallway",
    3: "sensor_error"
}
n_observations = 4
```

---

### Step 3: Define Action Space

**Example**: Movement actions

```python
# Actions
actions = {
    0: "move_to_room_1",
    1: "move_to_room_2",
    2: "stay"
}
n_actions = 3
```

---

### Step 4: Construct A Matrix (Observation Likelihood)

**Purpose**: P(observation | state)

**Example**: 90% accurate sensors, 10% errors

```python
import jax.numpy as jnp
from active_inference.core import normalize_distribution

# Create A matrix
A = jnp.zeros((n_observations, n_states))

# Room 1: 90% correct, 5% other rooms, 5% error
A = A.at[0, 0].set(0.90)  # Observe room_1 | in room_1
A = A.at[1, 0].set(0.05)  # Observe room_2 | in room_1
A = A.at[2, 0].set(0.00)  # Observe hallway | in room_1
A = A.at[3, 0].set(0.05)  # Observe error | in room_1

# Room 2: similar pattern
A = A.at[0, 1].set(0.05)
A = A.at[1, 1].set(0.90)
A = A.at[2, 1].set(0.00)
A = A.at[3, 1].set(0.05)

# Hallway: different pattern
A = A.at[0, 2].set(0.10)
A = A.at[1, 2].set(0.10)
A = A.at[2, 2].set(0.75)
A = A.at[3, 2].set(0.05)

# Normalize columns (sum to 1)
A = normalize_distribution(A, axis=0)

print(f"A matrix shape: {A.shape}")  # (4, 3)
```

**Guidelines**:
- Each column must sum to 1
- Higher values = more likely observation given state
- Include noise/ambiguity for realism

---

### Step 5: Construct B Tensor (State Transitions)

**Purpose**: P(next_state | current_state, action)

**Example**: Deterministic transitions with small noise

```python
B = jnp.zeros((n_states, n_states, n_actions))

# Action 0: Move to room 1
# From any state → room 1 (95% success)
B = B.at[0, 0, 0].set(0.95)  # room_1 → room_1
B = B.at[0, 1, 0].set(0.95)  # room_2 → room_1
B = B.at[0, 2, 0].set(0.95)  # hallway → room_1

# 5% chance of staying
B = B.at[1, 1, 0].set(0.05)
B = B.at[2, 2, 0].set(0.05)

# Action 1: Move to room 2
B = B.at[1, 0, 1].set(0.95)
B = B.at[1, 1, 1].set(0.95)
B = B.at[1, 2, 1].set(0.95)
B = B.at[0, 0, 1].set(0.05)
B = B.at[2, 2, 1].set(0.05)

# Action 2: Stay
B = B.at[0, 0, 2].set(1.0)  # Stay in room_1
B = B.at[1, 1, 2].set(1.0)  # Stay in room_2
B = B.at[2, 2, 2].set(1.0)  # Stay in hallway

# Normalize rows for each action (sum to 1)
for action in range(n_actions):
    B = B.at[:, :, action].set(
        normalize_distribution(B[:, :, action], axis=0)
    )

print(f"B tensor shape: {B.shape}")  # (3, 3, 3)
```

**Guidelines**:
- Each row (for each action) must sum to 1
- Deterministic transitions: single 1.0 entry per row
- Stochastic transitions: distribute probability

---

### Step 6: Define C Vector (Preferences)

**Purpose**: Which observations are preferred (goals)

**Example**: Goal is to be in room_2

```python
C = jnp.zeros(n_observations)

# High preference for observing room_2
C = C.at[1].set(5.0)  # Strong preference

# Low preference for hallway
C = C.at[2].set(1.0)  # Mild preference

# Negative preference for errors
C = C.at[3].set(-2.0)  # Avoid errors

print(f"C vector shape: {C.shape}")  # (4,)
```

**Guidelines**:
- Positive values = preferred observations
- Negative values = avoided observations
- Zero = neutral
- Scale affects behavior intensity

---

### Step 7: Define D Vector (Initial Prior)

**Purpose**: Initial belief about state

**Example**: Start in room_1 with certainty

```python
D = jnp.zeros(n_states)

# Start in room_1
D = D.at[0].set(1.0)

# Or uncertain start
D_uncertain = jnp.ones(n_states) / n_states

# Normalize
D = normalize_distribution(D)

print(f"D vector shape: {D.shape}")  # (3,)
```

**Guidelines**:
- Must sum to 1
- Sharp prior (near 1.0) = confident start
- Flat prior (uniform) = uncertain start

---

### Step 8: Create Model

**Complete Example**:

```python
from active_inference.core import GenerativeModel

model = GenerativeModel(
    n_states=3,
    n_observations=4,
    n_actions=3,
    A=A,  # Observation likelihood
    B=B,  # State transitions
    C=C,  # Preferences
    D=D   # Initial prior
)

print("Model created successfully!")
```

---

## Validation

### Validate Model Structure

```python
from active_inference.utils import DataValidator

validator = DataValidator()

# Validate model
results = validator.validate_generative_model(model)

if validator.all_passed():
    print("✓ Model is valid")
else:
    validator.print_report()
    validator.generate_html_report("model_validation.html")
```

**Validation Checks**:
- ✅ A matrix columns sum to 1
- ✅ B tensor rows sum to 1 for each action
- ✅ D vector sums to 1
- ✅ C vector contains finite values
- ✅ No NaN or Inf values
- ✅ Correct shapes

---

## Testing Model Behavior

### Test Inference

```python
from active_inference.inference import infer_states

# Test observation
observation = 1  # Sense room_2

# Infer state
posterior, fe = infer_states(
    observation=observation,
    prior_belief=model.D,
    model=model,
    n_iterations=16
)

print(f"Posterior belief: {posterior}")
print(f"Most likely state: {jnp.argmax(posterior)}")
print(f"Free energy: {fe:.3f}")
```

### Test Planning

```python
from active_inference.core import expected_free_energy

# Evaluate all actions
efes = []
for action in range(model.n_actions):
    efe = expected_free_energy(posterior, action, model)
    efes.append(efe)
    print(f"Action {action}: EFE = {efe:.3f}")

best_action = jnp.argmin(jnp.array(efes))
print(f"Best action: {best_action}")
```

---

## Advanced Patterns

### Pattern 1: Sparse Transitions

For large state spaces with local connectivity:

```python
def create_sparse_B(n_states, n_actions, connectivity):
    """Create B tensor with sparse transitions."""
    B = jnp.zeros((n_states, n_states, n_actions))

    for action in range(n_actions):
        for s in range(n_states):
            # Only connect to neighboring states
            neighbors = connectivity[s]
            n_neighbors = len(neighbors)

            for s_next in neighbors:
                B = B.at[s_next, s, action].set(1.0 / n_neighbors)

    return B
```

### Pattern 2: Factored State Space

For compositional states:

```python
def create_factored_model(factor_sizes, factor_names):
    """Create model with factored state representation."""
    n_states = jnp.prod(jnp.array(factor_sizes))

    # States are tuples of factor values
    # e.g., (position, direction, mode)

    # Build A, B, C, D for factored structure
    # ...

    return model
```

### Pattern 3: Hierarchical Observations

Multiple observation modalities:

```python
def create_multi_modal_A(modalities):
    """Create A matrix with multiple observation types."""
    n_obs_total = sum(m['n_obs'] for m in modalities)
    n_states = modalities[0]['n_states']

    A = jnp.zeros((n_obs_total, n_states))

    offset = 0
    for modality in modalities:
        A_mod = modality['likelihood_fn'](n_states, modality['n_obs'])
        A = A.at[offset:offset+modality['n_obs'], :].set(A_mod)
        offset += modality['n_obs']

    return A
```

---

## Model Utilities

### Save and Load Models

```python
import pickle

# Save model
with open('custom_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('custom_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Visualize Model Structure

```python
from active_inference.visualization import plot_generative_model

fig, axes = plot_generative_model(
    model=model,
    save_path="model_structure.png"
)
```

### Model Statistics

```python
def print_model_stats(model):
    print(f"State space size: {model.n_states}")
    print(f"Observation space size: {model.n_observations}")
    print(f"Action space size: {model.n_actions}")
    print(f"Total parameters: {model.A.size + model.B.size + model.C.size + model.D.size}")
    print(f"A matrix entropy: {calculate_entropy(model.A.mean(axis=1)):.3f}")
    print(f"B tensor sparsity: {(model.B == 0).mean():.2%}")
```

---

## Common Patterns by Domain

### Navigation Tasks

```python
from active_inference.models import build_grid_world_model
from active_inference.environments import GridWorldConfig

# Use pre-built builder as template
config = GridWorldConfig(size=5, goal_location=(4, 4))
model = build_grid_world_model(config)

# Customize
model = model.replace(C=custom_preferences)
```

### Decision Tasks

```python
# T-maze style
from active_inference.models import build_tmaze_model

model = build_tmaze_model(reward_side=0)
# Modify for custom task
```

### Continuous Control Approximation

```python
# Discretize continuous space
def discretize_continuous(state_range, n_bins):
    """Create discrete approximation of continuous space."""
    bins = jnp.linspace(state_range[0], state_range[1], n_bins)

    def map_to_discrete(continuous_state):
        return jnp.digitize(continuous_state, bins)

    return bins, map_to_discrete
```

---

## Cross-References

- [Core Module](module_core.md) - GenerativeModel API
- [Model Module](module_models.md) - Pre-built models
- [Utils Module](module_utils.md#validation) - Validation tools
- [Visualization Module](module_visualization.md) - Model visualization
- [Theory](theory.md#generative-models) - Mathematical background

---

## Examples

- [Example 04: MDP Example](../examples/04_mdp_example.py)
- [Example 05: POMDP Example](../examples/05_pomdp_example.py)

---

> **Next**: [Model Module](module_models.md) | [Core Module](module_core.md) | [Custom Environments](custom_environments.md)
