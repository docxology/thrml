# Core Module Agents and Functions

## Generative Models

### `GenerativeModel`
**Purpose**: Represents a POMDP-style generative model for active inference.

**Attributes**:
- `A`: Observation likelihood matrix [n_observations, n_states]
- `B`: State transition tensor [n_states, n_states, n_actions]
- `C`: Preferred observations (log preferences) [n_observations]
- `D`: Initial state prior [n_states]

**Methods**:
- `get_observation_likelihood(observation)`: Get P(o|s) for observation
- `get_state_transition(action)`: Get P(s'|s,a) for action
- `predict_observation(state_belief)`: Predict observation distribution
- `predict_next_state(state_belief, action)`: Predict next state distribution

**THRML Integration Potential**:
- Use `thrml.CategoricalNode` for state representations
- Convert to `thrml.factor.AbstractFactor` for energy-based formulation

### `HierarchicalGenerativeModel`
**Purpose**: Multi-level generative models for hierarchical active inference.

**Attributes**:
- `levels`: List of GenerativeModel instances
- `n_levels`: Number of hierarchical levels

**Methods**:
- `get_level(level)`: Get generative model at specific level

## Free Energy Functions

### `variational_free_energy(observation, state_belief, model)`
**Purpose**: Calculate variational free energy for inference quality assessment.

**Formula**: F = -E_Q[log P(o|s)] + KL[Q(s) || P(s)]

**Returns**: Scalar free energy value

**THRML Integration**: Can be reformulated as energy minimization using THRML factors.

### `expected_free_energy(state_belief, action, model, planning_horizon)`
**Purpose**: Calculate expected free energy for action selection.

**Formula**: G = KL[Q(o|π) || P(o)] + E_Q[H[P(o|s)]]

**Returns**: Scalar expected free energy

**THRML Integration**: Can use THRML sampling to estimate expectations.

### `batch_expected_free_energy(state_belief, model, planning_horizon)`
**Purpose**: Calculate EFE for all actions simultaneously.

**Arguments**:
- `state_belief`: Current posterior over states
- `model`: GenerativeModel instance
- `planning_horizon`: Number of steps to look ahead (default: 1)

**Returns**: Array of EFE values [n_actions]

**Note**: Not exported in `core/__init__.py` - import directly: `from active_inference.core.free_energy import batch_expected_free_energy`

## Precision Weighting

### `MessageType`
**Purpose**: Enum for types of messages in hierarchical inference.

**Values**:
- `BOTTOM_UP`: Prediction errors from lower levels
- `TOP_DOWN`: Predictions from higher levels
- `LATERAL`: Messages within a level

### `Message`
**Purpose**: A message passed between inference nodes.

**Attributes**:
- `content`: The message payload (probability distribution or prediction error)
- `message_type`: The type of message (MessageType)
- `precision`: The reliability/confidence in this message

### `Precision`
**Purpose**: Precision (inverse variance) parameters for weighted inference.

**Attributes**:
- `sensory_precision`: Reliability of observations
- `state_precision`: Reliability of state transitions
- `action_precision`: Inverse temperature for action selection

### `PrecisionWeighting`
**Purpose**: Utility class for precision-weighted operations.

**Static Methods**:
- `weight_prediction_error(prediction_error, precision)`: Apply precision weighting
- `softmax_with_precision(values, precision)`: Precision-weighted softmax
- `update_sensory_precision(prediction_errors, learning_rate)`: Online precision estimation

## Helper Functions

### `normalize_distribution(x)`
**Purpose**: Normalize array to sum to 1.

**Location**: `generative_model.py` (module-level function, not method)

**Arguments**:
- `x`: Array to normalize

**Returns**: Normalized array

**Note**: Not exported in `core/__init__.py` - import directly: `from active_inference.core.generative_model import normalize_distribution`

### `softmax_stable(x)`
**Purpose**: Numerically stable softmax.

**Location**: `generative_model.py` (module-level function, not method)

**Arguments**:
- `x`: Log probabilities

**Returns**: Normalized probabilities

**Note**: Not exported in `core/__init__.py` - import directly: `from active_inference.core.generative_model import softmax_stable`

## Usage Patterns

### Basic Inference
```python
model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
belief = jnp.array([0.25, 0.25, 0.25, 0.25])
fe = variational_free_energy(observation=0, state_belief=belief, model=model)
```

### Action Selection
```python
efe_values = batch_expected_free_energy(belief, model, horizon=1)
best_action = jnp.argmin(efe_values)
```

### Precision Weighting
```python
precision = Precision(action_precision=2.0)
action_probs = PrecisionWeighting.softmax_with_precision(-efe_values, precision.action_precision)
```
