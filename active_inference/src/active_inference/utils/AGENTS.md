# Utils Module: Functions

## Metrics (`metrics.py`)

### `calculate_kl_divergence(p, q)`
**Purpose**: Calculate KL divergence KL[P || Q].

**Formula**: KL[P || Q] = Σ P(x) log(P(x) / Q(x))

**Arguments**:
- `p`: First distribution [n]
- `q`: Second distribution [n]

**Returns**: KL divergence value (scalar)

**Usage**: Measure difference between beliefs, posteriors, etc.

**THRML Integration**: Can use THRML observers to accumulate KL during sampling.

### `calculate_prediction_accuracy(predicted_observations, actual_observations)`
**Purpose**: Calculate prediction accuracy.

**Algorithm**: Compare predicted observation (argmax) with actual observation.

**Arguments**:
- `predicted_observations`: List of predicted observation distributions
- `actual_observations`: List of actual observation indices

**Returns**: Accuracy (0 to 1)

**Usage**: Evaluate how well agent predicts observations.

### `calculate_policy_entropy(action_probs)`
**Purpose**: Calculate entropy of action distribution.

**Formula**: H[P(a)] = -Σ P(a) log P(a)

**Arguments**:
- `action_probs`: Action probability distribution [n_actions]

**Returns**: Entropy value (scalar)

**Usage**:
- High entropy: Exploration
- Low entropy: Exploitation

**THRML Integration**: Can track entropy via THRML observers.

**Note**: Defined in `metrics.py` but not exported in `__init__.py`. Import directly: `from active_inference.utils.metrics import calculate_policy_entropy`

### `calculate_expected_utility(outcomes, utilities)`
**Purpose**: Calculate expected utility.

**Formula**: E[U] = Σ P(outcome) × U(outcome)

**Arguments**:
- `outcomes`: Probability of each outcome [n_outcomes]
- `utilities`: Utility of each outcome [n_outcomes]

**Returns**: Expected utility (scalar)

**Usage**: Evaluate policy value.

**Note**: Defined in `metrics.py` but not exported in `__init__.py`. Import directly: `from active_inference.utils.metrics import calculate_expected_utility`

## Visualization (`visualization.py`)

### `plot_belief_trajectory(beliefs, true_states, figsize)`
**Purpose**: Plot belief trajectory over time.

**Output**:
- Heatmap of beliefs over time
- Entropy evolution over time

**Arguments**:
- `beliefs`: List of belief distributions over time
- `true_states`: Optional list of true state indices
- `figsize`: Figure size tuple

**Returns**: Matplotlib figure and axes

**THRML Integration**: Can visualize THRML sampling trajectories.

### `plot_free_energy(free_energies, figsize)`
**Purpose**: Plot free energy over time.

**Output**:
- Free energy vs time
- Moving average (if enough points)

**Arguments**:
- `free_energies`: Array of free energy values [n_steps]
- `figsize`: Figure size tuple

**Returns**: Matplotlib figure and axes

**Usage**: Monitor convergence and inference quality.

**THRML Integration**: Can track free energy via THRML observers.

### `plot_action_distribution(action_probs, action_names, figsize)`
**Purpose**: Plot action probability distribution.

**Output**: Bar chart of action probabilities

**Arguments**:
- `action_probs`: Probability distribution over actions [n_actions]
- `action_names`: Optional names for actions
- `figsize`: Figure size tuple

**Returns**: Matplotlib figure and axes

**Usage**: Visualize action selection strategy.

## Usage Patterns

### Evaluate Agent Performance
```python
from active_inference.utils import (
    calculate_kl_divergence,
    calculate_prediction_accuracy,
    calculate_policy_entropy,
)

# Track KL divergence
kl = calculate_kl_divergence(posterior, prior)

# Evaluate predictions
accuracy = calculate_prediction_accuracy(predicted_obs, actual_obs)

# Measure exploration
entropy = calculate_policy_entropy(action_probs)
```

### Visualize Agent Behavior
```python
from active_inference.utils import (
    plot_belief_trajectory,
    plot_free_energy,
    plot_action_distribution,
)

# Belief evolution
fig, axes = plot_belief_trajectory(beliefs, true_states=true_states)

# Free energy
fig, ax = plot_free_energy(free_energies)

# Action selection
fig, ax = plot_action_distribution(action_probs, action_names=["Up", "Right", "Down", "Left"])
```

## THRML Integration Opportunities

### Metrics via Observers
```python
# Future: Use THRML observers for metrics
from thrml.observers import AbstractObserver

class MetricsObserver(AbstractObserver):
    """Collect metrics during THRML sampling."""
    def __call__(self, program, state_free, state_clamped, carry, iteration):
        # Calculate KL divergence
        kl = calculate_kl_divergence(current_belief, prior_belief)

        # Track in carry
        carry['kl_history'] = carry['kl_history'].at[iteration].set(kl)

        return carry, {}
```

### Visualization of THRML Sampling
```python
# Future: Visualize THRML sampling trajectories
def plot_thrml_sampling(samples, observers):
    """Plot THRML sampling results."""
    # Extract beliefs from samples
    beliefs = extract_beliefs(samples)

    # Plot trajectory
    plot_belief_trajectory(beliefs)

    # Plot metrics from observers
    if 'kl_history' in observers:
        plot_metric(observers['kl_history'])
```

## Integration Status

### ✅ Current
- Real metric calculations
- Matplotlib-based visualization
- JAX-compatible operations

### 🔄 Future
- THRML observer integration
- THRML sampling visualization
- Factor graph visualization
- Interactive plots
