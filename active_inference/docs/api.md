# API Reference

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [Theory](theory.md) | [Module Index](module_index.md) | [Workflows](workflows_patterns.md)

Complete API reference for all active_inference components.

## Quick Navigation

- [Core Components](#core-components) - GenerativeModel, Free Energy, Precision
- [Inference](#inference) - State inference engines
- [Agents](#agents) - Active inference agents
- [Environments](#environments) - Test environments
- [Models](#models) - Pre-built models
- [Utilities](#utilities) - Analysis and validation
- [Visualization](#visualization) - Plotting tools

**Detailed Documentation**: See [Module Index](module_index.md) for comprehensive module-by-module documentation.

## Core Components

### GenerativeModel

```python
from active_inference.core import GenerativeModel

model = GenerativeModel(
    n_states=10,
    n_observations=5,
    n_actions=4,
)
```

**Attributes:**
- `A`: Observation likelihood matrix [n_obs, n_states]
- `B`: State transition tensor [n_states, n_states, n_actions]
- `C`: Preferred observations [n_obs]
- `D`: Initial state prior [n_states]

**Methods:**
- `get_observation_likelihood(observation)`: Get P(o|s)
- `get_state_transition(action)`: Get P(s'|s,a)
- `predict_observation(state_belief)`: Predict observations
- `predict_next_state(state_belief, action)`: Predict next states

### Free Energy Functions

```python
from active_inference.core import variational_free_energy, expected_free_energy

# Perception (inference)
fe = variational_free_energy(observation, state_belief, model)

# Action selection (planning)
efe = expected_free_energy(state_belief, action, model)
```

### Precision

```python
from active_inference.core import Precision

precision = Precision(
    sensory_precision=1.0,
    state_precision=1.0,
    action_precision=2.0,
)
```

## Inference

### State Inference

```python
from active_inference.inference import infer_states

posterior, free_energy = infer_states(
    observation=obs,
    prior_belief=prior,
    model=model,
    n_iterations=16,
)
```

### THRML Inference Engine

```python
from active_inference.inference import ThrmlInferenceEngine

engine = ThrmlInferenceEngine(
    model=model,
    n_samples=1000,
    n_warmup=100,
)

posterior = engine.infer_with_sampling(key, observation)
```

## Agents

### Active Inference Agent

```python
from active_inference.agents import ActiveInferenceAgent

agent = ActiveInferenceAgent(
    model=model,
    precision=precision,
    planning_horizon=3,
)

# Perception-action cycle
action, agent_state, free_energy = agent.step(key, observation, agent_state)
```

## Environments

### Grid World

```python
from active_inference.environments import GridWorld, GridWorldConfig

config = GridWorldConfig(
    size=5,
    n_observations=10,
    goal_location=(4, 4),
)

env = GridWorld(config=config)
obs = env.reset(key)
obs, reward, done = env.step(key, action)
```

### T-Maze

```python
from active_inference.environments import TMaze

env = TMaze(reward_side=0)
obs, cue_presented = env.reset(key)
obs, reward, done = env.step(action)
```

## Models

### Building Models

```python
from active_inference.models import build_grid_world_model, build_tmaze_model

# Grid world model
grid_model = build_grid_world_model(
    config=grid_config,
    goal_preference_strength=2.0,
)

# T-maze model
tmaze_model = build_tmaze_model(
    reward_side=0,
    prior_confidence=0.5,
)
```

## Utilities

### Metrics

```python
from active_inference.utils import calculate_kl_divergence, calculate_prediction_accuracy

kl = calculate_kl_divergence(p, q)
accuracy = calculate_prediction_accuracy(predictions, actuals)
```

### Statistical Analysis

```python
from active_inference.utils import linear_regression, pearson_correlation

# Linear regression
reg_results = linear_regression(x, y)
print(f"R² = {reg_results.r_squared:.3f}, p = {reg_results.p_value:.4f}")

# Correlation
corr_results = pearson_correlation(x, y)
print(f"r = {corr_results.correlation:.3f}")
```

### Data Validation

```python
from active_inference.utils import DataValidator

validator = DataValidator()
validator.validate_generative_model(model)
validator.print_report()
```

### Resource Tracking

```python
from active_inference.utils import ResourceTracker

tracker = ResourceTracker()
tracker.start()
# ... code to track ...
tracker.stop()
print(tracker.generate_report())
```

### Visualization

```python
from active_inference.utils import plot_belief_trajectory, plot_free_energy
from active_inference import visualization as viz

# Basic plots
fig, axes = plot_belief_trajectory(beliefs, true_states)
fig, ax = plot_free_energy(free_energies)

# Statistical plots
viz.plot_scatter_with_regression(x, y, save_path="regression.png")
viz.plot_correlation_matrix(data, save_path="correlation.png")
```

---

## Cross-References

### Detailed Module Documentation
- [Core Module](module_core.md) - Complete core API
- [Inference Module](module_inference.md) - All inference methods
- [Agent Module](module_agents.md) - Agent implementation details
- [Model Module](module_models.md) - Model builders
- [Environment Module](module_environments.md) - Environment API
- [Utils Module](module_utils.md) - Utility functions
- [Visualization Module](module_visualization.md) - All plotting functions

### Related Guides
- [Getting Started](getting_started.md) - Setup and first steps
- [Workflows & Patterns](workflows_patterns.md) - Common patterns
- [THRML Integration](thrml_integration.md) - THRML methods
- [Analysis & Validation](analysis_validation.md) - Statistical tools

---

> **Next**: [Module Index](module_index.md) | [Core Module](module_core.md) | [Theory](theory.md)
