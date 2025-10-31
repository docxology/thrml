# Test Environments

## Overview

The `environments` module provides test environments for evaluating active inference agents. These environments implement discrete MDPs that can be used to test perception, action selection, and planning algorithms.

## Modules

### `grid_world.py`
Implements a 2D grid world navigation task:
- **GridWorld**: Main environment class
- **GridWorldConfig**: Configuration for grid world setup

### `tmaze.py`
Implements a T-maze epistemic foraging task:
- **TMaze**: Environment for testing information-seeking behavior

## THRML Integration

### Current Usage
- Environments provide deterministic transitions
- Observations are discrete and deterministic
- Can be used with THRML-based inference engines

### Integration Opportunities
1. **State Representation**: Use `thrml.CategoricalNode` for state nodes
2. **Observation Modeling**: Use THRML factors for observation likelihood
3. **Transition Modeling**: Use THRML factors for state transitions
4. **Sampling**: Use THRML sampling for probabilistic environments

## Dependencies

### Internal Dependencies
- `core`: GenerativeModel for environment modeling

### THRML Integration Points
- `thrml.CategoricalNode`: For state representations (future)
- `thrml.factor`: For transition and observation factors (future)

## Usage

### Grid World
```python
from active_inference.environments import GridWorld, GridWorldConfig

config = GridWorldConfig(
    size=5,
    goal_location=(4, 4),
    obstacle_locations=[(2, 2)],
    n_observations=10,
    observation_noise=0.1,
)

env = GridWorld(config=config)

# Reset environment
key = jax.random.key(42)
observation = env.reset(key)

# Step environment
observation, reward, done = env.step(key, action=1)
```

### T-Maze
```python
from active_inference.environments import TMaze

env = TMaze(reward_side=0)  # Reward on left

# Use with agent
observation, cue_presented = env.reset(key)
# ... agent loop ...
observation, reward, done = env.step(action)  # Note: no key needed
```

## Design Principles

1. **Deterministic**: Transitions and observations are deterministic (with optional noise)
2. **Discrete**: States, actions, and observations are discrete
3. **Configurable**: Easy to configure different scenarios
4. **THRML Compatible**: Designed for THRML integration

## Testing

Comprehensive tests in `tests/test_environments.py`:
- Environment initialization
- State transitions
- Observation generation
- Reward calculation
- Edge cases (obstacles, boundaries)

## Future Enhancements

1. THRML factor-based state representation
2. Probabilistic transitions
3. Continuous action spaces
4. Multi-agent environments
5. More complex environments (mazes, navigation tasks)
