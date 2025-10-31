# Model Builders

## Overview

The `models` module provides functions to automatically build generative models from environment specifications. These builders create properly structured `GenerativeModel` instances that match the dynamics of specific environments.

## Modules

### `discrete_mdp.py`
Implements model builders for discrete MDPs:
- **build_grid_world_model**: Builds generative model for grid world
- **build_tmaze_model**: Builds generative model for T-maze

## THRML Integration

### Current Usage
- Builders create standard `GenerativeModel` instances
- Models use matrix-based representations (A, B, C, D)
- Compatible with both variational and THRML inference

### Integration Opportunities
1. **Factor Construction**: Convert built models to THRML factors
2. **Node Creation**: Create THRML nodes matching model structure
3. **Factor Graphs**: Build THRML factor graphs from models
4. **Energy Formulation**: Convert to energy-based model representation

## Dependencies

### Internal Dependencies
- `core`: GenerativeModel class
- `environments`: GridWorldConfig for configuration

### THRML Integration Points
- `thrml.factor.AbstractFactor`: For converting models to factors (future)
- `thrml.CategoricalNode`: For creating matching nodes (future)
- `thrml.models.discrete_ebm`: For EBM conversion (future)

## Usage

### Grid World Model
```python
from active_inference.models import build_grid_world_model
from active_inference.environments import GridWorldConfig

config = GridWorldConfig(
    size=5,
    goal_location=(4, 4),
    obstacle_locations=[(2, 2)],
)

model = build_grid_world_model(
    config=config,
    goal_preference_strength=2.0,
)
```

### T-Maze Model
```python
from active_inference.models import build_tmaze_model

model = build_tmaze_model(
    reward_side=0,  # Reward on left
    prior_confidence=0.5,
)
```

## Design Principles

1. **Automatic Construction**: Build models from environment specs
2. **Consistent Interface**: Same structure across different environments
3. **Configurable**: Allow customization of model parameters
4. **THRML Compatible**: Models can be converted to THRML factors

## Testing

Comprehensive tests in `tests/test_integration.py`:
- Model construction correctness
- Model-environment consistency
- Parameter validation
- Edge cases

## Future Enhancements

1. THRML factor conversion
2. More environment builders
3. Hierarchical model builders
4. Learning model builders (from data)
5. Continuous state space builders
