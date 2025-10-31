# Models Module: Functions

## Model Builder Functions

### `build_grid_world_model(config, goal_preference_strength)`
**Purpose**: Build generative model for grid world navigation.

**Algorithm**:
1. Extract environment specifications from config
2. Build observation model A[o, s] with noise
3. Build transition model B[s', s, a] based on grid dynamics
4. Build preference model C[o] favoring goal observations
5. Set uniform prior D[s]

**Arguments**:
- `config`: GridWorldConfig instance
- `goal_preference_strength`: How strongly agent prefers goal (default: 2.0)

**Returns**: GenerativeModel instance

**Model Structure**:
- `n_states`: `size * size` (one per grid cell)
- `n_observations`: `config.n_observations`
- `n_actions`: 4 (up, right, down, left)

**Observations**:
- Each state has preferred observation with noise
- Noise level controlled by `config.observation_noise`

**Transitions**:
- Deterministic based on grid boundaries and obstacles
- Actions that hit obstacles keep agent in place

**Preferences**:
- Strong preference for observations associated with goal location

**THRML Integration**: Can convert to THRML factors:
- Observation factor: `CategoricalEBMFactor` with A matrix weights
- Transition factor: Custom factor with B tensor weights
- Preference factor: Custom factor with C vector weights

### `build_tmaze_model(reward_side, prior_confidence)`
**Purpose**: Build generative model for T-maze epistemic foraging.

**Algorithm**:
1. Define 4 states (start, junction, left, right)
2. Build observation model with cues
3. Build transition model for maze navigation
4. Build preferences favoring reward side
5. Set initial state to start

**Arguments**:
- `reward_side`: Which side has reward (0=left, 1=right)
- `prior_confidence`: Agent's prior confidence about reward side (default: 0.5)

**Returns**: GenerativeModel instance

**Model Structure**:
- `n_states`: 4 (start, junction, left arm, right arm)
- `n_observations`: 8 (various observations including cues)
- `n_actions`: 3 (forward, left, right)

**Observations**:
- Start state: Neutral observation + left/right cues
- Junction: Distinct observation
- Left/Right arms: Distinct observations

**Transitions**:
- From start: Forward → junction
- From junction: Left → left arm, Right → right arm
- Terminal states: Absorbing

**Preferences**:
- Prefer observations from reward side

**Purpose**: Tests epistemic value (information-seeking).

**THRML Integration**: Can convert to THRML factors similar to grid world.

## Usage Patterns

### Basic Model Building
```python
from active_inference.models import build_grid_world_model
from active_inference.environments import GridWorldConfig

config = GridWorldConfig(size=5, goal_location=(4, 4))
model = build_grid_world_model(config, goal_preference_strength=2.0)
```

### T-Maze Model
```python
from active_inference.models import build_tmaze_model

model = build_tmaze_model(reward_side=0, prior_confidence=0.5)
```

### Use with Agent
```python
from active_inference.agents import ActiveInferenceAgent
from active_inference.models import build_grid_world_model

model = build_grid_world_model(config)
agent = ActiveInferenceAgent(model=model)
```

## THRML Integration Opportunities

### Convert to THRML Factors
```python
# Future: Convert model to THRML factors
from thrml.factor import AbstractFactor
from thrml.models.discrete_ebm import CategoricalEBMFactor

def model_to_thrml_factors(model: GenerativeModel) -> list[AbstractFactor]:
    """Convert GenerativeModel to THRML factors."""
    # Observation factor
    obs_factor = CategoricalEBMFactor(
        weights=model.A.T,  # [n_states, n_observations]
        node_groups=[observation_nodes, state_nodes],
    )

    # Transition factor
    trans_factor = TransitionFactor(
        weights=model.B,  # [n_states, n_states, n_actions]
        node_groups=[next_state_nodes, state_nodes, action_nodes],
    )

    return [obs_factor, trans_factor]
```

### Create Matching Nodes
```python
# Future: Create THRML nodes matching model
from thrml import CategoricalNode, Block

def create_model_nodes(model: GenerativeModel) -> tuple[Block, ...]:
    """Create THRML nodes matching model structure."""
    state_nodes = [CategoricalNode() for _ in range(model.n_states)]
    obs_nodes = [CategoricalNode() for _ in range(model.n_observations)]
    action_nodes = [CategoricalNode() for _ in range(model.n_actions)]

    return (
        Block(state_nodes),
        Block(obs_nodes),
        Block(action_nodes),
    )
```

## Integration Status

### ✅ Current
- Automatic model construction
- Matrix-based representations
- Compatible with variational inference

### 🔄 Future
- THRML factor conversion
- THRML node creation
- Energy-based model conversion
- Factor graph construction
