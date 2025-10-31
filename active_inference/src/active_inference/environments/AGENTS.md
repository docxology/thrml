# Environments Module: Classes and Functions

## Grid World (`grid_world.py`)

### `GridWorldConfig`
**Purpose**: Configuration for grid world environment.

**Attributes**:
- `size`: Grid size (square grid)
- `n_observations`: Number of unique observations
- `observation_noise`: Probability of noisy observation
- `goal_location`: Coordinates of goal (None for no goal)
- `obstacle_locations`: List of obstacle coordinates

**Usage**: Configure grid world before creating environment.

### `GridWorld`
**Purpose**: 2D grid world environment for navigation.

**Attributes**:
- `config`: GridWorldConfig instance
- `n_states`: Total number of states (grid cells)
- `n_observations`: Number of possible observations
- `n_actions`: Number of possible actions (4 directions)
- `current_state`: Current agent location [2]

**Actions**:
- 0: Up
- 1: Right
- 2: Down
- 3: Left

**Methods**:

#### `reset(key)`
**Purpose**: Reset environment to initial state.

**Returns**: Initial observation (int)

#### `step(key, action)`
**Purpose**: Execute action and return new observation, reward, done.

**Arguments**:
- `key`: JAX random key
- `action`: Action index (0-3)

**Returns**: Tuple of (observation, reward, done)

**THRML Integration**: Can use THRML for probabilistic transitions.

#### `get_observation(key)`
**Purpose**: Get observation from current state.

**Arguments**:
- `key`: JAX random key for noise

**Returns**: Observation index (int)

**Algorithm**: Deterministic observation with optional noise based on `observation_noise` parameter.

#### `state_to_index(state)`
**Purpose**: Convert 2D state coordinates to flat index.

**Arguments**:
- `state`: 2D coordinates [row, col]

**Returns**: Flat state index (int)

#### `index_to_state(index)`
**Purpose**: Convert flat index to 2D state coordinates.

**Arguments**:
- `index`: Flat state index

**Returns**: 2D coordinates [row, col]

#### `is_goal(state)`
**Purpose**: Check if state is goal location.

**Arguments**:
- `state`: Optional state to check (current if None)

**Returns**: Boolean (True if at goal)

## T-Maze (`tmaze.py`)

### `TMaze`
**Purpose**: T-maze environment for epistemic foraging.

**Attributes**:
- `reward_side`: Which side has reward (0=left, 1=right)
- `n_states`: Number of states (4: start, junction, left, right)
- `n_observations`: Number of observations (8)
- `n_actions`: Number of actions (3: forward, left, right)

**States**:
- 0: Start (base)
- 1: Junction
- 2: Left arm
- 3: Right arm

**Purpose**: Tests information-seeking behavior (epistemic value).

**Methods**:

#### `reset(key)`
**Purpose**: Reset to start state.

**Returns**: Tuple of (observation, cue_was_presented)
- `observation`: Initial observation index
- `cue_was_presented`: Boolean indicating if cue was shown

#### `step(action)`
**Purpose**: Execute action (note: no key parameter needed).

**Arguments**:
- `action`: Action index (0=forward, 1=left, 2=right)

**Returns**: Tuple of (observation, reward, done)

#### `get_reward_side()`
**Purpose**: Get which side has the reward.

**Returns**: Side with reward (0=left, 1=right)

## Usage Patterns

### Basic Grid World
```python
from active_inference.environments import GridWorld, GridWorldConfig

config = GridWorldConfig(size=5, goal_location=(4, 4))
env = GridWorld(config=config)

observation = env.reset(key)
for step in range(100):
    action = agent.select_action(observation)
    observation, reward, done = env.step(key, action)
    if done:
        break
```

### Custom Grid World
```python
config = GridWorldConfig(
    size=10,
    goal_location=(9, 9),
    obstacle_locations=[(3, 3), (5, 5), (7, 7)],
    n_observations=20,
    observation_noise=0.2,
)
env = GridWorld(config=config)
```

### T-Maze
```python
from active_inference.environments import TMaze

env = TMaze(reward_side=0)  # Reward on left
observation = env.reset(key)
# Agent must explore to find reward location
```

## THRML Integration Opportunities

### State Representation
```python
# Current: Integer state indices
state = env.current_state

# Future: THRML nodes
from thrml import CategoricalNode
state_nodes = [CategoricalNode() for _ in range(env.n_states)]
state_block = Block(state_nodes)
```

### Transition Factors
```python
# Future: Use THRML factors for transitions
from thrml.factor import AbstractFactor

class TransitionFactor(AbstractFactor):
    """THRML factor for state transitions."""
    def to_interaction_groups(self):
        # Implement transition interactions
        pass
```

### Observation Factors
```python
# Future: Use THRML factors for observations
class ObservationFactor(AbstractFactor):
    """THRML factor for observation likelihood."""
    def to_interaction_groups(self):
        # Implement observation interactions
        pass
```

## Integration Status

### ✅ Current
- Deterministic environments
- Discrete states/actions/observations
- Compatible with agent interfaces

### 🔄 Future
- THRML node-based state representation
- THRML factor-based transitions
- Probabilistic environments
- THRML sampling for state transitions
