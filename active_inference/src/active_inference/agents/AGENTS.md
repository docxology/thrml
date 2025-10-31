# Agents Module: Classes and Functions

## Core Agent Classes

### `ActiveInferenceAgent`
**Purpose**: Main active inference agent implementing perception-action loop.

**Attributes**:
- `model`: GenerativeModel instance
- `precision`: Precision parameters for inference and action
- `planning_horizon`: Number of steps to plan ahead
- `inference_iterations`: Iterations for state inference

**Methods**:

#### `perceive(observation, prior_belief)`
**Purpose**: Infer hidden states from observation (perception).

**Algorithm**: Uses `infer_states` to minimize variational free energy.

**Returns**: Tuple of (posterior_belief, free_energy)

**THRML Integration**: Can use `ThrmlInferenceEngine.infer_with_sampling` for sampling-based perception.

#### `act(key, state_belief)`
**Purpose**: Select action that minimizes expected free energy (action).

**Algorithm**:
1. Calculate EFE for all actions
2. Apply precision-weighted softmax
3. Sample action from distribution

**Returns**: Selected action index (int)

**THRML Integration**: Can use THRML sampling to evaluate EFE under uncertainty.

#### `step(key, observation, agent_state)`
**Purpose**: Perform one complete perception-action cycle.

**Algorithm**:
1. Perceive: infer states from observation
2. Act: select action minimizing EFE
3. Predict: update belief based on action
4. Track: update history and free energy

**Returns**: Tuple of (action, new_agent_state, free_energy)

#### `reset()`
**Purpose**: Reset agent to initial state.

**Returns**: Fresh `AgentState` with initial belief

#### `get_action_distribution(state_belief)`
**Purpose**: Get action probability distribution without sampling.

**Returns**: Action probability distribution [n_actions]

### `AgentState`
**Purpose**: Internal state of an active inference agent.

**Attributes**:
- `belief`: Current posterior belief over states [n_states]
- `observation_history`: List of observed indices
- `action_history`: List of actions taken
- `free_energy_history`: Array of free energy values [history_length]

**Usage**: Track agent's internal state across time steps.

## Planning Functions (`planning.py`)

### `plan_action(state_belief, model, horizon)`
**Purpose**: Plan action by minimizing expected free energy (greedy).

**Algorithm**:
1. Calculate EFE for all actions
2. Select action with minimum EFE

**Arguments**:
- `state_belief`: Current belief over states
- `model`: GenerativeModel instance
- `horizon`: Planning horizon (currently only 1 supported)

**Returns**: Best action index (int)

**THRML Integration**: Can use THRML sampling to evaluate EFE with uncertainty.

### `plan_with_tree_search(state_belief, model, horizon, branching_factor)`
**Purpose**: Plan sequence of actions using tree search.

**Algorithm**: Depth-first search through policy tree, evaluating EFE for sequences.

**Arguments**:
- `state_belief`: Current belief over states
- `model`: GenerativeModel instance
- `horizon`: Number of steps to plan ahead
- `branching_factor`: Max actions to consider at each step (all if None)

**Returns**: Tuple of (best_action_sequence, total_expected_free_energy)

**THRML Integration**: Can use THRML sampling for evaluating policies under uncertainty.

### `evaluate_policy(policy, initial_belief, model)`
**Purpose**: Evaluate total expected free energy for a policy.

**Algorithm**: Sum EFE across policy sequence.

**Arguments**:
- `policy`: Sequence of actions
- `initial_belief`: Starting belief over states
- `model`: GenerativeModel instance

**Returns**: Total expected free energy (scalar)

**Location**: `planning.py`

**Note**: Now exported in `agents/__init__.py` - can import directly from `active_inference.agents`

## Usage Patterns

### Complete Agent Loop
```python
from active_inference.agents import ActiveInferenceAgent
from active_inference.core import GenerativeModel, Precision

# Setup
model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
precision = Precision(action_precision=2.0)
agent = ActiveInferenceAgent(model=model, precision=precision)

# Initialize
agent_state = agent.reset()

# Run loop
for step in range(10):
    observation = env.observe()
    action, agent_state, fe = agent.step(key, observation, agent_state)
    env.step(action)
```

### Planning Only
```python
from active_inference.agents import plan_action, plan_with_tree_search

# Greedy planning
action = plan_action(belief, model, horizon=1)

# Multi-step planning
sequence, total_efe = plan_with_tree_search(belief, model, horizon=3)
```

### Custom Precision
```python
from active_inference.core import Precision

# High precision = exploitation
exploitation_precision = Precision(action_precision=10.0)

# Low precision = exploration
exploration_precision = Precision(action_precision=0.5)

agent = ActiveInferenceAgent(model=model, precision=exploitation_precision)
```

## THRML Integration Opportunities

### Perception with THRML
```python
# Current: Variational inference
posterior, fe = agent.perceive(observation, prior_belief)

# Future: THRML sampling
from active_inference.inference import ThrmlInferenceEngine
thrml_engine = ThrmlInferenceEngine(model=agent.model)
posterior = thrml_engine.infer_with_sampling(key, observation)
```

### Planning with THRML
- Use THRML sampling to evaluate EFE under uncertainty
- Use THRML observers to monitor planning process
- Use THRML factors to represent belief distributions

### Multi-Agent with THRML
- Use THRML batch operations for parallel agent inference
- Use THRML factors for shared beliefs
- Use THRML observers for agent coordination

□
