# Active Inference Agents

## Overview

The `agents` module implements active inference agents that combine perception (state inference) and action (policy selection) into a unified perception-action loop. Agents use the Free Energy Principle to minimize variational free energy through perception and expected free energy through action.

## Modules

### `base_agent.py`
Implements the core active inference agent:
- **ActiveInferenceAgent**: Main agent class with perception-action cycle
- **AgentState**: Internal state tracking (beliefs, history, free energy)

### `planning.py`
Implements planning algorithms for action selection:
- **plan_action**: Greedy action planning (single step lookahead)
- **plan_with_tree_search**: Tree search planning (multi-step lookahead)
- **evaluate_policy**: Policy evaluation via expected free energy

## Architecture

### Perception-Action Loop
1. **Perceive**: Infer hidden states from observations (minimize VFE)
2. **Act**: Select actions that minimize expected free energy
3. **Predict**: Update beliefs based on selected actions
4. **Update**: Track history and free energy

### Agent Components
- Uses `inference.infer_states` for perception
- Uses `core.expected_free_energy` for action selection
- Uses `core.Precision` for precision-weighted action selection
- Uses `planning` functions for advanced planning

## THRML Integration

### Current Usage
- Agents use variational inference (not yet THRML sampling)
- Action selection uses direct EFE calculation
- Can be extended to use THRML sampling for perception

### Integration Opportunities
1. **Perception**: Use `ThrmlInferenceEngine` for sampling-based inference
2. **Planning**: Use THRML sampling for evaluating policies under uncertainty
3. **Belief Tracking**: Use THRML observers to monitor belief evolution
4. **Batch Agents**: Use THRML's batch capabilities for multi-agent scenarios

## Dependencies

### Internal Dependencies
- `core`: GenerativeModel, Precision, expected_free_energy
- `inference`: infer_states for perception

### THRML Integration Points
- `thrml.sample_states`: For sampling-based perception (future)
- `thrml.observers`: For monitoring agent behavior (future)
- `thrml.factor`: For belief representation as factors (future)

## Usage

### Basic Agent
```python
from active_inference.agents import ActiveInferenceAgent, AgentState
from active_inference.core import GenerativeModel, Precision

model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
precision = Precision(action_precision=2.0)

agent = ActiveInferenceAgent(
    model=model,
    precision=precision,
    planning_horizon=3,
    inference_iterations=16,
)

# Initialize agent
agent_state = agent.reset()

# Perception-action cycle
key = jax.random.key(42)
observation = 0
action, new_state, free_energy = agent.step(key, observation, agent_state)
```

### Planning
```python
from active_inference.agents import plan_action, plan_with_tree_search

# Greedy planning
best_action = plan_action(state_belief, model, horizon=1)

# Tree search planning
action_sequence, total_efe = plan_with_tree_search(
    state_belief,
    model,
    horizon=3,
    branching_factor=2,
)
```

## Design Principles

1. **Unified Perception-Action**: Single agent class combines both
2. **Modular Planning**: Separate planning functions for flexibility
3. **State Tracking**: Full history and free energy tracking
4. **Real Operations**: No mocks, actual inference and action selection
5. **THRML Ready**: Designed for THRML integration

## Testing

Comprehensive tests in `tests/test_agents.py`:
- Perception correctness
- Action selection (greedy and planned)
- State tracking
- Free energy minimization
- Planning algorithm correctness

## Future Enhancements

1. THRML-based perception via `ThrmlInferenceEngine`
2. Hierarchical planning across model levels
3. Model learning and parameter updates
4. Multi-agent coordination
5. Continuous action spaces
