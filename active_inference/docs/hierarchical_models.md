# Hierarchical Models

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Core Module](module_core.md) | [Model Module](module_models.md)

Guide to building and using hierarchical generative models in active inference.

## Overview

Hierarchical models enable multi-scale active inference by organizing generative models into levels, where higher levels provide context for lower levels, and lower levels provide prediction errors to higher levels.

## Hierarchical Structure

### Basic Hierarchy

```
Higher Level: Abstract goals/contexts
      ↓ top-down predictions
      ↑ bottom-up prediction errors
Lower Level: Concrete states/actions
```

### Mathematical Formulation

**Two-Level Model**:
- Lower: P(o|s_low)P(s'_low|s_low,a_low)P(s_low)
- Higher: P(s_low|s_high)P(s'_high|s_high,a_high)P(s_high)

---

## Building Hierarchical Models

### Step 1: Define Levels

```python
from active_inference.core import GenerativeModel

# Lower level: Fine-grained control
lower_model = GenerativeModel(
    n_states=100,         # Detailed states
    n_observations=50,
    n_actions=10          # Primitive actions
)

# Higher level: Abstract planning
higher_model = GenerativeModel(
    n_states=10,          # Abstract states
    n_observations=5,
    n_actions=3           # Abstract actions
)
```

---

### Step 2: Define Inter-Level Connections

```python
import jax.numpy as jnp

def create_hierarchical_connections(n_high, n_low):
    """Create connection matrix between levels.

    Maps high-level states to low-level state distributions.
    """
    connections = jnp.zeros((n_low, n_high))

    # Each high-level state corresponds to a region of low-level states
    states_per_high = n_low // n_high

    for h in range(n_high):
        start = h * states_per_high
        end = start + states_per_high
        connections = connections.at[start:end, h].set(1.0 / states_per_high)

    return connections

connections = create_hierarchical_connections(
    n_high=10,
    n_low=100
)
```

---

### Step 3: Build Hierarchical Model

```python
from active_inference.core import HierarchicalGenerativeModel

hierarchical_model = HierarchicalGenerativeModel(
    levels=[lower_model, higher_model],
    connections=connections,
    connection_strength=1.0
)
```

---

## Hierarchical Inference

### Message Passing

**Bottom-Up Messages** (Prediction Errors):
```python
from active_inference.core import Message, MessageType

def compute_bottom_up_message(low_posterior, low_prior, connections):
    """Compute prediction error from lower to higher level."""

    # Prediction error in lower level
    prediction_error = low_posterior - low_prior

    # Map to higher level
    high_error = connections.T @ prediction_error

    message = Message(
        content=high_error,
        source="lower",
        target="higher",
        message_type=MessageType.BOTTOM_UP
    )

    return message
```

**Top-Down Messages** (Predictions):
```python
def compute_top_down_message(high_posterior, connections):
    """Compute prediction from higher to lower level."""

    # Map high-level belief to low-level prior
    low_prior = connections @ high_posterior

    message = Message(
        content=low_prior,
        source="higher",
        target="lower",
        message_type=MessageType.TOP_DOWN
    )

    return message
```

---

### Complete Hierarchical Inference

```python
def hierarchical_inference(observation, hierarchical_model, n_iterations=16):
    """Perform hierarchical state inference."""

    # Initialize beliefs
    low_belief = hierarchical_model.levels[0].D
    high_belief = hierarchical_model.levels[1].D

    for iteration in range(n_iterations):
        # Bottom-up pass: Infer lower level
        low_posterior, _ = infer_states(
            observation,
            low_belief,
            hierarchical_model.levels[0]
        )

        # Compute bottom-up message
        bottom_up_msg = compute_bottom_up_message(
            low_posterior,
            low_belief,
            hierarchical_model.connections
        )

        # Update higher level with prediction error
        high_belief_updated = high_belief + 0.1 * bottom_up_msg.content
        high_belief_updated = normalize_distribution(high_belief_updated)

        # Top-down pass: Predict for lower level
        top_down_msg = compute_top_down_message(
            high_belief_updated,
            hierarchical_model.connections
        )

        # Update lower level belief with top-down prediction
        low_belief = 0.5 * low_posterior + 0.5 * top_down_msg.content
        low_belief = normalize_distribution(low_belief)

        high_belief = high_belief_updated

    return low_belief, high_belief
```

---

## Hierarchical Planning

### Multi-Level Action Selection

```python
def hierarchical_planning(low_belief, high_belief, hierarchical_model):
    """Plan actions at multiple levels."""

    # High-level planning: Select abstract action/goal
    high_action = plan_action(
        high_belief,
        hierarchical_model.levels[1],
        precision
    )

    # Map high-level action to low-level goal
    low_goal = map_high_to_low_goal(
        high_action,
        hierarchical_model.connections
    )

    # Low-level planning: Achieve goal
    # Temporarily modify low-level preferences
    modified_low_model = hierarchical_model.levels[0].replace(
        C=low_goal
    )

    low_action = plan_action(
        low_belief,
        modified_low_model,
        precision
    )

    return low_action, high_action
```

---

## Practical Examples

### Example 1: Navigation Hierarchy

**High Level**: Room selection
**Low Level**: Path finding within room

```python
def build_navigation_hierarchy():
    """Build two-level navigation model."""

    # Low level: Detailed grid positions within rooms
    low_model = GenerativeModel(
        n_states=100,      # 10x10 grid
        n_observations=100,
        n_actions=4        # up, down, left, right
    )

    # High level: Which room
    high_model = GenerativeModel(
        n_states=4,        # 4 rooms
        n_observations=4,
        n_actions=4        # move to room 0,1,2,3
    )

    # Connections: Map rooms to grid regions
    connections = create_room_connections(
        n_rooms=4,
        grid_size=10
    )

    return HierarchicalGenerativeModel(
        levels=[low_model, high_model],
        connections=connections
    )

def create_room_connections(n_rooms, grid_size):
    """Map rooms to grid positions."""
    connections = jnp.zeros((grid_size * grid_size, n_rooms))

    # Divide grid into rooms
    room_size = grid_size // int(jnp.sqrt(n_rooms))

    for room in range(n_rooms):
        row = room // int(jnp.sqrt(n_rooms))
        col = room % int(jnp.sqrt(n_rooms))

        for i in range(room_size):
            for j in range(room_size):
                pos = (row * room_size + i) * grid_size + (col * room_size + j)
                connections = connections.at[pos, room].set(1.0)

    # Normalize
    connections = connections / connections.sum(axis=0, keepdims=True)

    return connections
```

---

### Example 2: Motor Control Hierarchy

**High Level**: Goal selection (grasp, reach, release)
**Low Level**: Joint angles

```python
def build_motor_hierarchy():
    """Build motor control hierarchy."""

    # Low level: Joint configurations
    low_model = GenerativeModel(
        n_states=1000,     # Discretized joint space
        n_observations=100, # Proprioception
        n_actions=20       # Motor commands
    )

    # High level: Motor primitives
    high_model = GenerativeModel(
        n_states=10,       # Grasp types, reach targets
        n_observations=5,  # Task feedback
        n_actions=5        # Primitive selection
    )

    # Connections: Map primitives to joint configurations
    connections = create_motor_connections(
        n_primitives=10,
        n_configurations=1000
    )

    return HierarchicalGenerativeModel(
        levels=[low_model, high_model],
        connections=connections
    )
```

---

## Temporal Abstraction

### Different Timescales

Higher levels operate at slower timescales:

```python
class TemporallyAbstractHierarchy:
    """Hierarchy with different update rates."""

    def __init__(self, hierarchical_model):
        self.model = hierarchical_model
        self.high_update_frequency = 10  # Update every 10 low-level steps
        self.low_step_count = 0

    def step(self, key, observation, low_belief, high_belief):
        """Hierarchical step with temporal abstraction."""

        # Always update lower level
        low_action, low_belief = self.low_level_step(
            key,
            observation,
            low_belief,
            high_belief
        )

        # Update higher level less frequently
        if self.low_step_count % self.high_update_frequency == 0:
            high_action, high_belief = self.high_level_step(
                key,
                low_belief,
                high_belief
            )

        self.low_step_count += 1

        return low_action, low_belief, high_belief
```

---

## Advanced Patterns

### Pattern 1: Deep Hierarchies (3+ Levels)

```python
def build_deep_hierarchy(n_levels=3):
    """Build multi-level hierarchy."""

    levels = []
    state_sizes = [1000, 100, 10]  # Decreasing abstraction

    for i in range(n_levels):
        model = GenerativeModel(
            n_states=state_sizes[i],
            n_observations=state_sizes[i] // 2,
            n_actions=max(4, state_sizes[i] // 50)
        )
        levels.append(model)

    # Create connections between adjacent levels
    connections_list = []
    for i in range(n_levels - 1):
        conn = create_hierarchical_connections(
            n_high=state_sizes[i+1],
            n_low=state_sizes[i]
        )
        connections_list.append(conn)

    return MultiLevelHierarchy(levels, connections_list)
```

---

### Pattern 2: Parallel Hierarchies

Different hierarchies for different modalities:

```python
class MultiModalHierarchy:
    """Separate hierarchies for different modalities."""

    def __init__(self):
        self.vision_hierarchy = build_vision_hierarchy()
        self.motor_hierarchy = build_motor_hierarchy()
        self.cross_modal_connections = create_cross_modal_links()

    def integrated_inference(self, visual_obs, proprioceptive_obs):
        """Perform inference across modalities."""

        # Infer in each hierarchy
        vision_beliefs = hierarchical_inference(
            visual_obs,
            self.vision_hierarchy
        )

        motor_beliefs = hierarchical_inference(
            proprioceptive_obs,
            self.motor_hierarchy
        )

        # Cross-modal integration
        integrated = self.integrate_modalities(
            vision_beliefs,
            motor_beliefs
        )

        return integrated
```

---

## Validation

### Check Hierarchical Structure

```python
def validate_hierarchical_model(hierarchical_model):
    """Validate hierarchical model structure."""

    print("Validating hierarchical model...")

    # Check level consistency
    n_levels = len(hierarchical_model.levels)
    print(f"Number of levels: {n_levels}")

    # Check connections
    for i in range(n_levels - 1):
        conn = hierarchical_model.connections[i]

        # Connections should map low to high
        expected_shape = (
            hierarchical_model.levels[i].n_states,
            hierarchical_model.levels[i+1].n_states
        )

        assert conn.shape == expected_shape, \
            f"Connection shape mismatch at level {i}"

        # Columns should sum to 1 (valid distributions)
        col_sums = conn.sum(axis=0)
        assert jnp.allclose(col_sums, 1.0), \
            f"Connections at level {i} not normalized"

    print("✓ Hierarchical model valid!")
```

---

## Cross-References

- [Core Module](module_core.md) - HierarchicalGenerativeModel
- [Model Module](module_models.md#hierarchical-models) - Model builders
- [Theory](theory.md) - Hierarchical inference theory
- [Agent Module](module_agents.md) - Hierarchical agents

---

> **Next**: [Model Module](module_models.md) | [Core Module](module_core.md) | [Custom Models](custom_models.md)
