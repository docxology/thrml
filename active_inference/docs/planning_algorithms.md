# Planning Algorithms

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Agent Module](module_agents.md) | [Theory](theory.md)

Detailed guide to planning and policy selection algorithms in active inference.

## Overview

Planning in active inference involves selecting actions that minimize expected free energy. This guide covers different planning strategies from simple greedy selection to sophisticated tree search.

## Planning Strategies

### 1. Greedy Planning (Horizon = 1)

**Purpose**: Select action with minimum immediate expected free energy

**Algorithm**:
```
For each action a:
    1. Predict next state: P(s'|s,a)
    2. Predict observation: P(o'|s')
    3. Calculate G(a) = EFE(s', a)
Select: a* = argmin_a G(a)
```

**Implementation**:
```python
from active_inference.agents import plan_action
from active_inference.core import Precision

action = plan_action(
    state_belief=current_belief,
    model=model,
    precision=Precision(action_precision=2.0)
)
```

**Complexity**: O(n_actions × n_states²)

**When to Use**:
- Real-time control (< 10ms latency)
- Simple tasks
- Large state spaces
- Quick iterations needed

---

### 2. Fixed Horizon Planning

**Purpose**: Plan N steps ahead with fixed action sequence

**Algorithm**:
```
For each action sequence π = [a1, a2, ..., aN]:
    1. Simulate trajectory: s1, s2, ..., sN
    2. Calculate cumulative EFE: G(π) = Σ G(st, at)
    3. Weight by probability
Select: π* = argmin_π G(π)
Return: first action of π*
```

**Implementation**:
```python
from active_inference.agents import plan_with_tree_search
import jax

action = plan_with_tree_search(
    key=key,
    state_belief=current_belief,
    model=model,
    horizon=3,           # Look 3 steps ahead
    n_samples=100        # Sample 100 trajectories
)
```

**Complexity**: O(n_actions^horizon × n_samples)

**When to Use**:
- Strategic planning
- Sequential decisions
- Medium state spaces
- Offline/batch planning

---

### 3. Monte Carlo Tree Search (MCTS)

**Purpose**: Efficiently explore large action spaces

**Algorithm**:
```
Repeat until budget exhausted:
    1. Selection: Traverse tree using UCB
    2. Expansion: Add new child node
    3. Simulation: Rollout to terminal state
    4. Backpropagation: Update values
Return: Best action from root
```

**Implementation**:
```python
class MCTSNode:
    def __init__(self, belief, parent=None, action=None):
        self.belief = belief
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, exploration=1.41):
        """UCB1 selection criterion."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration_bonus = exploration * jnp.sqrt(
            jnp.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration_bonus

def mcts_plan(root_belief, model, n_iterations=1000):
    """Monte Carlo Tree Search for action selection."""
    root = MCTSNode(root_belief)

    for _ in range(n_iterations):
        node = root

        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score())

        # Expansion
        if node.visits > 0:
            for action in range(model.n_actions):
                next_belief = model.predict_next_state(node.belief, action)
                child = MCTSNode(next_belief, parent=node, action=action)
                node.children.append(child)
            node = node.children[0]

        # Simulation (rollout)
        value = simulate_rollout(node.belief, model, depth=10)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    # Select best action
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child.action
```

**Complexity**: O(n_iterations × depth)

**When to Use**:
- Large branching factor
- Long horizons
- Complex decision trees
- Computational budget available

---

### 4. Beam Search

**Purpose**: Prune poor action sequences early

**Algorithm**:
```
1. Start with root belief
2. For each level in tree:
    a. Expand top k beliefs
    b. Score all children
    c. Keep only top k children (beam width)
3. Return best final sequence
```

**Implementation**:
```python
def beam_search(belief, model, horizon=5, beam_width=3):
    """Beam search planning."""

    # Initialize beam with root
    beam = [(belief, [], 0.0)]  # (belief, actions, cumulative_efe)

    for depth in range(horizon):
        candidates = []

        for current_belief, actions, cum_efe in beam:
            # Expand all actions
            for action in range(model.n_actions):
                next_belief = model.predict_next_state(current_belief, action)
                efe = expected_free_energy(current_belief, action, model)

                new_actions = actions + [action]
                new_cum_efe = cum_efe + efe

                candidates.append((next_belief, new_actions, new_cum_efe))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[2])
        beam = candidates[:beam_width]

    # Return best sequence
    best_sequence = beam[0][1]
    return best_sequence[0]  # First action
```

**Complexity**: O(horizon × beam_width × n_actions)

**When to Use**:
- Balance between greedy and exhaustive search
- Medium-large action spaces
- Constrained computation

---

### 5. Hierarchical Planning

**Purpose**: Plan at multiple levels of abstraction

**Algorithm**:
```
1. High-level planning:
    - Select abstract goal
    - Plan sequence of subgoals
2. Low-level planning:
    - For each subgoal:
        - Plan concrete actions
        - Execute until subgoal reached
```

**Implementation**:
```python
class HierarchicalPlanner:
    def __init__(self, high_level_model, low_level_model):
        self.high_model = high_level_model
        self.low_model = low_level_model

    def plan(self, belief):
        # High-level planning
        subgoals = self.plan_high_level(belief)

        # Low-level planning
        actions = []
        current_belief = belief

        for subgoal in subgoals:
            subgoal_actions = self.plan_to_subgoal(
                current_belief,
                subgoal
            )
            actions.extend(subgoal_actions)

            # Update belief
            for action in subgoal_actions:
                current_belief = self.low_model.predict_next_state(
                    current_belief,
                    action
                )

        return actions[0]  # Return first action

    def plan_high_level(self, belief):
        """Plan sequence of abstract subgoals."""
        # Use high-level model
        subgoals = []
        current_abstract = self.abstract_belief(belief)

        for _ in range(self.high_horizon):
            action = plan_action(current_abstract, self.high_model)
            subgoal = self.high_model.predict_next_state(
                current_abstract,
                action
            )
            subgoals.append(subgoal)
            current_abstract = subgoal

        return subgoals

    def plan_to_subgoal(self, belief, subgoal):
        """Plan low-level actions to reach subgoal."""
        # Temporarily modify preferences to reach subgoal
        modified_model = self.low_model.replace(
            C=self.subgoal_to_preferences(subgoal)
        )

        # Plan with modified preferences
        return plan_action(belief, modified_model)
```

---

## Policy Evaluation

### Expected Free Energy Decomposition

**Pragmatic Value** (Goal-seeking):
\[ G_{\text{pragmatic}}(a) = -E_{Q(s'|a)}[\log P(o')] \]

**Epistemic Value** (Information-seeking):
\[ G_{\text{epistemic}}(a) = H[P(o'|s')] - E_{Q(s'|a)}[H[P(o'|s')]] \]

**Implementation**:
```python
def decompose_efe(belief, action, model):
    """Decompose EFE into pragmatic and epistemic components."""

    # Predict next state
    next_belief = model.predict_next_state(belief, action)

    # Pragmatic: Expected log preference
    expected_obs = model.predict_observation(next_belief)
    pragmatic = -(expected_obs * model.C).sum()

    # Epistemic: Information gain
    # H[P(o')] - E[H[P(o'|s')]]
    obs_entropy = calculate_entropy(expected_obs)

    expected_cond_entropy = 0.0
    for s in range(model.n_states):
        obs_given_s = model.A[:, s]
        cond_entropy = calculate_entropy(obs_given_s)
        expected_cond_entropy += next_belief[s] * cond_entropy

    epistemic = obs_entropy - expected_cond_entropy

    total_efe = pragmatic + epistemic

    return {
        'total': total_efe,
        'pragmatic': pragmatic,
        'epistemic': epistemic
    }
```

---

## Policy Optimization

### Gradient-Based Policy Optimization

```python
import jax
import jax.numpy as jnp
from jax import grad

def policy_gradient_step(policy_params, belief, model, learning_rate=0.01):
    """Update policy parameters using gradient descent."""

    def policy_loss(params):
        """Compute policy loss (expected free energy)."""
        action_logits = policy_network(params, belief)
        action_probs = jax.nn.softmax(action_logits)

        # Expected EFE under policy
        efes = jnp.array([
            expected_free_energy(belief, action, model)
            for action in range(model.n_actions)
        ])

        expected_efe = (action_probs * efes).sum()
        return expected_efe

    # Compute gradient
    loss, grads = jax.value_and_grad(policy_loss)(policy_params)

    # Update parameters
    new_params = jax.tree_map(
        lambda p, g: p - learning_rate * g,
        policy_params,
        grads
    )

    return new_params, loss
```

---

## Planning Diagnostics

### Visualize Planning Tree

```python
from active_inference.visualization import plot_planning_tree

def visualize_planning(belief, model, horizon=3):
    """Visualize planning tree."""

    # Build tree
    tree = build_planning_tree(belief, model, horizon)

    # Plot
    fig, ax = plot_planning_tree(
        tree,
        show_efe=True,
        show_probabilities=True,
        save_path="planning_tree.png"
    )
```

### Analyze Planning Quality

```python
def analyze_planning_quality(planned_actions, actual_rewards):
    """Analyze how good planning was."""

    # Correlation between predicted EFE and actual reward
    from active_inference.utils import pearson_correlation

    corr = pearson_correlation(-predicted_efes, actual_rewards)

    print(f"Planning quality correlation: {corr.correlation:.3f}")
    print(f"P-value: {corr.p_value:.4f}")

    if corr.correlation > 0.5:
        print("✓ Good planning quality")
    else:
        print("⚠️ Poor planning quality")
```

---

## Performance Comparison

| Algorithm | Time per Decision | Optimality | Memory | Best For |
|-----------|------------------|------------|---------|----------|
| Greedy | ~1ms | Fair | O(n_states) | Real-time |
| Fixed Horizon | ~10ms | Good | O(horizon) | Medium tasks |
| MCTS | ~100ms | Very Good | O(iterations) | Complex planning |
| Beam Search | ~50ms | Good | O(beam_width) | Large action spaces |
| Hierarchical | ~20ms | Very Good | O(levels) | Structured tasks |

---

## Cross-References

- [Agent Module](module_agents.md#planning) - Planning implementation
- [Theory](theory.md#action-as-inference) - Theoretical background
- [Performance Guide](performance.md) - Planning optimization
- [Core Module](module_core.md#free-energy) - EFE calculation

---

## Examples

- [Example 02: Grid World Agent](../examples/02_grid_world_agent.py)
- [Example 04: MDP Example](../examples/04_mdp_example.py)

---

> **Next**: [Agent Module](module_agents.md) | [Performance Guide](performance.md) | [Theory](theory.md)
