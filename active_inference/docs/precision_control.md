# Precision Control

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Core Module](module_core.md) | [Theory](theory.md)

Understanding and controlling precision parameters in active inference.

## Overview

Precision parameters control the influence of different information sources in active inference, affecting exploration vs exploitation, belief updating speed, and action selection strategies.

## Precision Types

### 1. Sensory Precision (γ_sensory)

**Purpose**: Controls trust in observations

**Formula**: Weighted prediction error
\[ \epsilon_{\text{weighted}} = \gamma_{\text{sensory}} \cdot (o - \hat{o}) \]

**Effects**:
- **High (> 1.0)**: Trust observations more, faster belief updates
- **Low (< 1.0)**: Trust prior more, slower belief updates

**Example**:
```python
from active_inference.core import Precision

# High sensory precision: Trust observations
high_sensory = Precision(sensory_precision=2.0)

# Low sensory precision: Trust prior/model
low_sensory = Precision(sensory_precision=0.5)
```

---

### 2. State Precision (γ_state)

**Purpose**: Controls trust in state predictions

**Effects**:
- **High**: Confident in model predictions
- **Low**: Uncertain about predictions

**Usage**:
```python
# Trust model transitions
confident_model = Precision(state_precision=2.0)

# Uncertain about model
uncertain_model = Precision(state_precision=0.5)
```

---

### 3. Action Precision (β)

**Purpose**: Controls decisiveness in action selection

**Formula**: Softmax temperature for action distribution
\[ P(a) = \frac{\exp(-\beta \cdot G(a))}{\sum_{a'} \exp(-\beta \cdot G(a'))} \]

**Effects**:
- **High (> 1.0)**: Deterministic, exploit best action
- **Low (< 1.0)**: Stochastic, explore multiple actions

**Example**:
```python
# Exploitation (deterministic)
exploitative = Precision(action_precision=5.0)

# Exploration (stochastic)
exploratory = Precision(action_precision=0.5)
```

---

## Precision Effects Visualization

### Sensory Precision

```python
from active_inference.core import Precision, GenerativeModel
from active_inference.inference import infer_states
import jax.numpy as jnp

model = GenerativeModel(n_states=4, n_observations=2, n_actions=2)

observation = 1
prior = model.D

# Compare precisions
precisions = [0.5, 1.0, 2.0, 5.0]

for prec in precisions:
    # Apply precision to likelihood
    likelihood = model.get_observation_likelihood(observation)
    weighted_likelihood = likelihood ** prec

    # Infer with weighted likelihood
    posterior = weighted_likelihood * prior
    posterior = posterior / posterior.sum()

    print(f"Precision {prec:.1f}: {posterior}")
```

**Output interpretation**:
- Low precision → Posterior closer to prior (slow update)
- High precision → Posterior closer to likelihood (fast update)

---

### Action Precision

```python
from active_inference.core import expected_free_energy
import jax

def get_action_distribution(belief, model, action_precision):
    """Get action distribution with given precision."""

    # Calculate EFE for all actions
    efes = jnp.array([
        expected_free_energy(belief, action, model)
        for action in range(model.n_actions)
    ])

    # Apply precision (inverse temperature)
    action_probs = jax.nn.softmax(-action_precision * efes)

    return action_probs

# Compare precisions
belief = jnp.array([0.6, 0.3, 0.1, 0.0])

for prec in [0.5, 1.0, 2.0, 5.0]:
    action_dist = get_action_distribution(belief, model, prec)
    print(f"Precision {prec:.1f}: {action_dist}")
    print(f"  Entropy: {-(action_dist * jnp.log(action_dist + 1e-10)).sum():.3f}")
```

**Output interpretation**:
- Low precision → Flat distribution (high entropy, exploration)
- High precision → Peaked distribution (low entropy, exploitation)

---

## Practical Applications

### Application 1: Temperature Scheduling

Anneal action precision over time (like simulated annealing):

```python
def get_annealed_precision(step, max_steps, start_prec=0.5, end_prec=5.0):
    """Linearly anneal precision from exploration to exploitation."""
    progress = step / max_steps
    return start_prec + (end_prec - start_prec) * progress

# Usage in agent loop
for step in range(max_steps):
    action_precision = get_annealed_precision(step, max_steps)
    precision = Precision(action_precision=action_precision)

    agent = ActiveInferenceAgent(model, precision=precision)
    action, state, fe = agent.step(key, obs, state)
```

---

### Application 2: Adaptive Precision

Adjust precision based on uncertainty:

```python
def adaptive_precision(belief):
    """Increase precision when confident, decrease when uncertain."""
    entropy = -(belief * jnp.log(belief + 1e-10)).sum()
    max_entropy = jnp.log(len(belief))

    # Low entropy → high precision (exploit)
    # High entropy → low precision (explore)
    normalized_entropy = entropy / max_entropy
    action_precision = 0.5 + 4.5 * (1.0 - normalized_entropy)

    return Precision(action_precision=action_precision)

# Usage
precision = adaptive_precision(current_belief)
agent = ActiveInferenceAgent(model, precision=precision)
```

---

### Application 3: Context-Dependent Precision

Different precisions for different contexts:

```python
class ContextualPrecision:
    """Precision that adapts to context."""

    def __init__(self):
        self.exploration_precision = Precision(action_precision=0.5)
        self.exploitation_precision = Precision(action_precision=5.0)

    def get_precision(self, context):
        if context == "explore":
            return self.exploration_precision
        elif context == "exploit":
            return self.exploitation_precision
        else:
            return Precision(action_precision=1.0)
```

---

## Experimental Comparison

### Setup Experiment

```python
from active_inference.agents import ActiveInferenceAgent
from active_inference.environments import GridWorld
import jax

def run_episode_with_precision(env, model, precision, key):
    """Run episode with given precision."""
    agent = ActiveInferenceAgent(model, precision=precision)
    agent_state = agent.reset()
    obs = env.reset(key)

    total_reward = 0
    steps = 0

    while steps < 100:
        key, subkey = jax.random.split(key)
        action, agent_state, fe = agent.step(subkey, obs, agent_state)

        key, subkey = jax.random.split(key)
        obs, reward, done = env.step(subkey, action)

        total_reward += reward
        steps += 1

        if done:
            break

    return total_reward, steps

# Test different precisions
env = GridWorld(size=5)
model = build_grid_world_model(env.config)

precisions = [0.5, 1.0, 2.0, 5.0]
results = {}

for prec in precisions:
    precision = Precision(action_precision=prec)
    key = jax.random.key(42)

    # Run multiple episodes
    rewards = []
    for episode in range(10):
        key, subkey = jax.random.split(key)
        reward, steps = run_episode_with_precision(env, model, precision, subkey)
        rewards.append(reward)

    results[prec] = {
        'mean_reward': jnp.mean(jnp.array(rewards)),
        'std_reward': jnp.std(jnp.array(rewards))
    }

    print(f"Precision {prec}: {results[prec]['mean_reward']:.2f} ± {results[prec]['std_reward']:.2f}")
```

---

## Visualization

### Plot Precision Effects

```python
from active_inference import visualization as viz

# Collect data with different precisions
beliefs_low = []  # Low precision trajectory
beliefs_high = []  # High precision trajectory

# ... run episodes and collect beliefs ...

# Visualize
fig, axes = viz.plot_precision_effects(
    beliefs_low=beliefs_low,
    beliefs_high=beliefs_high,
    precision_values=[0.5, 5.0],
    save_path="precision_effects.png"
)
```

---

## Theoretical Background

### Free Energy Principle

Precision weighting in free energy:

\[ F = E_Q[\log Q(s) - \log P(o,s)] = -\gamma_{\text{sensory}} \cdot E_Q[\log P(o|s)] + D_{KL}[Q(s) || P(s)] \]

Higher sensory precision → Stronger influence of observation likelihood

### Expected Free Energy

Action selection with precision:

\[ P(a) \propto \exp(-\beta \cdot G(a)) \]

Where:
- β: action precision (inverse temperature)
- G(a): expected free energy of action a

---

## Best Practices

### 1. Start with Default (1.0)

```python
# Balanced precision
default_precision = Precision(
    sensory_precision=1.0,
    action_precision=1.0
)
```

### 2. Tune Based on Task

| Task Type | Sensory | Action | Rationale |
|-----------|---------|--------|-----------|
| Noisy observations | Low (0.5) | Medium (1.0) | Don't over-trust noisy data |
| Accurate sensors | High (2.0) | Medium (1.0) | Trust reliable observations |
| Exploration phase | Medium (1.0) | Low (0.5) | Encourage diverse actions |
| Exploitation phase | Medium (1.0) | High (5.0) | Commit to best action |

### 3. Monitor Behavior

```python
def diagnose_precision(agent, env, precision):
    """Analyze agent behavior with given precision."""

    # Collect action distribution
    action_counts = jnp.zeros(model.n_actions)

    for _ in range(100):
        action, _, _ = agent.step(key, obs, state)
        action_counts = action_counts.at[action].add(1)

    # Analyze diversity
    action_dist = action_counts / action_counts.sum()
    entropy = -(action_dist * jnp.log(action_dist + 1e-10)).sum()

    print(f"Action entropy: {entropy:.3f}")
    print(f"Max entropy: {jnp.log(model.n_actions):.3f}")

    if entropy < 0.5:
        print("⚠️ Low entropy: May be too exploitative")
    elif entropy > jnp.log(model.n_actions) - 0.5:
        print("⚠️ High entropy: May be too exploratory")
```

---

## Cross-References

- [Core Module](module_core.md#precision) - Precision API
- [Theory](theory.md) - Mathematical background
- [Agent Module](module_agents.md) - Using precision in agents
- [Example 03](../examples/03_precision_control.py) - Precision demonstration

---

> **Next**: [Core Module](module_core.md) | [Agent Module](module_agents.md) | [Example 03](../examples/03_precision_control.py)
