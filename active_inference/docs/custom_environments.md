# Building Custom Environments

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [Environment Module](module_environments.md)

Guide to creating custom environments for testing and evaluating active inference agents.

## Overview

Custom environments allow you to test agents in specific scenarios beyond the pre-built GridWorld and TMaze environments. This guide covers the complete process from interface definition to integration.

## Environment Interface

### Standard API

All environments should implement:

```python
import equinox as eqx
import jax.numpy as jnp
from typing import Tuple

class CustomEnvironment(eqx.Module):
    """Template for custom environments."""

    # Configuration
    state_dim: int
    obs_dim: int
    action_dim: int

    def reset(self, key: jax.random.PRNGKey) -> int:
        """Reset environment to initial state.

        Args:
            key: JAX random key

        Returns:
            observation: Initial observation index
        """
        pass

    def step(self, key: jax.random.PRNGKey, action: int) -> Tuple[int, float, bool]:
        """Execute action and return next observation.

        Args:
            key: JAX random key
            action: Action index

        Returns:
            observation: Next observation index
            reward: Reward value
            done: Episode termination flag
        """
        pass

    def render(self) -> str:
        """Return string visualization of current state."""
        pass

    def get_observation(self, state) -> int:
        """Map state to observation."""
        pass
```

---

## Step-by-Step Environment Building

### Step 1: Define State Space

**Example**: Simple card game

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class CardGameEnvironment(eqx.Module):
    """Simple card selection game."""

    # Configuration
    n_cards: int = 5
    n_observations: int = 5
    n_actions: int = 2  # select or pass

    # State
    current_card: int = 0
    target_card: int = 0
    score: int = 0
    step_count: int = 0
    max_steps: int = 10
```

---

### Step 2: Implement Reset

```python
def reset(self, key: jax.random.PRNGKey) -> int:
    """Initialize new game."""
    key1, key2 = jax.random.split(key)

    # Random target card
    target = jax.random.randint(key1, (), 0, self.n_cards)

    # Random starting card
    current = jax.random.randint(key2, (), 0, self.n_cards)

    # Reset state
    self = eqx.tree_at(
        lambda env: (env.current_card, env.target_card, env.score, env.step_count),
        self,
        (current, target, 0, 0)
    )

    # Return observation
    return self.get_observation(self)
```

---

### Step 3: Implement Step

```python
def step(self, key: jax.random.PRNGKey, action: int) -> Tuple[int, float, bool]:
    """Execute action."""

    # Action 0: Select current card
    if action == 0:
        reward = 1.0 if self.current_card == self.target_card else -0.5
        done = True
    # Action 1: Pass, draw new card
    else:
        self = eqx.tree_at(
            lambda env: env.current_card,
            self,
            jax.random.randint(key, (), 0, self.n_cards)
        )
        reward = -0.01  # Small step cost
        done = False

    # Increment step counter
    self = eqx.tree_at(
        lambda env: env.step_count,
        self,
        self.step_count + 1
    )

    # Check max steps
    if self.step_count >= self.max_steps:
        done = True
        reward = -1.0  # Timeout penalty

    observation = self.get_observation(self)

    return observation, reward, done
```

---

### Step 4: Implement Observation Mapping

```python
def get_observation(self, state) -> int:
    """Map state to observation.

    Observations encode relative position of current card to target.
    """
    diff = self.current_card - self.target_card

    # Map difference to observation
    # 0: current = target
    # 1: current > target (far)
    # 2: current > target (close)
    # 3: current < target (close)
    # 4: current < target (far)

    if diff == 0:
        return 0
    elif diff > 2:
        return 1
    elif diff > 0:
        return 2
    elif diff < -2:
        return 4
    else:
        return 3
```

---

### Step 5: Implement Rendering

```python
def render(self) -> str:
    """Visual representation of game state."""
    cards = ["□"] * self.n_cards
    cards[self.current_card] = "▣"  # Current card
    cards[self.target_card] = "★"  # Target card

    if self.current_card == self.target_card:
        cards[self.current_card] = "✓"  # Match!

    display = " ".join(cards)

    info = f"Step {self.step_count}/{self.max_steps} | Score: {self.score}"

    return f"{display}\n{info}"
```

---

## Complete Environment Example

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple

class CardGameEnvironment(eqx.Module):
    """Complete card game environment."""

    n_cards: int = 5
    n_observations: int = 5
    n_actions: int = 2
    max_steps: int = 10

    current_card: int = 0
    target_card: int = 0
    score: int = 0
    step_count: int = 0

    def reset(self, key: jax.random.PRNGKey) -> int:
        key1, key2 = jax.random.split(key)

        target = jax.random.randint(key1, (), 0, self.n_cards)
        current = jax.random.randint(key2, (), 0, self.n_cards)

        self = eqx.tree_at(
            lambda env: (env.current_card, env.target_card, env.score, env.step_count),
            self,
            (current, target, 0, 0)
        )

        return self.get_observation(self)

    def step(self, key: jax.random.PRNGKey, action: int) -> Tuple[int, float, bool]:
        if action == 0:  # Select
            reward = 1.0 if self.current_card == self.target_card else -0.5
            done = True
        else:  # Pass
            self = eqx.tree_at(
                lambda env: env.current_card,
                self,
                jax.random.randint(key, (), 0, self.n_cards)
            )
            reward = -0.01
            done = False

        self = eqx.tree_at(
            lambda env: (env.step_count, env.score),
            self,
            (self.step_count + 1, self.score + reward)
        )

        if self.step_count >= self.max_steps:
            done = True
            reward = -1.0

        return self.get_observation(self), reward, done

    def get_observation(self, state) -> int:
        diff = self.current_card - self.target_card

        if diff == 0:
            return 0
        elif diff > 2:
            return 1
        elif diff > 0:
            return 2
        elif diff < -2:
            return 4
        else:
            return 3

    def render(self) -> str:
        cards = ["□"] * self.n_cards
        cards[self.current_card] = "▣"
        cards[self.target_card] = "★"

        if self.current_card == self.target_card:
            cards[self.current_card] = "✓"

        display = " ".join(cards)
        info = f"Step {self.step_count}/{self.max_steps} | Score: {self.score:.2f}"

        return f"{display}\n{info}"
```

---

## Testing the Environment

### Basic Test

```python
import jax

# Create environment
env = CardGameEnvironment()

# Test reset
key = jax.random.key(42)
obs = env.reset(key)
print(f"Initial observation: {obs}")
print(env.render())

# Test steps
for step in range(5):
    key, subkey = jax.random.split(key)
    action = jax.random.randint(subkey, (), 0, env.n_actions)

    obs, reward, done = env.step(subkey, action)

    print(f"\nStep {step + 1}: action={action}, reward={reward:.2f}")
    print(env.render())

    if done:
        print("Episode ended!")
        break
```

### Validation

```python
def validate_environment(env, n_episodes=10):
    """Validate environment implementation."""
    key = jax.random.key(42)

    for episode in range(n_episodes):
        key, subkey = jax.random.split(key)
        obs = env.reset(subkey)

        # Check observation valid
        assert 0 <= obs < env.n_observations, f"Invalid observation: {obs}"

        done = False
        step_count = 0

        while not done and step_count < 100:
            key, subkey = jax.random.split(key)
            action = jax.random.randint(subkey, (), 0, env.n_actions)

            obs, reward, done = env.step(subkey, action)

            # Check return types
            assert 0 <= obs < env.n_observations
            assert isinstance(reward, (int, float, jnp.ndarray))
            assert isinstance(done, (bool, jnp.ndarray))

            step_count += 1

        print(f"Episode {episode + 1}: {step_count} steps")

    print("✓ Environment validation passed!")

# Run validation
validate_environment(env)
```

---

## Integration with Agents

### Build Matching Generative Model

```python
from active_inference.core import GenerativeModel
from active_inference.models import create_observation_matrix, create_transition_tensor

# Create model matching environment
def build_card_game_model(env):
    """Build generative model for card game environment."""

    # A matrix: observation likelihood
    # Simple identity mapping
    A = create_observation_matrix(
        n_states=env.n_cards,
        n_observations=env.n_observations,
        noise_level=0.1
    )

    # B tensor: state transitions
    # Action 1 (pass) causes random transition
    B = jnp.zeros((env.n_cards, env.n_cards, env.n_actions))

    # Action 0 (select): stay in state
    for s in range(env.n_cards):
        B = B.at[s, s, 0].set(1.0)

    # Action 1 (pass): uniform transition
    B = B.at[:, :, 1].set(1.0 / env.n_cards)

    # C vector: prefer observation 0 (match)
    C = jnp.zeros(env.n_observations)
    C = C.at[0].set(5.0)  # Strong preference for match

    # D vector: uniform prior
    D = jnp.ones(env.n_cards) / env.n_cards

    model = GenerativeModel(
        n_states=env.n_cards,
        n_observations=env.n_observations,
        n_actions=env.n_actions,
        A=A, B=B, C=C, D=D
    )

    return model
```

### Run Agent in Environment

```python
from active_inference.agents import ActiveInferenceAgent

# Create environment and model
env = CardGameEnvironment()
model = build_card_game_model(env)

# Create agent
agent = ActiveInferenceAgent(model=model, planning_horizon=2)

# Run episode
key = jax.random.key(42)
agent_state = agent.reset()
obs = env.reset(key)

total_reward = 0

for step in range(20):
    # Agent selects action
    key, subkey = jax.random.split(key)
    action, agent_state, fe = agent.step(subkey, obs, agent_state)

    # Environment responds
    key, subkey = jax.random.split(key)
    obs, reward, done = env.step(subkey, action)

    total_reward += reward

    print(f"Step {step}: action={action}, reward={reward:.2f}, FE={fe:.3f}")
    print(env.render())

    if done:
        break

print(f"\nTotal reward: {total_reward:.2f}")
```

---

## Advanced Patterns

### Continuous State Discretization

```python
class ContinuousCartPole(eqx.Module):
    """Cart-pole with continuous state discretized for active inference."""

    n_position_bins: int = 10
    n_velocity_bins: int = 10
    n_angle_bins: int = 10

    def continuous_to_discrete(self, continuous_state):
        """Map continuous state to discrete observation."""
        position, velocity, angle, angular_velocity = continuous_state

        # Discretize each dimension
        pos_bin = jnp.digitize(position, self.position_bins)
        vel_bin = jnp.digitize(velocity, self.velocity_bins)
        angle_bin = jnp.digitize(angle, self.angle_bins)

        # Combine into single observation index
        obs = (pos_bin * self.n_velocity_bins * self.n_angle_bins +
               vel_bin * self.n_angle_bins +
               angle_bin)

        return obs
```

### Multi-Agent Environments

```python
class MultiAgentEnvironment(eqx.Module):
    """Environment with multiple agents."""

    n_agents: int

    def step(self, key, actions):
        """Step with action from each agent.

        Args:
            actions: Array[n_agents] of action indices

        Returns:
            observations: Array[n_agents] of observations
            rewards: Array[n_agents] of rewards
            done: bool for episode end
        """
        # Update environment based on joint actions
        # Return per-agent observations and rewards
        pass
```

### Hierarchical Environments

```python
class HierarchicalMaze(eqx.Module):
    """Maze with room structure for hierarchical agents."""

    def get_room_observation(self, position):
        """Abstract room-level observation."""
        pass

    def get_local_observation(self, position):
        """Detailed position-level observation."""
        pass

    def step(self, key, action):
        # Return both levels of observation
        room_obs = self.get_room_observation(self.position)
        local_obs = self.get_local_observation(self.position)

        return (room_obs, local_obs), reward, done
```

---

## Environment Testing Utilities

### Automated Testing

```python
def test_environment_interface(env_class, **env_kwargs):
    """Test environment follows standard interface."""
    env = env_class(**env_kwargs)
    key = jax.random.key(42)

    # Test reset
    obs = env.reset(key)
    assert isinstance(obs, (int, jnp.ndarray)), "reset() must return observation"

    # Test step
    key, subkey = jax.random.split(key)
    obs, reward, done = env.step(subkey, 0)

    assert isinstance(obs, (int, jnp.ndarray)), "step() must return observation"
    assert isinstance(reward, (int, float, jnp.ndarray)), "step() must return reward"
    assert isinstance(done, (bool, jnp.ndarray)), "step() must return done flag"

    # Test render
    render_str = env.render()
    assert isinstance(render_str, str), "render() must return string"

    print("✓ Environment interface test passed!")
```

### Performance Benchmarking

```python
import time

def benchmark_environment(env, n_steps=1000):
    """Measure environment performance."""
    key = jax.random.key(42)

    # Warmup
    for _ in range(10):
        obs = env.reset(key)
        env.step(key, 0)

    # Benchmark
    start = time.time()

    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, (), 0, env.n_actions)
        obs, reward, done = env.step(subkey, action)

        if done:
            obs = env.reset(subkey)

    elapsed = time.time() - start

    print(f"Benchmark: {n_steps} steps in {elapsed:.2f}s")
    print(f"Throughput: {n_steps / elapsed:.0f} steps/sec")
```

---

## Cross-References

- [Environment Module](module_environments.md) - Pre-built environments
- [Model Module](module_models.md) - Matching models
- [Agent Module](module_agents.md) - Running agents
- [Custom Models](custom_models.md) - Build matching generative models

---

## Examples

- [Example 02: Grid World Agent](../examples/02_grid_world_agent.py)
- [Example 05: POMDP Example](../examples/05_pomdp_example.py)

---

> **Next**: [Environment Module](module_environments.md) | [Custom Models](custom_models.md) | [Agent Module](module_agents.md)
