"""Grid world environment for testing active inference agents."""

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Key


@dataclass
class GridWorldConfig(eqx.Module):
    """Configuration for grid world environment.

    **Attributes:**

    - `size`: Grid size (square grid)
    - `n_observations`: Number of unique observations
    - `observation_noise`: Probability of noisy observation
    - `goal_location`: Coordinates of goal (None for no goal)
    - `obstacle_locations`: List of obstacle coordinates
    """

    size: int
    n_observations: int
    observation_noise: float
    goal_location: Optional[tuple[int, int]]
    obstacle_locations: list[tuple[int, int]]

    def __init__(
        self,
        size: int = 5,
        n_observations: int = 10,
        observation_noise: float = 0.1,
        goal_location: Optional[tuple[int, int]] = None,
        obstacle_locations: Optional[list[tuple[int, int]]] = None,
    ):
        """Initialize grid world configuration."""
        self.size = size
        self.n_observations = n_observations
        self.observation_noise = observation_noise
        self.goal_location = goal_location if goal_location is not None else (size - 1, size - 1)
        self.obstacle_locations = obstacle_locations if obstacle_locations is not None else []


class GridWorld(eqx.Module):
    """A grid world environment for active inference.

    Agent navigates a 2D grid, receiving observations and rewards.
    Observations are noisy indicators of location.

    Actions: 0=up, 1=right, 2=down, 3=left

    **Attributes:**

    - `config`: Grid world configuration
    - `n_states`: Total number of states (grid cells)
    - `n_observations`: Number of possible observations
    - `n_actions`: Number of possible actions (4 directions)
    - `current_state`: Current agent location
    """

    config: GridWorldConfig
    n_states: int
    n_observations: int
    n_actions: int = 4
    current_state: Int[Array, "2"]

    def __init__(self, config: Optional[GridWorldConfig] = None, **kwargs):
        """Initialize grid world.

        **Arguments:**

        - `config`: GridWorldConfig or None (uses kwargs if None)
        - `**kwargs`: Parameters for GridWorldConfig if config is None
        """
        if config is None:
            config = GridWorldConfig(**kwargs)

        self.config = config
        self.n_states = config.size * config.size
        self.n_observations = config.n_observations
        self.current_state = jnp.array([0, 0])

    def reset(self, key: Key[Array, ""]) -> tuple["GridWorld", int]:
        """Reset environment to random state.

        **Arguments:**

        - `key`: JAX random key

        **Returns:**

        - Tuple of (new_environment, initial_observation)
        """
        # Start at random non-obstacle location
        valid_start = False
        while not valid_start:
            row = jax.random.randint(key, (), 0, self.config.size)
            col = jax.random.randint(key, (), 0, self.config.size)
            location = (int(row), int(col))
            valid_start = location not in self.config.obstacle_locations

        new_state = jnp.array([row, col])
        new_env = eqx.tree_at(lambda e: e.current_state, self, new_state)
        return new_env, new_env.get_observation(key)

    def get_observation(self, key: Key[Array, ""]) -> int:
        """Get observation from current state.

        **Arguments:**

        - `key`: JAX random key for noise

        **Returns:**

        - Observation index
        """
        # True observation based on location
        state_idx = self.state_to_index(self.current_state)
        true_obs = state_idx % self.n_observations

        # Add observation noise
        if jax.random.uniform(key) < self.config.observation_noise:
            return int(jax.random.randint(key, (), 0, self.n_observations))
        return int(true_obs)

    def step(
        self,
        key: Key[Array, ""],
        action: int,
    ) -> tuple["GridWorld", int, float, bool]:
        """Take action and return new environment, observation, reward, done.

        **Arguments:**

        - `key`: JAX random key
        - `action`: Action index (0=up, 1=right, 2=down, 3=left)

        **Returns:**

        - Tuple of (new_environment, observation, reward, done)
        """
        # Apply action
        new_state = self.current_state.copy()

        if action == 0:  # up
            new_state = new_state.at[0].add(-1)
        elif action == 1:  # right
            new_state = new_state.at[1].add(1)
        elif action == 2:  # down
            new_state = new_state.at[0].add(1)
        elif action == 3:  # left
            new_state = new_state.at[1].add(-1)

        # Check bounds
        new_state = jnp.clip(new_state, 0, self.config.size - 1)

        # Check obstacles
        location = (int(new_state[0]), int(new_state[1]))
        if location in self.config.obstacle_locations:
            new_state = self.current_state  # Stay in place if obstacle

        # Update environment
        new_env = eqx.tree_at(lambda e: e.current_state, self, new_state)

        # Check goal
        done = tuple(new_env.current_state) == self.config.goal_location
        reward = 1.0 if done else 0.0

        # Get observation
        obs = new_env.get_observation(key)

        return new_env, obs, reward, done

    def state_to_index(self, state: Int[Array, "2"]) -> int:
        """Convert 2D state to flat index.

        **Arguments:**

        - `state`: 2D coordinates

        **Returns:**

        - Flat state index
        """
        return int(state[0] * self.config.size + state[1])

    def index_to_state(self, index: int) -> Int[Array, "2"]:
        """Convert flat index to 2D state.

        **Arguments:**

        - `index`: Flat state index

        **Returns:**

        - 2D coordinates
        """
        row = index // self.config.size
        col = index % self.config.size
        return jnp.array([row, col])

    def is_goal(self, state: Optional[Int[Array, "2"]] = None) -> bool:
        """Check if state is goal.

        **Arguments:**

        - `state`: State to check (current if None)

        **Returns:**

        - True if at goal
        """
        if state is None:
            state = self.current_state
        return tuple(state) == self.config.goal_location
