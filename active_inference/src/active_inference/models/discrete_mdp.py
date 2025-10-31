"""Build generative models for discrete MDPs."""

import jax.numpy as jnp

from ..core.generative_model import GenerativeModel
from ..environments.grid_world import GridWorldConfig


def build_grid_world_model(
    config: GridWorldConfig,
    goal_preference_strength: float = 2.0,
) -> GenerativeModel:
    """Build a generative model for grid world navigation.

    **Arguments:**

    - `config`: Grid world configuration
    - `goal_preference_strength`: How strongly agent prefers goal observations

    **Returns:**

    - GenerativeModel for the grid world
    """
    n_states = config.size * config.size
    n_obs = config.n_observations
    n_actions = 4  # up, right, down, left

    # Build observation model A[o, s]
    # Each state has a preferred observation with some noise
    A = jnp.ones((n_obs, n_states)) * config.observation_noise / n_obs

    for s in range(n_states):
        # Preferred observation for this state
        preferred_obs = s % n_obs
        A = A.at[preferred_obs, s].set(1.0 - config.observation_noise + config.observation_noise / n_obs)

    # Normalize
    A = A / jnp.sum(A, axis=0, keepdims=True)

    # Build transition model B[s', s, a]
    B = jnp.zeros((n_states, n_states, n_actions))

    for s in range(n_states):
        row = s // config.size
        col = s % config.size

        for a in range(n_actions):
            # Calculate next state based on action
            next_row, next_col = row, col

            if a == 0:  # up
                next_row = max(0, row - 1)
            elif a == 1:  # right
                next_col = min(config.size - 1, col + 1)
            elif a == 2:  # down
                next_row = min(config.size - 1, row + 1)
            elif a == 3:  # left
                next_col = max(0, col - 1)

            # Check if next state is obstacle
            if (next_row, next_col) in config.obstacle_locations:
                next_row, next_col = row, col  # Stay in place

            next_s = next_row * config.size + next_col
            B = B.at[next_s, s, a].set(1.0)

    # Build preference C[o]
    # Prefer observations associated with goal
    C = jnp.zeros(n_obs)
    if config.goal_location is not None:
        goal_s = config.goal_location[0] * config.size + config.goal_location[1]
        goal_obs = goal_s % n_obs
        C = C.at[goal_obs].set(goal_preference_strength)

    # Uniform prior over states
    D = jnp.ones(n_states) / n_states

    return GenerativeModel(
        n_states=n_states,
        n_observations=n_obs,
        n_actions=n_actions,
        A=A,
        B=B,
        C=C,
        D=D,
    )


def build_tmaze_model(
    reward_side: int,
    prior_confidence: float = 0.5,
) -> GenerativeModel:
    """Build a generative model for T-maze navigation.

    **Arguments:**

    - `reward_side`: Which side has reward (0=left, 1=right)
    - `prior_confidence`: Agent's prior confidence about reward side

    **Returns:**

    - GenerativeModel for T-maze
    """
    n_states = 4  # start, junction, left, right
    n_obs = 8  # various observations including cues
    n_actions = 3  # forward, left, right

    # Observation model A[o, s]
    A = jnp.zeros((n_obs, n_states))

    # Start state observations
    A = A.at[0, 0].set(0.5)  # Neutral observation
    A = A.at[4, 0].set(0.25)  # Left cue
    A = A.at[5, 0].set(0.25)  # Right cue

    # Junction observations
    A = A.at[1, 1].set(1.0)

    # Left arm observations
    A = A.at[2, 2].set(1.0)

    # Right arm observations
    A = A.at[3, 3].set(1.0)

    # Normalize
    A = A / jnp.sum(A, axis=0, keepdims=True)

    # Transition model B[s', s, a]
    B = jnp.zeros((n_states, n_states, n_actions))

    # From start (s=0)
    B = B.at[0, 0, 1].set(1.0)  # Left -> stay
    B = B.at[0, 0, 2].set(1.0)  # Right -> stay
    B = B.at[1, 0, 0].set(1.0)  # Forward -> junction

    # From junction (s=1)
    B = B.at[1, 1, 0].set(1.0)  # Forward -> stay
    B = B.at[2, 1, 1].set(1.0)  # Left -> left arm
    B = B.at[3, 1, 2].set(1.0)  # Right -> right arm

    # Terminal states (absorbing)
    for a in range(n_actions):
        B = B.at[2, 2, a].set(1.0)  # Left arm
        B = B.at[3, 3, a].set(1.0)  # Right arm

    # Preferences C[o]
    C = jnp.zeros(n_obs)

    # Prefer reward arm observations
    if reward_side == 0:  # Left has reward
        C = C.at[2].set(2.0)  # Prefer left arm observation
    else:  # Right has reward
        C = C.at[3].set(2.0)  # Prefer right arm observation

    # Initial state prior (start at base)
    D = jnp.array([1.0, 0.0, 0.0, 0.0])

    return GenerativeModel(
        n_states=n_states,
        n_observations=n_obs,
        n_actions=n_actions,
        A=A,
        B=B,
        C=C,
        D=D,
    )
