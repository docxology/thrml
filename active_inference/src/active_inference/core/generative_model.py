"""Generative model implementations for active inference.

This module provides generative model structures that define the agent's beliefs
about the world, including state transitions, observations, and preferences.
"""

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass
class GenerativeModel(eqx.Module):
    """A discrete generative model for active inference.

    This implements a partially observable Markov decision process (POMDP) style
    generative model with:
    - State transitions: P(s_t | s_{t-1}, a_{t-1})
    - Observations: P(o_t | s_t)
    - Preferences: P(o_t) (goal-directed prior)
    - Initial state prior: P(s_0)

    **Attributes:**

    - `A`: Observation likelihood matrix [n_observations, n_states]
    - `B`: State transition tensor [n_states, n_states, n_actions]
    - `C`: Preferred observations (log preferences) [n_observations]
    - `D`: Initial state prior [n_states]
    - `n_states`: Number of hidden states
    - `n_observations`: Number of observations
    - `n_actions`: Number of actions
    """

    A: Float[Array, "n_obs n_states"]
    B: Float[Array, "n_states n_states n_actions"]
    C: Float[Array, "n_obs"]
    D: Float[Array, "n_states"]
    n_states: int
    n_observations: int
    n_actions: int

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        n_actions: int,
        A: Optional[Float[Array, "n_obs n_states"]] = None,
        B: Optional[Float[Array, "n_states n_states n_actions"]] = None,
        C: Optional[Float[Array, "n_obs"]] = None,
        D: Optional[Float[Array, "n_states"]] = None,
    ):
        """Initialize a generative model.

        **Arguments:**

        - `n_states`: Number of hidden states
        - `n_observations`: Number of possible observations
        - `n_actions`: Number of possible actions
        - `A`: Observation likelihood matrix (uniform if None)
        - `B`: State transition tensor (uniform if None)
        - `C`: Preferred observations (flat if None)
        - `D`: Initial state prior (uniform if None)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions

        # Initialize with uniform distributions if not provided
        if A is None:
            A = jnp.ones((n_observations, n_states)) / n_observations
        if B is None:
            B = jnp.ones((n_states, n_states, n_actions)) / n_states
        if C is None:
            C = jnp.zeros(n_observations)
        if D is None:
            D = jnp.ones(n_states) / n_states

        # Normalize probability distributions
        self.A = A / jnp.sum(A, axis=0, keepdims=True)
        self.B = B / jnp.sum(B, axis=0, keepdims=True)
        self.C = C
        self.D = D / jnp.sum(D)

    def get_observation_likelihood(self, observation: int) -> Float[Array, "n_states"]:
        """Get likelihood P(o|s) for a given observation.

        **Arguments:**

        - `observation`: The observed index

        **Returns:**

        - Likelihood vector over states
        """
        return self.A[observation, :]

    def get_state_transition(self, action: int) -> Float[Array, "n_states n_states"]:
        """Get transition matrix P(s'|s,a) for a given action.

        **Arguments:**

        - `action`: The action index

        **Returns:**

        - Transition matrix [n_states, n_states]
        """
        return self.B[:, :, action]

    def predict_observation(self, state_belief: Float[Array, "n_states"]) -> Float[Array, "n_obs"]:
        """Predict observation distribution from state belief.

        **Arguments:**

        - `state_belief`: Posterior over states

        **Returns:**

        - Predicted observation distribution
        """
        return self.A @ state_belief

    def predict_next_state(self, state_belief: Float[Array, "n_states"], action: int) -> Float[Array, "n_states"]:
        """Predict next state distribution given current belief and action.

        **Arguments:**

        - `state_belief`: Current posterior over states
        - `action`: Action to take

        **Returns:**

        - Predicted next state distribution
        """
        transition_matrix = self.get_state_transition(action)
        return transition_matrix @ state_belief


@dataclass
class HierarchicalGenerativeModel(eqx.Module):
    """A hierarchical generative model with multiple levels.

    Implements a hierarchical POMDP where higher levels provide context
    for lower levels, enabling abstract reasoning and planning.

    **Attributes:**

    - `levels`: List of GenerativeModel instances for each level
    - `n_levels`: Number of hierarchical levels
    """

    levels: list[GenerativeModel]
    n_levels: int

    def __init__(self, levels: list[GenerativeModel]):
        """Initialize a hierarchical generative model.

        **Arguments:**

        - `levels`: List of GenerativeModel instances, ordered from lowest to highest
        """
        self.levels = levels
        self.n_levels = len(levels)

    def get_level(self, level: int) -> GenerativeModel:
        """Get the generative model at a specific level.

        **Arguments:**

        - `level`: Level index (0 = lowest level)

        **Returns:**

        - GenerativeModel at the specified level
        """
        return self.levels[level]


def normalize_distribution(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Normalize an array to sum to 1.

    **Arguments:**

    - `x`: Array to normalize

    **Returns:**

    - Normalized array
    """
    return x / (jnp.sum(x) + 1e-16)


def softmax_stable(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable softmax.

    **Arguments:**

    - `x`: Log probabilities

    **Returns:**

    - Normalized probabilities
    """
    x_max = jnp.max(x)
    exp_x = jnp.exp(x - x_max)
    return exp_x / (jnp.sum(exp_x) + 1e-16)
