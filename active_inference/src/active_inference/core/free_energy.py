"""Free energy calculations for active inference.

Implements variational free energy (for inference) and expected free energy
(for planning/action selection).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .generative_model import GenerativeModel


def variational_free_energy(
    observation: int,
    state_belief: Float[Array, "n_states"],
    model: GenerativeModel,
) -> Float[Array, ""]:
    """Calculate variational free energy for a given observation and belief.

    Variational free energy quantifies the quality of the posterior approximation:
    F = E_Q[log Q(s) - log P(o,s)]
      = -E_Q[log P(o|s)] - E_Q[log P(s)] + E_Q[log Q(s)]
      = Accuracy - Complexity

    Lower free energy indicates better inference.

    **Arguments:**

    - `observation`: Observed index
    - `state_belief`: Approximate posterior Q(s)
    - `model`: Generative model

    **Returns:**

    - Scalar free energy value
    """
    # Ensure numerical stability
    state_belief = state_belief + 1e-16
    state_belief = state_belief / jnp.sum(state_belief)

    # Accuracy: E_Q[log P(o|s)]
    obs_likelihood = model.get_observation_likelihood(observation)
    log_likelihood = jnp.log(obs_likelihood + 1e-16)
    accuracy = jnp.sum(state_belief * log_likelihood)

    # Complexity: KL[Q(s) || P(s)]
    log_prior = jnp.log(model.D + 1e-16)
    log_posterior = jnp.log(state_belief + 1e-16)
    complexity = jnp.sum(state_belief * (log_posterior - log_prior))

    # Free energy = -Accuracy + Complexity
    free_energy = -accuracy + complexity

    return free_energy


def expected_free_energy(
    state_belief: Float[Array, "n_states"],
    action: int,
    model: GenerativeModel,
    planning_horizon: int = 1,
) -> Float[Array, ""]:
    """Calculate expected free energy for action selection.

    Expected free energy quantifies the value of an action:
    G = E_Q[log Q(o|π) - log P(o)] - E_Q[H[P(o|s,π)]]
      = Pragmatic value (preference satisfaction) + Epistemic value (information gain)

    Lower expected free energy indicates more valuable actions.

    **Arguments:**

    - `state_belief`: Current posterior over states
    - `action`: Action to evaluate
    - `model`: Generative model
    - `planning_horizon`: Number of steps to look ahead (currently only supports 1)

    **Returns:**

    - Scalar expected free energy
    """
    # Ensure numerical stability
    state_belief = state_belief + 1e-16
    state_belief = state_belief / jnp.sum(state_belief)

    # Predict next state distribution
    next_state_belief = model.predict_next_state(state_belief, action)
    next_state_belief = next_state_belief + 1e-16
    next_state_belief = next_state_belief / jnp.sum(next_state_belief)

    # Predict observation distribution
    predicted_obs = model.predict_observation(next_state_belief)
    predicted_obs = predicted_obs + 1e-16
    predicted_obs = predicted_obs / jnp.sum(predicted_obs)

    # Pragmatic value: alignment with preferences
    # G_pragmatic = KL[Q(o|π) || P(o)]
    preferred_obs = jnp.exp(model.C)
    preferred_obs = preferred_obs / (jnp.sum(preferred_obs) + 1e-16)

    log_predicted = jnp.log(predicted_obs + 1e-16)
    log_preferred = jnp.log(preferred_obs + 1e-16)
    pragmatic_value = jnp.sum(predicted_obs * (log_predicted - log_preferred))

    # Epistemic value: information gain (ambiguity reduction)
    # G_epistemic = E_Q[H[P(o|s)]]
    # This is the expected entropy of observations given states
    entropy_per_state = jnp.zeros(model.n_states)
    for s in range(model.n_states):
        obs_dist = model.A[:, s]
        obs_dist = obs_dist + 1e-16
        obs_dist = obs_dist / jnp.sum(obs_dist)
        entropy_per_state = entropy_per_state.at[s].set(-jnp.sum(obs_dist * jnp.log(obs_dist + 1e-16)))

    epistemic_value = jnp.sum(next_state_belief * entropy_per_state)

    # Expected free energy
    # Lower is better: minimize pragmatic cost, minimize ambiguity
    efe = pragmatic_value + epistemic_value

    return efe


def batch_expected_free_energy(
    state_belief: Float[Array, "n_states"],
    model: GenerativeModel,
    planning_horizon: int = 1,
) -> Float[Array, "n_actions"]:
    """Calculate expected free energy for all actions.

    **Arguments:**

    - `state_belief`: Current posterior over states
    - `model`: Generative model
    - `planning_horizon`: Number of steps to look ahead

    **Returns:**

    - Expected free energy for each action
    """
    efe_values = jnp.zeros(model.n_actions)
    for action in range(model.n_actions):
        efe_values = efe_values.at[action].set(expected_free_energy(state_belief, action, model, planning_horizon))
    return efe_values
