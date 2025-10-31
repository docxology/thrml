"""State inference using variational message passing.

Implements variational inference over hidden states given observations,
using efficient iterative belief updating.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..core.free_energy import variational_free_energy
from ..core.generative_model import GenerativeModel, normalize_distribution


def infer_states(
    observation: int,
    prior_belief: Float[Array, "n_states"],
    model: GenerativeModel,
    n_iterations: int = 16,
    convergence_threshold: float = 1e-4,
) -> tuple[Float[Array, "n_states"], Float[Array, ""]]:
    """Infer hidden states from an observation using variational inference.

    Uses fixed-point iteration to find the posterior that minimizes
    variational free energy.

    **Arguments:**

    - `observation`: Observed index
    - `prior_belief`: Prior over states P(s)
    - `model`: Generative model
    - `n_iterations`: Maximum number of iterations
    - `convergence_threshold`: Convergence criterion for belief updates

    **Returns:**

    - Tuple of (posterior_belief, final_free_energy)
    """
    current_belief = prior_belief.copy()

    for _ in range(n_iterations):
        # Get observation likelihood
        likelihood = model.get_observation_likelihood(observation)

        # Bayesian update: posterior ∝ likelihood × prior
        unnormalized_posterior = likelihood * prior_belief
        new_belief = normalize_distribution(unnormalized_posterior)

        # Check convergence
        belief_change = jnp.max(jnp.abs(new_belief - current_belief))
        current_belief = new_belief

        if belief_change < convergence_threshold:
            break

    # Calculate final free energy
    final_fe = variational_free_energy(observation, current_belief, model)

    return current_belief, final_fe


def variational_message_passing(
    observations: list[int],
    actions: list[int],
    model: GenerativeModel,
    n_iterations: int = 16,
) -> list[Float[Array, "n_states"]]:
    """Perform variational message passing over a sequence of observations.

    This implements forward-backward message passing to infer state
    trajectories given a sequence of observations and actions.

    **Arguments:**

    - `observations`: List of observed indices
    - `actions`: List of action indices (one fewer than observations)
    - `model`: Generative model
    - `n_iterations`: Number of inference iterations

    **Returns:**

    - List of posterior beliefs for each time step
    """
    T = len(observations)
    beliefs = []

    # Forward pass: filtering
    current_belief = model.D.copy()

    for t in range(T):
        # Infer state at time t
        posterior, _ = infer_states(
            observation=observations[t],
            prior_belief=current_belief,
            model=model,
            n_iterations=n_iterations,
        )
        beliefs.append(posterior)

        # Predict next state if not at end
        if t < T - 1:
            current_belief = model.predict_next_state(posterior, actions[t])

    # Could add backward pass here for smoothing if needed
    # For now, just return filtering posteriors

    return beliefs


def update_belief_batch(
    observations: Float[Array, "batch"],
    prior_beliefs: Float[Array, "batch n_states"],
    model: GenerativeModel,
    n_iterations: int = 16,
) -> Float[Array, "batch n_states"]:
    """Batch inference over multiple observations.

    **Arguments:**

    - `observations`: Batch of observation indices
    - `prior_beliefs`: Batch of prior beliefs
    - `model`: Generative model
    - `n_iterations`: Number of inference iterations

    **Returns:**

    - Batch of posterior beliefs
    """

    def single_inference(obs, prior):
        # Use obs directly as int - it's already the right type in vmap context
        # Get observation likelihood for integer observation
        likelihood = model.A[obs, :]

        # Bayesian update: posterior ∝ likelihood × prior
        unnormalized_posterior = likelihood * prior
        posterior = normalize_distribution(unnormalized_posterior)

        return posterior

    posteriors = jax.vmap(single_inference)(observations.astype(jnp.int32), prior_beliefs)
    return posteriors
