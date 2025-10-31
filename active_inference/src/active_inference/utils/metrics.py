"""Metrics for evaluating active inference performance."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def calculate_kl_divergence(
    p: Float[Array, "n"],
    q: Float[Array, "n"],
) -> Float[Array, ""]:
    """Calculate KL divergence KL[P || Q].

    **Arguments:**

    - `p`: First distribution
    - `q`: Second distribution

    **Returns:**

    - KL divergence value
    """
    p = p + 1e-16
    q = q + 1e-16
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)

    kl = jnp.sum(p * (jnp.log(p) - jnp.log(q)))
    return kl


def calculate_prediction_accuracy(
    predicted_observations: list[Float[Array, "n_obs"]],
    actual_observations: list[int],
) -> float:
    """Calculate prediction accuracy.

    **Arguments:**

    - `predicted_observations`: List of predicted observation distributions
    - `actual_observations`: List of actual observation indices

    **Returns:**

    - Prediction accuracy (0 to 1)
    """
    correct = 0
    total = len(actual_observations)

    for pred_dist, actual_obs in zip(predicted_observations, actual_observations):
        predicted_obs = int(jnp.argmax(pred_dist))
        if predicted_obs == actual_obs:
            correct += 1

    return correct / total if total > 0 else 0.0


def calculate_policy_entropy(
    action_probs: Float[Array, "n_actions"],
) -> Float[Array, ""]:
    """Calculate entropy of action distribution.

    High entropy indicates exploration, low entropy indicates exploitation.

    **Arguments:**

    - `action_probs`: Action probability distribution

    **Returns:**

    - Entropy value
    """
    probs = action_probs + 1e-16
    probs = probs / jnp.sum(probs)
    entropy = -jnp.sum(probs * jnp.log(probs))
    return entropy


def calculate_expected_utility(
    outcomes: Float[Array, "n_outcomes"],
    utilities: Float[Array, "n_outcomes"],
) -> Float[Array, ""]:
    """Calculate expected utility.

    **Arguments:**

    - `outcomes`: Probability of each outcome
    - `utilities`: Utility of each outcome

    **Returns:**

    - Expected utility
    """
    return jnp.sum(outcomes * utilities)
