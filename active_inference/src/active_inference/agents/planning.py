"""Planning and policy optimization for active inference."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..core.free_energy import expected_free_energy
from ..core.generative_model import GenerativeModel


def plan_action(
    state_belief: Float[Array, "n_states"],
    model: GenerativeModel,
    horizon: int = 1,
) -> int:
    """Plan action by minimizing expected free energy.

    This is a simple greedy planner that looks ahead one step.

    **Arguments:**

    - `state_belief`: Current belief over states
    - `model`: Generative model
    - `horizon`: Planning horizon (currently only 1 supported)

    **Returns:**

    - Best action index
    """
    efe_values = jnp.zeros(model.n_actions)

    for action in range(model.n_actions):
        efe_values = efe_values.at[action].set(expected_free_energy(state_belief, action, model, horizon))

    # Select action with minimum expected free energy
    best_action = jnp.argmin(efe_values)

    return int(best_action)


def plan_with_tree_search(
    state_belief: Float[Array, "n_states"],
    model: GenerativeModel,
    horizon: int,
    branching_factor: int = None,
) -> tuple[list[int], Float[Array, ""]]:
    """Plan a sequence of actions using tree search.

    This performs a depth-first search through the policy tree,
    evaluating expected free energy for action sequences.

    **Arguments:**

    - `state_belief`: Current belief over states
    - `model`: Generative model
    - `horizon`: Number of steps to plan ahead
    - `branching_factor`: Max actions to consider at each step (all if None)

    **Returns:**

    - Tuple of (best_action_sequence, total_expected_free_energy)
    """
    if branching_factor is None:
        branching_factor = model.n_actions

    if horizon == 0:
        return [], jnp.array(0.0)

    # Evaluate all possible first actions
    best_sequence = []
    best_efe = jnp.inf

    for action in range(min(model.n_actions, branching_factor)):
        # Calculate EFE for this action
        current_efe = expected_free_energy(state_belief, action, model, 1)

        if horizon > 1:
            # Predict next state belief
            next_belief = model.predict_next_state(state_belief, action)

            # Recursively plan from next state
            future_sequence, future_efe = plan_with_tree_search(
                next_belief,
                model,
                horizon - 1,
                branching_factor,
            )

            total_efe = current_efe + future_efe
            full_sequence = [action] + future_sequence
        else:
            total_efe = current_efe
            full_sequence = [action]

        # Update best if this is better
        if total_efe < best_efe:
            best_efe = total_efe
            best_sequence = full_sequence

    return best_sequence, best_efe


def evaluate_policy(
    policy: list[int],
    initial_belief: Float[Array, "n_states"],
    model: GenerativeModel,
) -> Float[Array, ""]:
    """Evaluate total expected free energy for a policy.

    **Arguments:**

    - `policy`: Sequence of actions
    - `initial_belief`: Starting belief over states
    - `model`: Generative model

    **Returns:**

    - Total expected free energy for this policy
    """
    total_efe = jnp.array(0.0)
    current_belief = initial_belief

    for action in policy:
        efe = expected_free_energy(current_belief, action, model, 1)
        total_efe += efe
        current_belief = model.predict_next_state(current_belief, action)

    return total_efe
