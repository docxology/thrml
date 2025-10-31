"""Pytest configuration and fixtures."""

import jax
import jax.numpy as jnp
import pytest

# Configure JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture
def rng_key():
    """Provide a JAX random key for tests."""
    return jax.random.key(42)


@pytest.fixture
def simple_generative_model():
    """Create a simple generative model for testing."""
    from active_inference.core.generative_model import GenerativeModel

    n_states = 4
    n_observations = 4
    n_actions = 2

    # Simple identity observation model
    A = jnp.eye(n_observations, n_states)

    # Deterministic transitions
    B = jnp.zeros((n_states, n_states, n_actions))
    # Action 0: move to next state
    for s in range(n_states - 1):
        B = B.at[s + 1, s, 0].set(1.0)
    B = B.at[n_states - 1, n_states - 1, 0].set(1.0)  # Stay at last state

    # Action 1: stay in current state
    for s in range(n_states):
        B = B.at[s, s, 1].set(1.0)

    # Prefer last observation
    C = jnp.array([0.0, 0.0, 0.0, 2.0])

    # Uniform prior
    D = jnp.ones(n_states) / n_states

    return GenerativeModel(
        n_states=n_states,
        n_observations=n_observations,
        n_actions=n_actions,
        A=A,
        B=B,
        C=C,
        D=D,
    )


@pytest.fixture
def grid_world_config():
    """Create a grid world configuration for testing."""
    from active_inference.environments.grid_world import GridWorldConfig

    return GridWorldConfig(
        size=3,
        n_observations=5,
        observation_noise=0.1,
        goal_location=(2, 2),
        obstacle_locations=[(1, 1)],
    )


@pytest.fixture
def basic_agent(simple_generative_model):
    """Create a basic active inference agent for testing.

    Uses THRML-based inference with sampling parameters for real
    probabilistic computation on the agent's perception-action loop.
    """
    from active_inference.agents import ActiveInferenceAgent
    from active_inference.core.precision import Precision

    precision = Precision(
        sensory_precision=1.0,
        state_precision=1.0,
        action_precision=2.0,
    )

    # Agent now uses THRML sampling-based inference by default
    return ActiveInferenceAgent(
        model=simple_generative_model,
        precision=precision,
        planning_horizon=2,
        # THRML sampling parameters for real probabilistic inference
        n_samples=200,  # Number of samples for inference
        n_warmup=50,  # Warmup samples to reach equilibrium
        steps_per_sample=5,  # Gibbs steps between samples
    )
