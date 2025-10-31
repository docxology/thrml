"""Base active inference agent implementation."""

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ..core.free_energy import batch_expected_free_energy
from ..core.generative_model import GenerativeModel
from ..core.precision import Precision, PrecisionWeighting
from ..inference.thrml_inference import ThrmlInferenceEngine


@dataclass
class AgentState(eqx.Module):
    """Internal state of an active inference agent.

    **Attributes:**

    - `belief`: Current posterior belief over states
    - `observation_history`: History of observations
    - `action_history`: History of actions taken
    - `free_energy_history`: History of free energy values
    """

    belief: Float[Array, "n_states"]
    observation_history: list[int]
    action_history: list[int]
    free_energy_history: Float[Array, "history_length"]


class ActiveInferenceAgent(eqx.Module):
    """A discrete active inference agent.

    This agent performs perception (state inference), action selection
    (policy optimization), and learning (parameter updates) using the
    Free Energy Principle.

    **Attributes:**

    - `model`: Generative model of the environment
    - `precision`: Precision parameters for inference and action
    - `planning_horizon`: Number of steps to plan ahead
    - `thrml_engine`: THRML inference engine for GPU-accelerated sampling (always required)
    """

    model: GenerativeModel
    precision: Precision
    planning_horizon: int
    thrml_engine: ThrmlInferenceEngine

    def __init__(
        self,
        model: GenerativeModel,
        precision: Optional[Precision] = None,
        planning_horizon: int = 3,
        thrml_engine: Optional[ThrmlInferenceEngine] = None,
        n_samples: int = 200,
        n_warmup: int = 50,
        steps_per_sample: int = 5,
    ):
        """Initialize an active inference agent with THRML inference.

        **Arguments:**

        - `model`: Generative model
        - `precision`: Precision parameters (uses defaults if None)
        - `planning_horizon`: Steps to plan ahead
        - `thrml_engine`: THRML inference engine (created automatically if None)
        - `n_samples`: Number of THRML samples (used if thrml_engine is None)
        - `n_warmup`: Warmup samples (used if thrml_engine is None)
        - `steps_per_sample`: Steps per sample (used if thrml_engine is None)
        """
        self.model = model
        self.precision = precision if precision is not None else Precision()
        self.planning_horizon = planning_horizon

        # Create THRML engine if not provided
        if thrml_engine is None:
            from ..inference.thrml_inference import ThrmlInferenceEngine

            thrml_engine = ThrmlInferenceEngine(
                model=model,
                n_samples=n_samples,
                n_warmup=n_warmup,
                steps_per_sample=steps_per_sample,
            )
        self.thrml_engine = thrml_engine

    def perceive(
        self,
        observation: int,
        prior_belief: Float[Array, "n_states"],
        key: Key[Array, ""],
    ) -> tuple[Float[Array, "n_states"], Float[Array, ""]]:
        """Infer hidden states from observation using THRML sampling (perception).

        Always uses THRML sampling-based inference for GPU-accelerated, energy-efficient inference.

        **Arguments:**

        - `observation`: Observed index
        - `prior_belief`: Prior belief over states
        - `key`: JAX random key (required for THRML sampling)

        **Returns:**

        - Tuple of (posterior_belief, free_energy)
        """
        # Use THRML sampling-based inference
        # Update model's prior for THRML engine (needed for sampling)
        temp_model = eqx.tree_at(lambda m: m.D, self.model, prior_belief)
        temp_engine = eqx.tree_at(lambda e: e.model, self.thrml_engine, temp_model)

        # THRML inference using block Gibbs sampling
        posterior = temp_engine.infer_with_sampling(
            key=key,
            observation=observation,
        )

        # Compute free energy for compatibility
        from ..core.free_energy import variational_free_energy

        free_energy = variational_free_energy(observation, posterior, self.model)

        return posterior, free_energy

    def act(
        self,
        key: Key[Array, ""],
        state_belief: Float[Array, "n_states"],
    ) -> int:
        """Select action that minimizes expected free energy (action).

        **Arguments:**

        - `key`: JAX random key for stochastic action selection
        - `state_belief`: Current belief over states

        **Returns:**

        - Selected action index
        """
        # Calculate expected free energy for all actions
        efe_values = batch_expected_free_energy(
            state_belief,
            self.model,
            self.planning_horizon,
        )

        # Convert to action probabilities using softmax with precision
        # Note: EFE is a cost, so we negate it for softmax
        action_probs = PrecisionWeighting.softmax_with_precision(
            -efe_values,
            self.precision.action_precision,
        )

        # Sample action from distribution
        action = jax.random.choice(key, self.model.n_actions, p=action_probs)

        return int(action)

    def step(
        self,
        key: Key[Array, ""],
        observation: int,
        agent_state: AgentState,
    ) -> tuple[int, AgentState, Float[Array, ""]]:
        """Perform one perception-action cycle.

        **Arguments:**

        - `key`: JAX random key
        - `observation`: Current observation
        - `agent_state`: Current agent state

        **Returns:**

        - Tuple of (action, new_agent_state, free_energy)
        """
        # Split key for perception and action
        key_perceive, key_act = jax.random.split(key)

        # Perceive: infer hidden states
        posterior, free_energy = self.perceive(observation, agent_state.belief, key=key_perceive)

        # Act: select action
        action = self.act(key_act, posterior)

        # Predict next state given selected action
        predicted_next_belief = self.model.predict_next_state(posterior, action)

        # Update agent state
        new_observation_history = agent_state.observation_history + [observation]
        new_action_history = agent_state.action_history + [action]
        new_fe_history = jnp.concatenate([agent_state.free_energy_history, jnp.array([free_energy])])

        new_agent_state = AgentState(
            belief=predicted_next_belief,
            observation_history=new_observation_history,
            action_history=new_action_history,
            free_energy_history=new_fe_history,
        )

        return action, new_agent_state, free_energy

    def reset(self) -> AgentState:
        """Reset agent to initial state.

        **Returns:**

        - Fresh agent state with initial belief
        """
        return AgentState(
            belief=self.model.D.copy(),
            observation_history=[],
            action_history=[],
            free_energy_history=jnp.array([]),
        )

    def get_action_distribution(
        self,
        state_belief: Float[Array, "n_states"],
    ) -> Float[Array, "n_actions"]:
        """Get action probability distribution without sampling.

        **Arguments:**

        - `state_belief`: Current belief over states

        **Returns:**

        - Probability distribution over actions
        """
        efe_values = batch_expected_free_energy(
            state_belief,
            self.model,
            self.planning_horizon,
        )

        action_probs = PrecisionWeighting.softmax_with_precision(
            -efe_values,
            self.precision.action_precision,
        )

        return action_probs
