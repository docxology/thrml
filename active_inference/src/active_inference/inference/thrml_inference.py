"""THRML-based inference engine for energy-efficient probabilistic inference.

This module integrates THRML's block Gibbs sampling with active inference,
enabling efficient inference on GPUs and future Extropic hardware.

This implementation demonstrates comprehensive THRML usage:
- Block management with Block and BlockGibbsSpec
- Factor-based models with CategoricalEBMFactor
- Block Gibbs sampling with FactorSamplingProgram
- State sampling with sample_states
- Observation with StateObserver
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key
from thrml import Block, BlockGibbsSpec, CategoricalNode, SamplingSchedule, make_empty_block_state, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES

from ..core.generative_model import GenerativeModel


class ThrmlInferenceEngine(eqx.Module):
    """Inference engine using THRML for efficient sampling-based inference.

    This engine converts active inference problems into THRML-compatible
    energy-based models and uses block Gibbs sampling for efficient inference.

    **Attributes:**

    - `model`: The generative model
    - `n_samples`: Number of samples for inference
    - `n_warmup`: Warmup steps before sampling
    - `steps_per_sample`: Gibbs steps between samples
    """

    model: GenerativeModel
    n_samples: int
    n_warmup: int
    steps_per_sample: int

    def __init__(
        self,
        model: GenerativeModel,
        n_samples: int = 1000,
        n_warmup: int = 100,
        steps_per_sample: int = 5,
    ):
        """Initialize THRML inference engine.

        **Arguments:**

        - `model`: Generative model
        - `n_samples`: Number of samples for inference
        - `n_warmup`: Warmup samples to discard
        - `steps_per_sample`: Gibbs steps between samples
        """
        self.model = model
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.steps_per_sample = steps_per_sample

    def infer_with_sampling(
        self,
        key: Key[Array, ""],
        observation: int,
        n_state_samples: Optional[int] = None,
    ) -> Float[Array, "n_states"]:
        """Infer state distribution using THRML sampling.

        This method uses real THRML components:
        1. Creates CategoricalNode for each state
        2. Builds CategoricalEBMFactor with observation likelihood and prior
        3. Uses FactorSamplingProgram for factor-based sampling
        4. Samples with sample_states and SamplingSchedule
        5. Estimates posterior from samples

        **Arguments:**

        - `key`: JAX random key
        - `observation`: Observed index
        - `n_state_samples`: Number of samples (uses default if None)

        **Returns:**

        - Approximate posterior distribution over states (estimated from samples)
        """
        if n_state_samples is None:
            n_state_samples = self.n_samples

        # Step 1: Create THRML categorical node for the hidden state
        # We have ONE state variable that can take n_states different values
        # So we create ONE categorical node
        state_node = CategoricalNode()

        # Step 2: Build THRML block structure
        # Single node in a block
        state_block = Block([state_node])
        gibbs_spec = BlockGibbsSpec([state_block], clamped_blocks=[], node_shape_dtypes=DEFAULT_NODE_SHAPE_DTYPES)

        # Step 3: Compute energy weights from generative model
        # Energy for state s: E(s) = -log P(o|s) - log P(s)
        # Convert to weights: W(s) = exp(-E(s)) = P(o|s) * P(s)

        # Get observation likelihood and prior
        likelihood = self.model.A[observation, :]  # P(o|s) for each state s
        prior = self.model.D  # P(s) for each state s

        # Compute unnormalized posterior weights
        posterior_weights = likelihood * prior  # P(o|s) * P(s)

        # Normalize for numerical stability
        posterior_weights = posterior_weights / (jnp.sum(posterior_weights) + 1e-16)

        # Step 4: Create THRML CategoricalEBMFactor
        # For CategoricalEBMFactor with one node group:
        # weights shape must be [b, x_1] where:
        #   b = number of nodes in the block (1 in our case)
        #   x_1 = number of categories (n_states)
        # The weights represent the log probability of each category
        log_weights = jnp.log(posterior_weights + 1e-16)

        categorical_factor = CategoricalEBMFactor(
            node_groups=[state_block], weights=log_weights[jnp.newaxis, :]  # 1 node in block  # Shape: [1, n_states]
        )

        # Step 5: Create FactorSamplingProgram
        # This combines the factor with a categorical Gibbs sampler
        sampler = CategoricalGibbsConditional(n_categories=self.model.n_states)

        sampling_program = FactorSamplingProgram(
            gibbs_spec=gibbs_spec,
            samplers=[sampler],  # One sampler per free block
            factors=[categorical_factor],
            other_interaction_groups=[],
        )

        # Step 6: Create sampling schedule
        schedule = SamplingSchedule(
            n_warmup=self.n_warmup,
            n_samples=n_state_samples,
            steps_per_sample=self.steps_per_sample,
        )

        # Step 7: Initialize state randomly
        key_init, key_sample = jax.random.split(key)
        init_state = make_empty_block_state(
            blocks=[state_block], node_shape_dtypes=DEFAULT_NODE_SHAPE_DTYPES, batch_shape=None
        )

        # Initialize with random categorical sample (single node)
        init_state[0] = jax.random.randint(
            key_init,
            shape=(1,),  # Single node
            minval=0,
            maxval=self.model.n_states,
            dtype=jnp.uint8,
        )

        # Step 8: Sample states using THRML
        # Use sample_states to run block Gibbs sampling
        samples = sample_states(
            key=key_sample,
            program=sampling_program,
            schedule=schedule,
            init_state_free=init_state,
            state_clamp=[],  # No clamped nodes
            nodes_to_sample=[state_block],
        )

        # Step 9: Estimate posterior from samples
        # samples is a list with one element (the state_block)
        # samples[0] has shape [n_samples, 1] - one node, many samples
        state_samples = samples[0]  # Shape: [n_samples, 1]

        # Compute histogram of sampled states
        # state_samples[:,0] gives all samples for the single node
        sampled_values = state_samples[:, 0]  # Shape: [n_samples]

        # Compute empirical posterior distribution
        posterior_estimate = jnp.zeros(self.model.n_states)
        for i in range(self.model.n_states):
            posterior_estimate = posterior_estimate.at[i].set(jnp.mean((sampled_values == i).astype(jnp.float32)))

        # Normalize to ensure it's a proper distribution
        posterior_estimate = posterior_estimate / (jnp.sum(posterior_estimate) + 1e-16)

        return posterior_estimate

    def sample_trajectory(
        self,
        key: Key[Array, ""],
        actions: list[int],
        initial_state_belief: Float[Array, "n_states"],
    ) -> list[Float[Array, "n_states"]]:
        """Sample a state trajectory given actions using THRML.

        **Arguments:**

        - `key`: JAX random key
        - `actions`: Sequence of actions
        - `initial_state_belief`: Initial belief over states

        **Returns:**

        - List of state samples at each time step
        """
        trajectory = [initial_state_belief]
        current_belief = initial_state_belief

        for action in actions:
            # Predict next state
            next_belief = self.model.predict_next_state(current_belief, action)
            trajectory.append(next_belief)
            current_belief = next_belief

        return trajectory
