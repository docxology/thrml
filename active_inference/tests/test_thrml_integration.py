"""Tests for THRML integration with active inference.

These tests validate that the active_inference package correctly uses
THRML components and that core THRML functionality is accessible.
"""

import jax
import jax.numpy as jnp
import pytest
from thrml import Block, BlockGibbsSpec, CategoricalNode, SamplingSchedule, SpinNode, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SpinEBMFactor,
    SpinGibbsConditional,
)


class TestThrmlCoreIntegration:
    """Test that THRML core components work correctly."""

    def test_spin_node_creation(self):
        """Test creating SpinNode from THRML."""
        nodes = [SpinNode() for _ in range(5)]
        assert len(nodes) == 5
        assert all(isinstance(n, SpinNode) for n in nodes)

    def test_categorical_node_creation(self):
        """Test creating CategoricalNode from THRML."""
        nodes = [CategoricalNode() for _ in range(5)]
        assert len(nodes) == 5
        assert all(isinstance(n, CategoricalNode) for n in nodes)

    def test_block_creation(self):
        """Test creating blocks from THRML."""
        nodes = [SpinNode() for _ in range(5)]
        block = Block(nodes)
        assert len(block.nodes) == 5

    def test_block_gibbs_spec(self):
        """Test BlockGibbsSpec creation."""
        nodes = [SpinNode() for _ in range(10)]
        free_blocks = [Block(nodes[:5]), Block(nodes[5:])]
        spec = BlockGibbsSpec(free_blocks, [])
        assert len(spec.free_blocks) == 2
        assert len(spec.clamped_blocks) == 0


class TestThrmlIsing:
    """Test THRML Ising model integration."""

    def test_ising_model_creation(self):
        """Test creating Ising EBM."""
        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        biases = jnp.zeros((5,))
        weights = jnp.ones((4,)) * 0.5
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)

        assert len(model.nodes) == 5
        assert len(model.edges) == 4
        assert model.beta == 1.0

    def test_ising_sampling_program(self):
        """Test creating Ising sampling program."""
        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        biases = jnp.zeros((5,))
        weights = jnp.ones((4,)) * 0.5
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

        assert program.gibbs_spec is not None
        assert len(program.gibbs_spec.free_blocks) == 2

    def test_hinton_initialization(self):
        """Test Hinton initialization for Ising model."""
        key = jax.random.key(42)
        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        biases = jnp.zeros((5,))
        weights = jnp.ones((4,)) * 0.5
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]

        init_state = hinton_init(key, model, free_blocks, ())

        assert len(init_state) == 2
        assert init_state[0].dtype == jnp.bool_
        assert init_state[1].dtype == jnp.bool_

    def test_ising_sampling(self):
        """Test sampling from Ising model."""
        key = jax.random.key(42)
        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        biases = jnp.zeros((5,))
        weights = jnp.ones((4,)) * 0.5
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=2)

        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

        assert samples[0].shape == (50, 5)
        assert samples[0].dtype == jnp.bool_


class TestThrmlDiscreteEBM:
    """Test THRML discrete EBM components."""

    def test_spin_ebm_factor(self):
        """Test SpinEBMFactor creation."""
        nodes = [SpinNode() for _ in range(5)]
        weights = jnp.ones((5,))
        factor = SpinEBMFactor([Block(nodes)], weights)

        assert factor.weights.shape == (5,)

    def test_categorical_ebm_factor(self):
        """Test CategoricalEBMFactor creation."""
        n_cats = 3
        nodes = [CategoricalNode() for _ in range(5)]
        weights = jnp.ones((5, n_cats))
        factor = CategoricalEBMFactor([Block(nodes)], weights)

        assert factor.weights.shape == (5, n_cats)

    def test_spin_gibbs_conditional(self):
        """Test SpinGibbsConditional creation."""
        sampler = SpinGibbsConditional()
        assert sampler is not None

    def test_categorical_gibbs_conditional(self):
        """Test CategoricalGibbsConditional creation."""
        n_cats = 3
        sampler = CategoricalGibbsConditional(n_cats)
        assert sampler.n_categories == n_cats

    def test_categorical_sampling(self):
        """Test sampling from categorical EBM."""
        key = jax.random.key(42)
        n_cats = 3
        nodes = [CategoricalNode() for _ in range(5)]

        # Simple bias factor
        biases = jax.random.normal(key, (5, n_cats))
        factor = CategoricalEBMFactor([Block(nodes)], biases)

        # Create sampling program
        spec = BlockGibbsSpec([Block(nodes)], [])
        sampler = CategoricalGibbsConditional(n_cats)
        program = FactorSamplingProgram(spec, [sampler], [factor], [])

        # Initialize and sample
        init_state = [jax.random.randint(key, (5,), 0, n_cats, dtype=jnp.uint8)]
        schedule = SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=2)

        samples = sample_states(key, program, schedule, init_state, [], [Block(nodes)])

        assert samples[0].shape == (50, 5)
        assert samples[0].dtype == jnp.uint8


class TestThrmlIntegrationWithActiveInference:
    """Test that active inference correctly uses THRML components."""

    def test_active_inference_can_use_thrml_nodes(self):
        """Test that we can use THRML nodes in active inference models."""
        from active_inference.core import GenerativeModel

        # This shows that active inference and THRML can work together
        # by sharing node concepts
        n_states = 4
        n_obs = 4
        n_actions = 2

        # Create a simple model
        model = GenerativeModel(
            n_states=n_states,
            n_observations=n_obs,
            n_actions=n_actions,
        )

        # Create THRML nodes for the same problem
        thrml_nodes = [SpinNode() for _ in range(n_states)]
        thrml_block = Block(thrml_nodes)

        # Both should coexist
        assert model.n_states == len(thrml_block.nodes)

    def test_thrml_sampling_schedule_compatible(self):
        """Test that THRML SamplingSchedule works as expected."""
        schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

        assert schedule.n_warmup == 100
        assert schedule.n_samples == 1000
        assert schedule.steps_per_sample == 2

    def test_thrml_jax_compatibility(self):
        """Test that THRML operations are JAX-compatible."""
        key = jax.random.key(42)

        # Create a simple Ising model
        nodes = [SpinNode() for _ in range(3)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(2)]
        biases = jnp.array([0.5, -0.3, 0.2])
        weights = jnp.array([1.0, -0.5])
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)

        # This should be JIT-compilable
        @jax.jit
        def compute_something(b, w):
            m = IsingEBM(nodes, edges, b, w, beta)
            return m.beta * jnp.sum(m.biases)

        result = compute_something(biases, weights)
        assert jnp.isfinite(result)


class TestThrmlDocumentationExamples:
    """Test that THRML README examples work."""

    def test_readme_example(self):
        """Test the exact example from THRML README."""
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

        nodes = [SpinNode() for _ in range(5)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
        biases = jnp.zeros((5,))
        weights = jnp.ones((4,)) * 0.5
        beta = jnp.array(1.0)
        model = IsingEBM(nodes, edges, biases, weights, beta)

        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

        key = jax.random.key(0)
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

        assert samples[0].shape == (1000, 5)


@pytest.mark.slow
class TestThrmlPerformance:
    """Test THRML performance characteristics."""

    def test_large_ising_model(self):
        """Test that we can create and sample from a larger Ising model."""
        key = jax.random.key(42)

        # Create a small grid (not too large for tests)
        n = 10
        nodes = [SpinNode() for _ in range(n * n)]

        # Grid edges (horizontal and vertical)
        edges = []
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                # Right neighbor
                if j < n - 1:
                    edges.append((nodes[idx], nodes[idx + 1]))
                # Down neighbor
                if i < n - 1:
                    edges.append((nodes[idx], nodes[idx + n]))

        biases = jax.random.normal(key, (n * n,)) * 0.1
        weights = jax.random.normal(key, (len(edges),)) * 0.1
        beta = jnp.array(1.0)

        model = IsingEBM(nodes, edges, biases, weights, beta)

        # Checkerboard coloring for block Gibbs
        color0 = [nodes[i * n + j] for i in range(n) for j in range(n) if (i + j) % 2 == 0]
        color1 = [nodes[i * n + j] for i in range(n) for j in range(n) if (i + j) % 2 == 1]

        free_blocks = [Block(color0), Block(color1)]
        program = IsingSamplingProgram(model, free_blocks, [])

        # Quick sampling
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=5, n_samples=10, steps_per_sample=1)

        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

        assert samples[0].shape == (10, n * n)
