"""Tests for inference engines."""

import jax
import jax.numpy as jnp

from active_inference.inference.state_inference import infer_states, update_belief_batch, variational_message_passing
from active_inference.inference.thrml_inference import ThrmlInferenceEngine


class TestStateInference:
    """Test state inference functionality."""

    def test_infer_states_basic(self, simple_generative_model, rng_key):
        """Test basic state inference."""
        model = simple_generative_model

        observation = 0
        prior_belief = model.D

        posterior, fe = infer_states(observation, prior_belief, model)

        # Check output shapes and properties
        assert posterior.shape == (model.n_states,)
        assert jnp.allclose(jnp.sum(posterior), 1.0)
        assert jnp.all(posterior >= 0)
        assert jnp.isfinite(fe)

    def test_infer_states_convergence(self, simple_generative_model):
        """Test that inference converges."""
        model = simple_generative_model

        observation = 0
        prior_belief = model.D

        # Run with different iteration counts
        posterior_10, fe_10 = infer_states(observation, prior_belief, model, n_iterations=10)
        posterior_100, fe_100 = infer_states(observation, prior_belief, model, n_iterations=100)

        # More iterations should give similar or better (lower) free energy
        assert fe_100 <= fe_10 + 1e-3  # Small tolerance for numerical issues

    def test_infer_states_with_certain_observation(self, simple_generative_model):
        """Test inference with certain observation."""
        model = simple_generative_model

        # Observation 0 should strongly suggest state 0
        observation = 0
        prior_belief = jnp.ones(model.n_states) / model.n_states

        posterior, _ = infer_states(observation, prior_belief, model)

        # State 0 should have highest posterior
        assert jnp.argmax(posterior) == 0

    def test_variational_message_passing(self, simple_generative_model):
        """Test message passing over a sequence."""
        model = simple_generative_model

        observations = [0, 1, 2]
        actions = [0, 0]  # One fewer than observations

        beliefs = variational_message_passing(observations, actions, model)

        # Check we get one belief per observation
        assert len(beliefs) == len(observations)

        # Each belief should be normalized
        for belief in beliefs:
            assert jnp.allclose(jnp.sum(belief), 1.0)
            assert jnp.all(belief >= 0)

    def test_update_belief_batch(self, simple_generative_model, rng_key):
        """Test batch belief updates."""
        model = simple_generative_model

        batch_size = 5
        observations = jax.random.randint(rng_key, (batch_size,), 0, model.n_observations)
        prior_beliefs = jnp.tile(model.D, (batch_size, 1))

        posteriors = update_belief_batch(observations, prior_beliefs, model)

        # Check output shape
        assert posteriors.shape == (batch_size, model.n_states)

        # Check each posterior is normalized
        for i in range(batch_size):
            assert jnp.allclose(jnp.sum(posteriors[i]), 1.0)


class TestThrmlInference:
    """Test THRML-based inference engine."""

    def test_thrml_engine_initialization(self, simple_generative_model):
        """Test THRML engine initialization."""
        engine = ThrmlInferenceEngine(
            model=simple_generative_model,
            n_samples=100,
            n_warmup=10,
            steps_per_sample=2,
        )

        assert engine.model == simple_generative_model
        assert engine.n_samples == 100
        assert engine.n_warmup == 10
        assert engine.steps_per_sample == 2

    def test_thrml_infer_with_sampling(self, simple_generative_model, rng_key):
        """Test sampling-based inference with THRML."""
        engine = ThrmlInferenceEngine(
            model=simple_generative_model,
            n_samples=50,
            n_warmup=10,
        )

        observation = 0
        posterior = engine.infer_with_sampling(rng_key, observation, n_state_samples=50)

        # Check output properties
        assert posterior.shape == (simple_generative_model.n_states,)
        assert jnp.allclose(jnp.sum(posterior), 1.0)
        assert jnp.all(posterior >= 0)

    def test_thrml_sample_trajectory(self, simple_generative_model, rng_key):
        """Test trajectory sampling."""
        engine = ThrmlInferenceEngine(model=simple_generative_model)

        actions = [0, 0, 1]
        initial_belief = simple_generative_model.D

        trajectory = engine.sample_trajectory(rng_key, actions, initial_belief)

        # Should have one belief per action plus initial
        assert len(trajectory) == len(actions) + 1

        # Each should be a proper distribution
        for belief in trajectory:
            assert jnp.allclose(jnp.sum(belief), 1.0)
            assert jnp.all(belief >= 0)


class TestInferenceProperties:
    """Property-based tests for inference."""

    def test_inference_reduces_uncertainty(self, simple_generative_model):
        """Test that inference reduces uncertainty about states."""
        model = simple_generative_model

        # Start with maximum uncertainty (uniform prior)
        prior = jnp.ones(model.n_states) / model.n_states
        prior_entropy = -jnp.sum(prior * jnp.log(prior + 1e-16))

        # Observe something
        observation = 0
        posterior, _ = infer_states(observation, prior, model)

        # Calculate posterior entropy
        posterior_entropy = -jnp.sum(posterior * jnp.log(posterior + 1e-16))

        # Posterior should be more certain (lower entropy)
        # Note: this may not always hold with noisy observations
        # but should hold for our test model with identity A matrix
        assert posterior_entropy < prior_entropy + 1e-3

    def test_inference_respects_likelihood(self, simple_generative_model):
        """Test that inference respects observation likelihood."""
        model = simple_generative_model

        prior = jnp.ones(model.n_states) / model.n_states

        # Test each possible observation
        for obs in range(model.n_observations):
            posterior, _ = infer_states(obs, prior, model)

            # Get likelihood for this observation
            likelihood = model.get_observation_likelihood(obs)

            # Posterior should be proportional to likelihood * prior
            expected_posterior = likelihood * prior
            expected_posterior = expected_posterior / jnp.sum(expected_posterior)

            assert jnp.allclose(posterior, expected_posterior, atol=1e-3)
