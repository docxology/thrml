"""Tests for active inference agents."""

import jax
import jax.numpy as jnp

from active_inference.agents import ActiveInferenceAgent, AgentState, plan_action, plan_with_tree_search


class TestAgentState:
    """Test agent state management."""

    def test_agent_state_initialization(self):
        """Test agent state initialization."""
        initial_belief = jnp.array([0.25, 0.25, 0.25, 0.25])
        state = AgentState(
            belief=initial_belief,
            observation_history=[],
            action_history=[],
            free_energy_history=jnp.array([]),
        )

        assert jnp.array_equal(state.belief, initial_belief)
        assert state.observation_history == []
        assert state.action_history == []
        assert len(state.free_energy_history) == 0


class TestActiveInferenceAgent:
    """Test active inference agent."""

    def test_agent_initialization(self, basic_agent):
        """Test agent initialization with THRML inference engine."""
        agent = basic_agent

        assert agent.model is not None
        assert agent.precision is not None
        assert agent.planning_horizon == 2
        # Agent now uses THRML-based inference
        assert agent.thrml_engine is not None
        assert agent.thrml_engine.n_samples == 200
        assert agent.thrml_engine.n_warmup == 50
        assert agent.thrml_engine.steps_per_sample == 5

    def test_agent_perceive(self, basic_agent, rng_key):
        """Test perception (state inference) using THRML sampling."""
        agent = basic_agent

        observation = 0
        prior_belief = agent.model.D

        # THRML-based perceive requires a random key for sampling
        posterior, fe = agent.perceive(observation, prior_belief, key=rng_key)

        assert posterior.shape == (agent.model.n_states,)
        assert jnp.allclose(jnp.sum(posterior), 1.0)
        assert jnp.isfinite(fe)

    def test_agent_act(self, basic_agent, rng_key):
        """Test action selection."""
        agent = basic_agent

        state_belief = jnp.array([0.7, 0.2, 0.1, 0.0])
        action = agent.act(rng_key, state_belief)

        assert isinstance(action, int)
        assert 0 <= action < agent.model.n_actions

    def test_agent_step(self, basic_agent, rng_key):
        """Test full perception-action cycle."""
        agent = basic_agent
        agent_state = agent.reset()

        observation = 0
        action, new_state, fe = agent.step(rng_key, observation, agent_state)

        # Check action
        assert isinstance(action, int)
        assert 0 <= action < agent.model.n_actions

        # Check new state
        assert new_state.belief.shape == (agent.model.n_states,)
        assert jnp.allclose(jnp.sum(new_state.belief), 1.0)

        # Check history updated
        assert len(new_state.observation_history) == 1
        assert len(new_state.action_history) == 1
        assert len(new_state.free_energy_history) == 1

    def test_agent_reset(self, basic_agent):
        """Test agent reset."""
        agent = basic_agent
        initial_state = agent.reset()

        assert jnp.array_equal(initial_state.belief, agent.model.D)
        assert initial_state.observation_history == []
        assert initial_state.action_history == []

    def test_agent_get_action_distribution(self, basic_agent):
        """Test getting action distribution."""
        agent = basic_agent

        state_belief = jnp.array([0.7, 0.2, 0.1, 0.0])
        action_probs = agent.get_action_distribution(state_belief)

        assert action_probs.shape == (agent.model.n_actions,)
        assert jnp.allclose(jnp.sum(action_probs), 1.0)
        assert jnp.all(action_probs >= 0)

    def test_agent_seeks_goal(self, simple_generative_model, rng_key):
        """Test that agent seeks goal states."""
        from active_inference.core.precision import Precision

        # Create agent with high action precision (greedy)
        precision = Precision(action_precision=10.0)
        agent = ActiveInferenceAgent(
            model=simple_generative_model,
            precision=precision,
            planning_horizon=3,
        )

        # Start at beginning (far from goal)
        state_belief = jnp.array([1.0, 0.0, 0.0, 0.0])
        action_probs = agent.get_action_distribution(state_belief)

        # Should prefer action 0 (moves toward goal at state 3)
        assert jnp.argmax(action_probs) == 0


class TestPlanning:
    """Test planning and policy optimization."""

    def test_plan_action_greedy(self, simple_generative_model):
        """Test greedy action planning."""
        model = simple_generative_model

        # Start at beginning, goal is at end
        state_belief = jnp.array([1.0, 0.0, 0.0, 0.0])

        best_action = plan_action(state_belief, model, horizon=1)

        # Should choose to move forward (action 0)
        assert best_action == 0

    def test_plan_with_tree_search(self, simple_generative_model):
        """Test tree search planning."""
        model = simple_generative_model

        state_belief = jnp.array([1.0, 0.0, 0.0, 0.0])

        action_sequence, total_efe = plan_with_tree_search(state_belief, model, horizon=2)

        # Should get a sequence of 2 actions
        assert len(action_sequence) == 2

        # Should prefer moving forward
        assert action_sequence[0] == 0

        # EFE should be finite
        assert jnp.isfinite(total_efe)

    def test_tree_search_with_branching(self, simple_generative_model):
        """Test tree search with limited branching."""
        model = simple_generative_model

        state_belief = jnp.array([1.0, 0.0, 0.0, 0.0])

        # Limit branching factor
        action_sequence, total_efe = plan_with_tree_search(state_belief, model, horizon=2, branching_factor=1)

        assert len(action_sequence) == 2
        assert jnp.isfinite(total_efe)


class TestAgentBehavior:
    """Integration tests for agent behavior."""

    def test_agent_multi_step_episode(self, basic_agent, rng_key):
        """Test agent over multiple steps."""
        agent = basic_agent
        agent_state = agent.reset()

        n_steps = 5
        observations = [0, 1, 1, 2, 2]

        keys = jax.random.split(rng_key, n_steps)

        for i, obs in enumerate(observations):
            action, agent_state, fe = agent.step(keys[i], obs, agent_state)

            # Verify state consistency
            assert len(agent_state.observation_history) == i + 1
            assert len(agent_state.action_history) == i + 1
            assert agent_state.observation_history[i] == obs

    def test_agent_exploration_vs_exploitation(self, simple_generative_model, rng_key):
        """Test that precision controls exploration vs exploitation."""
        from active_inference.core.precision import Precision

        state_belief = jnp.array([0.25, 0.25, 0.25, 0.25])  # Uncertain

        # High precision (exploitation)
        high_precision = Precision(action_precision=10.0)
        exploitative_agent = ActiveInferenceAgent(
            model=simple_generative_model,
            precision=high_precision,
        )

        # Low precision (exploration)
        low_precision = Precision(action_precision=0.5)
        exploratory_agent = ActiveInferenceAgent(
            model=simple_generative_model,
            precision=low_precision,
        )

        exploit_probs = exploitative_agent.get_action_distribution(state_belief)
        explore_probs = exploratory_agent.get_action_distribution(state_belief)

        # Exploitative should be more peaked (higher max)
        assert jnp.max(exploit_probs) > jnp.max(explore_probs)

        # Exploratory should be more uniform (higher entropy)
        exploit_entropy = -jnp.sum(exploit_probs * jnp.log(exploit_probs + 1e-16))
        explore_entropy = -jnp.sum(explore_probs * jnp.log(explore_probs + 1e-16))

        assert explore_entropy > exploit_entropy
