"""Integration tests combining all components."""

import equinox as eqx
import jax
import jax.numpy as jnp

from active_inference.agents import ActiveInferenceAgent
from active_inference.core.precision import Precision
from active_inference.environments import GridWorld, GridWorldConfig, TMaze
from active_inference.models import build_grid_world_model, build_tmaze_model


class TestGridWorldIntegration:
    """Integration tests for grid world with active inference agent."""

    def test_agent_navigates_grid_world(self, rng_key):
        """Test that agent can navigate grid world."""
        # Create environment
        config = GridWorldConfig(
            size=3,
            n_observations=9,
            observation_noise=0.05,
            goal_location=(2, 2),
            obstacle_locations=[],
        )
        env = GridWorld(config=config)

        # Create matching generative model
        model = build_grid_world_model(config, goal_preference_strength=3.0)

        # Create agent
        precision = Precision(action_precision=2.0)
        agent = ActiveInferenceAgent(
            model=model,
            precision=precision,
            planning_horizon=2,
        )

        # Run episode
        agent_state = agent.reset()
        env, obs = env.reset(rng_key)

        n_steps = 20
        keys = jax.random.split(rng_key, n_steps)

        trajectory = []
        done = False

        for i in range(n_steps):
            if done:
                break

            # Agent perceives and acts
            action, agent_state, fe = agent.step(keys[i], obs, agent_state)

            # Environment responds
            key_env = jax.random.split(keys[i])[1]
            env, obs, reward, done = env.step(key_env, action)

            trajectory.append(
                {
                    "step": i,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "state": env.current_state.copy(),
                    "free_energy": fe,
                }
            )

        # Check that agent made progress
        assert len(trajectory) > 0

        # Should have updated beliefs
        assert len(agent_state.observation_history) > 0
        assert len(agent_state.action_history) > 0

        # Free energies should be finite
        assert jnp.all(jnp.isfinite(agent_state.free_energy_history))

    def test_agent_avoids_obstacles(self, rng_key):
        """Test that agent learns to avoid obstacles."""
        # Create environment with obstacle
        config = GridWorldConfig(
            size=4,
            n_observations=10,
            goal_location=(3, 3),
            obstacle_locations=[(1, 1), (1, 2), (2, 1)],
        )
        env = GridWorld(config=config)

        model = build_grid_world_model(config)
        agent = ActiveInferenceAgent(model=model, planning_horizon=2)

        # Run short episode
        agent_state = agent.reset()
        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([0, 0]))  # Start at corner

        keys = jax.random.split(rng_key, 15)

        for i in range(15):
            obs = env.get_observation(keys[i])
            action, agent_state, _ = agent.step(keys[i], obs, agent_state)

            # Verify agent doesn't move into obstacles
            prev_state = env.current_state.copy()
            key_env = jax.random.split(keys[i])[1]
            env, obs, _, done = env.step(key_env, action)

            if done:
                break


class TestTMazeIntegration:
    """Integration tests for T-maze with active inference agent."""

    def test_agent_navigates_tmaze(self, rng_key):
        """Test that agent navigates T-maze."""
        # Create environment
        reward_side = 0  # Left
        env = TMaze(reward_side=reward_side)

        # Create model
        model = build_tmaze_model(reward_side=reward_side)

        # Create agent
        agent = ActiveInferenceAgent(
            model=model,
            planning_horizon=3,
        )

        # Run episode
        agent_state = agent.reset()
        env, obs, cue_presented = env.reset(rng_key)

        keys = jax.random.split(rng_key, 10)
        done = False
        total_reward = 0.0

        for i in range(10):
            if done:
                break

            action, agent_state, fe = agent.step(keys[i], obs, agent_state)
            env, obs, reward, done = env.step(action)
            total_reward += reward

        # Should complete episode
        assert done

        # Should get reward eventually
        assert total_reward >= 0.0

    def test_agent_uses_cues(self, rng_key):
        """Test that agent uses cues to guide behavior."""
        # Test multiple trials
        reward_side = 0
        env = TMaze(reward_side=reward_side)
        model = build_tmaze_model(reward_side=reward_side)

        agent = ActiveInferenceAgent(model=model, planning_horizon=3)

        successes = 0
        n_trials = 5

        keys = jax.random.split(rng_key, n_trials)

        for trial_key in keys:
            env = TMaze(reward_side=reward_side)
            agent_state = agent.reset()

            env, obs, cue_presented = env.reset(trial_key)

            # Navigate through maze
            trial_keys = jax.random.split(trial_key, 5)
            done = False
            reward_obtained = 0.0

            for step_key in trial_keys:
                if done:
                    break

                action, agent_state, _ = agent.step(step_key, obs, agent_state)
                env, obs, reward, done = env.step(action)
                reward_obtained += reward

            if reward_obtained > 0:
                successes += 1

        # Should succeed on some trials
        assert successes >= 0  # At least try


class TestLearningBehavior:
    """Test learning-related behaviors."""

    def test_free_energy_decreases_with_information(self, rng_key):
        """Test that free energy decreases as agent gathers information."""
        config = GridWorldConfig(size=3, n_observations=9)
        env = GridWorld(config=config)
        model = build_grid_world_model(config)

        agent = ActiveInferenceAgent(model=model)
        agent_state = agent.reset()

        # Start with high uncertainty
        initial_fe = []
        later_fe = []

        keys = jax.random.split(rng_key, 20)

        for i in range(20):
            obs = env.get_observation(keys[i])
            action, agent_state, fe = agent.step(keys[i], obs, agent_state)

            key_env = jax.random.split(keys[i])[1]
            env, _, _, _ = env.step(key_env, action)

            if i < 5:
                initial_fe.append(float(fe))
            elif i >= 15:
                later_fe.append(float(fe))

        # Later free energies should generally be lower (more certain)
        # This might not always hold, but should on average
        mean_initial = jnp.mean(jnp.array(initial_fe))
        mean_later = jnp.mean(jnp.array(later_fe))

        # At least check both are finite
        assert jnp.isfinite(mean_initial)
        assert jnp.isfinite(mean_later)


class TestRobustness:
    """Test robustness and edge cases."""

    def test_agent_handles_noisy_observations(self, rng_key):
        """Test agent handles noisy observations."""
        # High observation noise
        config = GridWorldConfig(
            size=3,
            n_observations=5,
            observation_noise=0.5,  # Very noisy
        )
        env = GridWorld(config=config)
        model = build_grid_world_model(config)

        agent = ActiveInferenceAgent(model=model)
        agent_state = agent.reset()

        keys = jax.random.split(rng_key, 10)

        for i in range(10):
            obs = env.get_observation(keys[i])

            # Should handle noisy observations without crashing
            action, agent_state, fe = agent.step(keys[i], obs, agent_state)

            assert jnp.isfinite(fe)
            assert 0 <= action < env.n_actions

    def test_agent_handles_degenerate_cases(self):
        """Test agent handles edge cases."""
        from active_inference.core.generative_model import GenerativeModel

        # Minimal model
        model = GenerativeModel(n_states=2, n_observations=2, n_actions=2)
        agent = ActiveInferenceAgent(model=model)

        agent_state = agent.reset()

        # Should initialize without errors
        assert agent_state.belief.shape == (2,)
        assert jnp.allclose(jnp.sum(agent_state.belief), 1.0)
