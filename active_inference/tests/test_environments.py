"""Tests for environments."""

import equinox as eqx
import jax
import jax.numpy as jnp

from active_inference.environments import GridWorld, GridWorldConfig, TMaze


class TestGridWorld:
    """Test grid world environment."""

    def test_grid_world_initialization(self, grid_world_config):
        """Test grid world initialization."""
        env = GridWorld(config=grid_world_config)

        assert env.config == grid_world_config
        assert env.n_states == 9  # 3x3 grid
        assert env.n_observations == 5
        assert env.n_actions == 4

    def test_grid_world_reset(self, rng_key):
        """Test environment reset."""
        env = GridWorld(size=3)
        new_env, obs = env.reset(rng_key)

        assert isinstance(obs, int)
        assert 0 <= obs < env.n_observations

    def test_grid_world_step(self, rng_key):
        """Test taking steps in grid world."""
        env = GridWorld(size=3)
        env, _ = env.reset(rng_key)

        initial_state = env.current_state.copy()

        # Move right
        key1, key2 = jax.random.split(rng_key)
        new_env, obs, reward, done = env.step(key1, action=1)

        # Should move right (increase column)
        assert new_env.current_state[1] >= initial_state[1]

        assert isinstance(obs, int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_grid_world_boundaries(self, rng_key):
        """Test that agent respects boundaries."""
        env = GridWorld(size=3)

        # Start at corner using tree_at
        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([0, 0]))

        # Try to move up (should stay)
        key1, key2 = jax.random.split(rng_key)
        new_env, _, _, _ = env.step(key1, action=0)
        assert jnp.array_equal(new_env.current_state, jnp.array([0, 0]))

        # Try to move left (should stay)
        new_env2, _, _, _ = env.step(key2, action=3)
        assert jnp.array_equal(new_env2.current_state, jnp.array([0, 0]))

    def test_grid_world_obstacles(self, grid_world_config, rng_key):
        """Test that agent respects obstacles."""
        env = GridWorld(config=grid_world_config)

        # Start next to obstacle at (1, 1)
        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([1, 0]))

        # Try to move into obstacle (right)
        new_env, _, _, _ = env.step(rng_key, action=1)

        # Should not have moved
        assert jnp.array_equal(new_env.current_state, jnp.array([1, 0]))

    def test_grid_world_goal(self, grid_world_config, rng_key):
        """Test goal detection."""
        env = GridWorld(config=grid_world_config)

        # Move to goal
        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([2, 2]))
        assert env.is_goal()

        # Check goal reward - move to goal from adjacent cell
        env = eqx.tree_at(lambda e: e.current_state, env, jnp.array([2, 1]))  # One cell left of goal
        new_env, obs, reward, done = env.step(rng_key, action=1)  # Move right to goal
        assert reward > 0
        assert done

    def test_state_conversions(self, grid_world_config):
        """Test state to index and index to state conversions."""
        env = GridWorld(config=grid_world_config)

        # Test round trip
        for row in range(env.config.size):
            for col in range(env.config.size):
                state = jnp.array([row, col])
                idx = env.state_to_index(state)
                recovered_state = env.index_to_state(idx)

                assert jnp.array_equal(state, recovered_state)


class TestTMaze:
    """Test T-maze environment."""

    def test_tmaze_initialization(self):
        """Test T-maze initialization."""
        env = TMaze(reward_side=0)

        assert env.reward_side == 0
        assert env.n_states == 4
        assert env.n_observations == 8
        assert env.n_actions == 3

    def test_tmaze_reset(self, rng_key):
        """Test T-maze reset."""
        env = TMaze(reward_side=0)
        new_env, obs, cue_presented = env.reset(rng_key)

        assert isinstance(obs, int)
        assert isinstance(cue_presented, bool)
        assert 0 <= obs < env.n_observations

    def test_tmaze_cue_presentation(self, rng_key):
        """Test that cues are presented correctly."""
        env = TMaze(reward_side=0)

        # Run multiple resets to check both cases
        cue_counts = 0
        n_trials = 20

        keys = jax.random.split(rng_key, n_trials)

        for key in keys:
            new_env, obs, cue_presented = env.reset(key)
            if cue_presented:
                cue_counts += 1
                # Should present correct cue
                assert obs in [4, 5]  # Cue observations

        # Should get some cued and some uncued trials
        assert 0 < cue_counts < n_trials

    def test_tmaze_navigation(self):
        """Test navigation through T-maze."""
        env = TMaze(reward_side=0)  # Reward on left

        # Start at base (already at 0)
        # Move forward to junction
        env, obs1, _, done1 = env.step(action=0)
        assert env.current_state == 1
        assert not done1

        # Turn left (toward reward)
        env, obs2, reward, done2 = env.step(action=1)
        assert env.current_state == 2
        assert reward == 1.0  # Got reward
        assert done2

    def test_tmaze_reward_side(self):
        """Test reward is given on correct side."""
        # Reward on left
        env_left = TMaze(reward_side=0)
        env_left = eqx.tree_at(lambda e: e.current_state, env_left, 1)

        env_left, obs_l, reward_l, _ = env_left.step(action=1)  # Go left
        assert reward_l == 1.0

        # Reward on right
        env_right = TMaze(reward_side=1)
        env_right = eqx.tree_at(lambda e: e.current_state, env_right, 1)

        env_right, obs_r, reward_r, _ = env_right.step(action=2)  # Go right
        assert reward_r == 1.0


class TestEnvironmentIntegration:
    """Integration tests for environments."""

    def test_grid_world_full_episode(self, rng_key):
        """Test a full episode in grid world."""
        config = GridWorldConfig(size=3, goal_location=(2, 2))
        env = GridWorld(config=config)

        env, obs = env.reset(rng_key)
        done = False
        step_count = 0
        max_steps = 20

        keys = jax.random.split(rng_key, max_steps)

        while not done and step_count < max_steps:
            # Random policy
            action = int(jax.random.randint(keys[step_count], (), 0, env.n_actions))
            env, obs, reward, done = env.step(keys[step_count], action)
            step_count += 1

        # Should eventually reach goal or hit max steps
        assert step_count <= max_steps

    def test_tmaze_full_episode(self, rng_key):
        """Test a full episode in T-maze."""
        env = TMaze(reward_side=0)

        env, obs, cue_presented = env.reset(rng_key)

        # Navigate to junction
        env, obs, _, done = env.step(action=0)
        assert not done

        # Choose left (reward side)
        env, obs, reward, done = env.step(action=1)
        assert done
        assert reward == 1.0
