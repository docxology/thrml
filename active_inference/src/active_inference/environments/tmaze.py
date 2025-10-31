"""T-maze environment for testing epistemic foraging."""

import equinox as eqx
import jax
from jaxtyping import Array, Key


class TMaze(eqx.Module):
    """T-maze environment for epistemic behavior.

    The agent starts at the base of a T, must navigate to the top,
    then choose left or right. One side has reward, indicated by a cue
    at the start that the agent may or may not observe.

    This tests epistemic foraging: the agent should explore to resolve
    uncertainty about which side has the reward.

    States:
    0: Start (base of T)
    1: Junction (top of T)
    2: Left arm
    3: Right arm

    Actions:
    0: Forward
    1: Left
    2: Right

    **Attributes:**

    - `reward_side`: Which side has reward (0=left, 1=right)
    - `cue_observed`: Whether agent observed the cue
    - `current_state`: Current state index
    - `n_states`: Number of states (4)
    - `n_observations`: Number of observations
    - `n_actions`: Number of actions (3)
    """

    reward_side: int
    cue_observed: bool
    current_state: int
    n_states: int = 4
    n_observations: int = 8
    n_actions: int = 3

    def __init__(self, reward_side: int = 0):
        """Initialize T-maze.

        **Arguments:**

        - `reward_side`: Which side has reward (0=left, 1=right)
        """
        self.reward_side = reward_side
        self.cue_observed = False
        self.current_state = 0

    def reset(self, key: Key[Array, ""]) -> tuple["TMaze", int, bool]:
        """Reset environment.

        **Arguments:**

        - `key`: JAX random key

        **Returns:**

        - Tuple of (new_environment, observation, cue_was_presented)
        """
        # Randomly decide if cue is presented (50% chance)
        present_cue = jax.random.bernoulli(key, 0.5)
        cue_observed = bool(present_cue)

        if present_cue:
            # Show cue indicating reward side
            obs = 4 + self.reward_side  # Cue observations: 4=left, 5=right
        else:
            # Show neutral start observation
            obs = 0

        # Create new environment with reset state
        new_env = eqx.tree_at(lambda e: e.current_state, self, 0)
        new_env = eqx.tree_at(lambda e: e.cue_observed, new_env, cue_observed)

        return new_env, obs, bool(present_cue)

    def step(
        self,
        action: int,
    ) -> tuple["TMaze", int, float, bool]:
        """Take action and return new environment, observation, reward, done.

        **Arguments:**

        - `action`: Action index (0=forward, 1=left, 2=right)

        **Returns:**

        - Tuple of (new_environment, observation, reward, done)
        """
        done = False
        reward = 0.0
        new_state = self.current_state

        if self.current_state == 0:  # Start
            if action == 0:  # Forward
                new_state = 1
                obs = 1  # Junction observation
            else:
                obs = 0  # Stay at start

        elif self.current_state == 1:  # Junction
            if action == 1:  # Left
                new_state = 2
                obs = 2  # Left arm observation
                reward = 1.0 if self.reward_side == 0 else 0.0
                done = True
            elif action == 2:  # Right
                new_state = 3
                obs = 3  # Right arm observation
                reward = 1.0 if self.reward_side == 1 else 0.0
                done = True
            else:
                obs = 1  # Stay at junction

        else:  # Already at terminal state
            obs = 6 if self.current_state == 2 else 7
            done = True

        # Create new environment with updated state
        new_env = eqx.tree_at(lambda e: e.current_state, self, new_state)

        return new_env, int(obs), float(reward), done

    def get_reward_side(self) -> int:
        """Get which side has the reward.

        **Returns:**

        - Side with reward (0=left, 1=right)
        """
        return self.reward_side
