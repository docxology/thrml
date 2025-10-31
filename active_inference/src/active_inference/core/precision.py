"""Message passing and precision weighting for hierarchical inference."""

from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class MessageType(Enum):
    """Types of messages in hierarchical inference."""

    BOTTOM_UP = "bottom_up"  # Prediction errors from lower levels
    TOP_DOWN = "top_down"  # Predictions from higher levels
    LATERAL = "lateral"  # Messages within a level


@dataclass
class Message(eqx.Module):
    """A message passed between inference nodes.

    **Attributes:**

    - `content`: The message payload (typically a probability distribution or prediction error)
    - `message_type`: The type of message being passed
    - `precision`: The reliability/confidence in this message
    """

    content: Float[Array, "..."]
    message_type: MessageType
    precision: Float[Array, ""]

    def __init__(
        self,
        content: Float[Array, "..."],
        message_type: MessageType,
        precision: Float[Array, ""] = jnp.array(1.0),
    ):
        """Initialize a message.

        **Arguments:**

        - `content`: The message content
        - `message_type`: Type of message
        - `precision`: Confidence weight (default 1.0)
        """
        self.content = content
        self.message_type = message_type
        self.precision = precision


@dataclass
class Precision(eqx.Module):
    """Precision (inverse variance) parameters for inference.

    Precision weighting allows the brain to modulate the influence of
    different prediction errors, implementing attention and gain control.

    **Attributes:**

    - `sensory_precision`: Precision of sensory observations
    - `state_precision`: Precision of state beliefs
    - `action_precision`: Precision for action selection (inverse temperature)
    """

    sensory_precision: Float[Array, ""]
    state_precision: Float[Array, ""]
    action_precision: Float[Array, ""]

    def __init__(
        self,
        sensory_precision: float = 1.0,
        state_precision: float = 1.0,
        action_precision: float = 1.0,
    ):
        """Initialize precision parameters.

        **Arguments:**

        - `sensory_precision`: Reliability of sensory observations
        - `state_precision`: Reliability of state transitions
        - `action_precision`: Inverse temperature for action selection
        """
        self.sensory_precision = jnp.array(sensory_precision)
        self.state_precision = jnp.array(state_precision)
        self.action_precision = jnp.array(action_precision)


class PrecisionWeighting:
    """Utility functions for precision-weighted inference."""

    @staticmethod
    def weight_prediction_error(
        prediction_error: Float[Array, "..."],
        precision: Float[Array, ""],
    ) -> Float[Array, "..."]:
        """Apply precision weighting to prediction error.

        **Arguments:**

        - `prediction_error`: Raw prediction error
        - `precision`: Precision (inverse variance)

        **Returns:**

        - Precision-weighted prediction error
        """
        return precision * prediction_error

    @staticmethod
    def softmax_with_precision(
        values: Float[Array, "..."],
        precision: Float[Array, ""],
    ) -> Float[Array, "..."]:
        """Softmax with precision (inverse temperature).

        **Arguments:**

        - `values`: Negative expected free energies or utilities
        - `precision`: Inverse temperature parameter

        **Returns:**

        - Precision-weighted softmax probabilities
        """
        scaled_values = precision * values
        max_value = jnp.max(scaled_values)
        exp_values = jnp.exp(scaled_values - max_value)
        return exp_values / (jnp.sum(exp_values) + 1e-16)

    @staticmethod
    def update_sensory_precision(
        prediction_errors: Float[Array, "n_errors"],
        learning_rate: float = 0.1,
    ) -> Float[Array, ""]:
        """Update sensory precision based on prediction errors.

        Implements online precision estimation based on the variance
        of recent prediction errors.

        **Arguments:**

        - `prediction_errors`: Recent prediction errors
        - `learning_rate`: Learning rate for precision updates

        **Returns:**

        - Updated precision estimate
        """
        variance = jnp.var(prediction_errors) + 1e-16
        precision = 1.0 / variance
        return jnp.array(precision)
