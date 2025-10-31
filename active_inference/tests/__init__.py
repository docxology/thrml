"""Test suite for active inference implementation."""

import jax

# Configure JAX for strict dtype promotion
jax.config.update("jax_numpy_dtype_promotion", "strict")
