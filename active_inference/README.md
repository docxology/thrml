# Active Inference with THRML

A comprehensive implementation of Active Inference using the THRML library for energy-efficient probabilistic inference on GPUs and future Extropic hardware.

## Overview

This package implements Active Inference—a unified framework for perception, action, and learning based on the Free Energy Principle—using THRML's efficient block Gibbs sampling for probabilistic graphical models.

Active Inference provides a principled approach to:
- **Perception**: Inferring hidden states of the world from observations
- **Action**: Selecting actions that minimize expected free energy
- **Learning**: Updating generative models through experience

By leveraging THRML's GPU-accelerated sampling and compatibility with future Extropic hardware, this implementation enables efficient active inference at scale.

## Features

- **Generative Models**: Build hierarchical generative models using THRML's PGM components
- **Variational Inference**: Efficient variational message passing using block Gibbs sampling
- **Action Selection**: Policy optimization through expected free energy minimization
- **Model Learning**: Parameter learning via gradient-based optimization
- **Grid World Agents**: Reference implementations for discrete state-action spaces
- **Continuous Inference**: Extensions for continuous state spaces
- **Comprehensive Tests**: Full test coverage with property-based testing

## Installation

Requires [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install the package
cd active_inference
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from active_inference import GenerativeModel, ActiveInferenceAgent
from active_inference.environments import GridWorld

# Create a simple grid world environment
env = GridWorld(size=5, n_observations=10)

# Build a generative model
model = GenerativeModel(
    n_states=env.n_states,
    n_observations=env.n_observations,
    n_actions=env.n_actions,
    state_prior=jnp.ones(env.n_states) / env.n_states,
)

# Create an active inference agent
agent = ActiveInferenceAgent(model=model, planning_horizon=3)

# Run inference
key = jax.random.key(42)
observation = env.reset(key)
action = agent.step(key, observation)
```

## Project Structure

```
active_inference/
├── src/active_inference/     # Main package source
│   ├── core/                 # Core active inference components
│   ├── models/               # Generative model implementations
│   ├── inference/            # Variational inference engines
│   ├── agents/               # Agent implementations
│   ├── environments/         # Test environments
│   └── utils/                # Utility functions
├── tests/                    # Comprehensive test suite
│   ├── test_core.py
│   ├── test_models.py
│   ├── test_inference.py
│   ├── test_agents.py
│   └── test_integration.py
├── examples/                 # Example notebooks and scripts
│   ├── 01_basic_inference.ipynb
│   ├── 02_grid_world_agent.ipynb
│   └── 03_hierarchical_models.ipynb
├── docs/                     # Documentation
└── scripts/                  # Development and utility scripts
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[development]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=active_inference --cov-report=html

# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
pyright src/

# Run all checks
./scripts/check.sh
```

## Testing Philosophy

This project follows test-driven development (TDD) with:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Property-based tests**: Use Hypothesis for robust testing
- **Real data**: No mocks—test with actual JAX operations
- **High coverage**: Aim for >90% test coverage

## Examples

See the `examples/` directory for comprehensive examples including:
- Basic variational inference with THRML
- Grid world navigation with active inference
- Hierarchical generative models
- Continuous state-space inference
- Multi-agent active inference

## Documentation

Full documentation is available in the `docs/` directory, including:
- Theoretical background on Active Inference
- API reference
- Tutorials and guides
- Implementation notes

## Citation

If you use this code in your research, please cite:

```bibtex
@software{active_inference_thrml,
  title = {Active Inference with THRML},
  author = {Active Inference Research Team},
  year = {2025},
  url = {https://github.com/extropic-ai/thrml}
}
```

## License

See LICENSE file in the root directory.

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- THRML Documentation: https://docs.thrml.ai/
