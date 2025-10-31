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

- **Generative Models**: POMDP-style models (A, B, C, D matrices) for discrete and continuous spaces
- **Variational Inference**: Fixed-point iteration and variational message passing
- **THRML Integration**: GPU-accelerated block Gibbs sampling via `ThrmlInferenceEngine`
- **Active Inference Agents**: Complete perception-action-learning loops
- **Test Environments**: GridWorld, T-maze, and custom environment support
- **Comprehensive Visualization**: Belief trajectories, free energy, agent performance, THRML sampling
- **Statistical Analysis**: Regression, correlation, validation, and resource tracking
- **16 Examples**: From basic inference to comprehensive THRML integration (examples 00-15)
- **Test Suite**: >90% coverage with unit, integration, and property-based tests

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
from active_inference.core import GenerativeModel
from active_inference.inference import infer_states
from active_inference.agents import ActiveInferenceAgent
from active_inference.environments import GridWorld
from active_inference.models import build_grid_world_model

# 1. Basic Inference
model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
key = jax.random.key(42)
posterior, free_energy = infer_states(
    observation=0,
    prior_belief=model.D,
    model=model
)

# 2. Active Inference Agent
env = GridWorld(size=5, goal_location=(4, 4))
model = build_grid_world_model(env.config)
agent = ActiveInferenceAgent(model=model, planning_horizon=3)

# Run agent-environment loop
agent_state = agent.reset()
observation = env.reset(key)[1]

for step in range(100):
    action, agent_state, fe = agent.step(key, observation, agent_state)
    env, observation, reward, done = env.step(key, action)
    if done:
        break
```

## Project Structure

```
active_inference/
├── src/active_inference/     # Main package source
│   ├── core/                 # Core active inference (models, free energy, precision)
│   ├── inference/            # Inference engines (variational, THRML)
│   ├── agents/               # Active inference agents and planning
│   ├── environments/         # Test environments (GridWorld, TMaze)
│   ├── models/               # Generative model builders
│   ├── utils/                # Metrics, validation, statistical analysis, resource tracking
│   └── visualization/        # Comprehensive plotting and animation
├── tests/                    # Comprehensive test suite (>90% coverage)
│   ├── test_core.py          # Core component tests
│   ├── test_inference.py     # Inference engine tests
│   ├── test_agents.py        # Agent behavior tests
│   ├── test_environments.py  # Environment tests
│   ├── test_integration.py   # End-to-end integration tests
│   └── test_thrml_integration.py  # THRML-specific tests
├── examples/                 # 16 example scripts (00-15)
│   ├── 00-02: THRML notebooks translated (Potts, Gaussian PGM, Ising)
│   ├── 03-10: Active inference examples (MDP, POMDP, precision, etc.)
│   ├── 11: PRIMARY comprehensive THRML example
│   ├── 12-13: Statistical validation and meta-analysis
│   └── 14-15: Basic inference and grid world agent
├── docs/                     # Comprehensive documentation
│   ├── getting_started.md    # Installation and quick start
│   ├── theory.md             # Active inference theory
│   ├── api.md                # API reference
│   ├── thrml_integration.md  # THRML integration guide
│   ├── module_*.md           # Per-module documentation (7 modules)
│   └── workflows_patterns.md # Best practices
├── scripts/                  # Development and utility scripts
│   ├── setup.sh              # Automated setup
│   ├── run_tests.sh          # Test runner
│   ├── run_all_examples.sh   # Example runner
│   ├── check.sh              # Code quality checks
│   └── format.sh             # Code formatting
└── output/                   # Example outputs (plots, data, logs)
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

See the `examples/` directory for 16 comprehensive examples (00-15):

### THRML Foundation (00-02)
- **00**: Probabilistic computing with Potts models (notebook translation)
- **01**: Comprehensive THRML methods - Gaussian PGM (notebook translation)
- **02**: Spin models and EBM training (notebook translation)

### Active Inference (03-10)
- **03**: Precision control (exploration vs exploitation)
- **04**: Markov Decision Process (fully observable)
- **05**: Partially Observable MDP (Tiger problem)
- **06**: Bayesian coin flip inference
- **07**: THRML performance benchmarking
- **08**: Signal processing with THRML
- **09**: Control theory (LQR, PID)
- **10**: Active inference fundamentals

### Advanced (11-15)
- **11**: **PRIMARY THRML example** - comprehensive integration
- **12**: Statistical validation and analysis
- **13**: Coin flip meta-analysis
- **14**: Basic THRML sampling inference
- **15**: Grid world agent navigation

**All examples use real THRML methods** - no mocks, actual GPU-accelerated sampling.

Run examples:
```bash
python3 examples/11_thrml_comprehensive.py  # PRIMARY THRML example
python3 examples/15_grid_world_agent.py     # Agent navigation
python3 examples/04_mdp_example.py          # Basic active inference
```

## Documentation

Comprehensive documentation in `docs/` directory:

### Core Documentation
- **[Getting Started](docs/getting_started.md)** - Installation, quick start, first steps
- **[Theory](docs/theory.md)** - Active inference mathematical foundations
- **[Architecture](docs/architecture.md)** - System design and components
- **[API Reference](docs/api.md)** - Complete API documentation

### Module Documentation
- **[Core](docs/module_core.md)** - Generative models, free energy, precision
- **[Inference](docs/module_inference.md)** - Variational and THRML inference
- **[Agents](docs/module_agents.md)** - Active inference agents and planning
- **[Environments](docs/module_environments.md)** - GridWorld, TMaze
- **[Models](docs/module_models.md)** - Model builders
- **[Utils](docs/module_utils.md)** - Metrics, validation, statistics, resources
- **[Visualization](docs/module_visualization.md)** - Plotting and animation

### Guides
- **[THRML Integration](docs/thrml_integration.md)** - Complete THRML methods reference
- **[Workflows & Patterns](docs/workflows_patterns.md)** - Best practices
- **[Custom Models](docs/custom_models.md)** - Building custom generative models
- **[Custom Environments](docs/custom_environments.md)** - Creating environments
- **[Analysis & Validation](docs/analysis_validation.md)** - Statistical tools

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
