# Quick Reference Guide

## Installation

```bash
cd active_inference
./scripts/setup.sh
source .venv/bin/activate
```

## Common Commands

### Development
```bash
# Format code
./scripts/format.sh

# Run all checks
./scripts/check.sh

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=active_inference --cov-report=html

# Run specific test
pytest tests/test_core.py::TestGenerativeModel -v
```

### Examples
```bash
python3 examples/01_basic_inference.py
python3 examples/02_grid_world_agent.py
python3 examples/03_precision_control.py
```

## Code Snippets

### Create a Generative Model
```python
from active_inference.core import GenerativeModel
import jax.numpy as jnp

model = GenerativeModel(
    n_states=4,
    n_observations=4,
    n_actions=2,
    A=jnp.eye(4),  # observation likelihood
    # B, C, D will be initialized automatically if not provided
)
```

### Perform Inference
```python
from active_inference.inference import infer_states

posterior, free_energy = infer_states(
    observation=0,
    prior_belief=model.D,
    model=model,
    n_iterations=16,
)
```

### Create an Agent
```python
from active_inference.agents import ActiveInferenceAgent
from active_inference.core import Precision

precision = Precision(action_precision=2.0)
agent = ActiveInferenceAgent(
    model=model,
    precision=precision,
    planning_horizon=3,
)

# Run perception-action cycle
action, agent_state, fe = agent.step(key, observation, agent_state)
```

### Create an Environment
```python
from active_inference.environments import GridWorld, GridWorldConfig

config = GridWorldConfig(
    size=5,
    goal_location=(4, 4),
    obstacle_locations=[(2, 2)],
)
env = GridWorld(config=config)

# Reset and step
obs = env.reset(key)
obs, reward, done = env.step(key, action)
```

### Build a Model for an Environment
```python
from active_inference.models import build_grid_world_model

model = build_grid_world_model(
    config=env.config,
    goal_preference_strength=2.0,
)
```

### Calculate Free Energy
```python
from active_inference.core import variational_free_energy, expected_free_energy

# For inference (perception)
vfe = variational_free_energy(observation, belief, model)

# For planning (action)
efe = expected_free_energy(belief, action, model, horizon=1)
```

### Batch Operations
```python
import jax

# Batch inference
def run_inference(key, obs):
    return infer_states(obs, model.D, model)

keys = jax.random.split(key, 10)
observations = jnp.array([0, 1, 0, 1, 2, 2, 3, 1, 0, 2])

posteriors = jax.vmap(run_inference)(keys, observations)
```

## Project Structure Quick Look

```
active_inference/
├── src/active_inference/    # Source code
│   ├── core/                # Core components
│   ├── inference/           # Inference engines
│   ├── agents/              # Agent implementations
│   ├── environments/        # Test environments
│   ├── models/              # Model builders
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── examples/                # Example scripts
├── docs/                    # Documentation
└── scripts/                 # Dev scripts
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `core.generative_model` | POMDP generative models |
| `core.free_energy` | VFE and EFE calculations |
| `core.precision` | Precision weighting |
| `inference.state_inference` | Variational inference |
| `agents.base_agent` | Active inference agent |
| `agents.planning` | Planning algorithms |
| `environments.grid_world` | Grid navigation |
| `environments.tmaze` | T-maze task |
| `models.discrete_mdp` | Model builders |
| `utils.metrics` | Evaluation metrics |
| `utils.visualization` | Plotting |

## Import Patterns

```python
# Core
from active_inference.core import (
    GenerativeModel,
    variational_free_energy,
    expected_free_energy,
    Precision,
)

# Inference
from active_inference.inference import (
    infer_states,
    ThrmlInferenceEngine,
)

# Agents
from active_inference.agents import (
    ActiveInferenceAgent,
    AgentState,
)

# Environments
from active_inference.environments import (
    GridWorld,
    GridWorldConfig,
    TMaze,
)

# Models
from active_inference.models import (
    build_grid_world_model,
    build_tmaze_model,
)

# Utils
from active_inference.utils import (
    calculate_kl_divergence,
    plot_belief_trajectory,
)
```

## Testing Patterns

```python
# In tests/test_something.py
import pytest
from active_inference.core import GenerativeModel

def test_something(simple_generative_model):
    """Test description."""
    # Arrange
    model = simple_generative_model

    # Act
    result = model.predict_observation(model.D)

    # Assert
    assert result.shape == (model.n_observations,)
    assert jnp.allclose(jnp.sum(result), 1.0)
```

## Configuration Files

- `pyproject.toml`: Project metadata and dependencies
- `.pre-commit-config.yaml`: Pre-commit hooks
- `tests/conftest.py`: Pytest fixtures

## Documentation Files

- `README.md`: Project overview
- `docs/getting_started.md`: Setup guide
- `docs/theory.md`: Theoretical background
- `docs/api.md`: API reference
- `PROJECT_SUMMARY.md`: Comprehensive summary
- `CHANGELOG.md`: Version history

## Useful pytest Options

```bash
# Verbose output
pytest -v

# Very verbose
pytest -vv

# Show print statements
pytest -s

# Run specific test
pytest tests/test_core.py::TestClass::test_method

# Run tests matching pattern
pytest -k "test_model"

# Stop on first failure
pytest -x

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Generate coverage report
pytest --cov=active_inference --cov-report=html

# Show slowest tests
pytest --durations=10
```

## JAX Tips

```python
import jax
import jax.numpy as jnp

# JIT compile for speed
@jax.jit
def fast_function(x):
    return x ** 2

# Vectorize over batch
batch_fn = jax.vmap(fast_function)

# Random keys
key = jax.random.key(42)
key1, key2 = jax.random.split(key)

# Functional updates (JAX arrays are immutable)
arr = jnp.array([1, 2, 3])
new_arr = arr.at[0].set(10)
```

## Common Issues

### Import Error
```bash
# Solution: Install in editable mode
uv pip install -e .
```

### JAX Device Error
```bash
# Solution: Update JAX
uv pip install --upgrade jax jaxlib
```

### Test Failure
```bash
# Solution: Run with verbose output
pytest tests/ -vv --tb=long
```

## Performance Tips

1. Use `jax.jit` for frequently called functions
2. Use `jax.vmap` for batch operations
3. Keep arrays on GPU when possible
4. Profile with `jax.profiler`

## Next Steps

1. Read `docs/getting_started.md`
2. Run examples
3. Read `docs/theory.md`
4. Explore `docs/api.md`
5. Write your own code!
