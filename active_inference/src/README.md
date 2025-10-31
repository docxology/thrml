# Source Package: active_inference

## Overview

The `src/active_inference/` directory contains the main Python package implementation. This is the installed package that users import when using active inference.

## Package Structure

```
src/active_inference/
├── __init__.py           # Package initialization and exports
├── agents/               # Active inference agents
├── core/                 # Core components (models, free energy)
├── environments/         # Test environments
├── inference/            # Inference engines
├── models/               # Model builders
├── utils/                # Utility functions
└── visualization/        # Visualization suite
```

## Module Organization

### Core Hierarchy

1. **Foundational** (`core/`): Mathematical primitives
2. **Inference** (`inference/`): State inference algorithms
3. **Agents** (`agents/`): Perception-action agents
4. **Models** (`models/`): Model builders
5. **Environments** (`environments/`): Test environments
6. **Utilities** (`utils/`, `visualization/`): Supporting functions

### Dependency Flow

```
environments/ ──┐
                ├──> models/ ──> agents/ ──> (user code)
core/ ──────────┘        ↑
                         │
inference/ ──────────────┘
                         ↑
                         │
utils/ ──────────────────┘
visualization/ ──────────┘
```

## Package Exports

### Top-Level Exports (`__init__.py`)

The package exports core functionality:

```python
from active_inference import (
    # Core
    GenerativeModel,
    HierarchicalGenerativeModel,
    Precision,
    variational_free_energy,
    expected_free_energy,

    # Inference
    infer_states,
    variational_message_passing,
    ThrmlInferenceEngine,

    # Agents
    ActiveInferenceAgent,
    AgentState,
    plan_action,
    plan_with_tree_search,

    # Environments
    GridWorld,
    GridWorldConfig,
    TMaze,

    # Models
    build_grid_world_model,
    build_tmaze_model,

    # Utils
    calculate_kl_divergence,
    calculate_prediction_accuracy,
    plot_belief_trajectory,
    plot_free_energy,
    plot_action_distribution,

    # Visualization (as submodule)
    visualization,
)
```

## Module Documentation

Each module has comprehensive documentation:

- **README.md**: Module overview, usage, design principles
- **AGENTS.md**: Detailed API reference for classes and functions

### Core Modules

| Module | README | AGENTS | Description |
|--------|--------|--------|-------------|
| `core/` | [README](core/README.md) | [AGENTS](core/AGENTS.md) | Generative models, free energy |
| `inference/` | [README](inference/README.md) | [AGENTS](inference/AGENTS.md) | State inference engines |
| `agents/` | [README](agents/README.md) | [AGENTS](agents/AGENTS.md) | Active inference agents |

### Application Modules

| Module | README | AGENTS | Description |
|--------|--------|--------|-------------|
| `models/` | [README](models/README.md) | [AGENTS](models/AGENTS.md) | Model builders |
| `environments/` | [README](environments/README.md) | [AGENTS](environments/AGENTS.md) | Test environments |

### Utility Modules

| Module | README | AGENTS | Description |
|--------|--------|--------|-------------|
| `utils/` | [README](utils/README.md) | [AGENTS](utils/AGENTS.md) | Metrics and utilities |
| `visualization/` | [README](visualization/README.md) | - | Visualization suite |

## Installation

Requires [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Development installation (editable)
cd active_inference
uv pip install -e .

# With development dependencies
uv pip install -e ".[development]"

# With all dependencies
uv pip install -e ".[all]"
```

## Import Patterns

### Recommended Imports

```python
# Import from top-level package
from active_inference import (
    GenerativeModel,
    ActiveInferenceAgent,
    infer_states,
)

# Import modules
from active_inference import core, agents, inference

# Import submodules
from active_inference.core import generative_model
from active_inference.agents import planning
```

### Direct Imports (When Needed)

Some functions are not exported at top level but can be imported directly:

```python
# Not exported in __init__.py - import directly
from active_inference.core.free_energy import batch_expected_free_energy
from active_inference.core.generative_model import normalize_distribution, softmax_stable
from active_inference.utils.metrics import calculate_policy_entropy, calculate_expected_utility
from active_inference.inference.state_inference import update_belief_batch
```

## THRML Integration

### Current Integration

The package uses THRML components:

```python
from thrml import Block, CategoricalNode, BlockGibbsSpec
from thrml import sample_states, SamplingSchedule
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor
```

### Integration Status

- ✅ THRML types imported and used
- ✅ `ThrmlInferenceEngine` template implemented
- ⚠️ Full factor-based inference in progress
- 🔄 Hardware acceleration future work

See [docs/thrml_integration.md](../../docs/thrml_integration.md) for details.

## Testing

The package has comprehensive tests in `tests/`:

```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_core.py
pytest tests/test_inference.py
pytest tests/test_agents.py

# With coverage
pytest tests/ --cov=active_inference --cov-report=html
```

See [tests/README.md](../../tests/README.md) for test organization.

## Development

### Code Structure

- **Type Annotations**: Full type hints with `jaxtyping`
- **JAX Compatible**: All functions work with JAX JIT
- **Equinox Modules**: Classes use `equinox.Module`
- **Functional**: Pure functions where possible
- **No Mocks**: Real operations, no mocking

### Style Guidelines

- **Line Length**: 100 characters
- **Formatting**: Black + isort
- **Linting**: Ruff
- **Type Checking**: Pyright

### Adding New Modules

When adding new functionality:

1. Create module directory: `src/active_inference/new_module/`
2. Add `__init__.py` with exports
3. Create implementation files
4. Add `README.md` and `AGENTS.md`
5. Export key functions in package `__init__.py`
6. Add tests in `tests/test_new_module.py`
7. Add examples in `examples/`
8. Update documentation

## Package Metadata

Package metadata is defined in `pyproject.toml`:

```toml
[project]
name = "active_inference"
version = "0.1.0"
description = "Active Inference with THRML"
requires-python = ">=3.10"

dependencies = [
    "jax>=0.4.0",
    "jaxtyping>=0.2.0",
    "equinox>=0.11.0",
    "thrml>=0.1.0",
]
```

## Dependencies

### Core Dependencies

- `jax` >= 0.4.0: Array operations and JIT compilation
- `jaxtyping` >= 0.2.0: Type annotations for JAX arrays
- `equinox` >= 0.11.0: Module system and utilities
- `thrml` >= 0.1.0: THRML probabilistic inference

### Optional Dependencies

- `matplotlib` >= 3.7.1: Visualization
- `seaborn` >= 0.12.0: Enhanced visualization
- `networkx` >= 3.0: Network graphs
- `pytest` >= 7.2.0: Testing
- `pytest-cov` >= 4.0.0: Coverage
- `ruff`: Linting
- `black`: Formatting
- `pyright`: Type checking

Install groups:
```bash
uv pip install -e ".[viz]"          # Visualization
uv pip install -e ".[test]"         # Testing
uv pip install -e ".[development]"  # Development tools
uv pip install -e ".[all]"          # Everything
```

## Build and Distribution

### Build Package

```bash
python -m build
```

This creates:
- `dist/active_inference-0.1.0.tar.gz`
- `dist/active_inference-0.1.0-py3-none-any.whl`

### Install from Distribution

```bash
uv pip install dist/active_inference-0.1.0-py3-none-any.whl
```

## Version Management

Version is managed in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

For future releases, follow semantic versioning:
- `0.1.0` → `0.2.0`: New features
- `0.1.0` → `0.1.1`: Bug fixes
- `0.1.0` → `1.0.0`: Stable API

## Package Files Generated

When installed, the package generates:

- `src/active_inference.egg-info/`: Package metadata
- `src/active_inference/__pycache__/`: Compiled Python files
- `.pytest_cache/`: Test cache
- `.ruff_cache/`: Linter cache
- `htmlcov/`: Coverage reports

## Related Documentation

- **Main README**: [../../README.md](../../README.md)
- **Documentation**: [../../docs/README.md](../../docs/README.md)
- **Examples**: [../../examples/README.md](../../examples/README.md)
- **Tests**: [../../tests/README.md](../../tests/README.md)
- **Quick Reference**: [../../QUICK_REFERENCE.md](../../QUICK_REFERENCE.md)
