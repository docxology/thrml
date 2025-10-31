# Active Inference Package: Component Overview

## Package Structure

The active inference package is organized into modular components, each with clear responsibilities and integration points.

## Core Modules

### `core/`
**Purpose**: Fundamental active inference components implementing the Free Energy Principle.

**Key Components**:
- `GenerativeModel`: POMDP-style generative models (A, B, C, D matrices)
- `HierarchicalGenerativeModel`: Multi-level hierarchical models
- Free energy functions: `variational_free_energy`, `expected_free_energy`
- `Precision`: Precision weighting for inference and action
- `PrecisionWeighting`: Utility functions for precision-weighted operations
- Message passing: `Message`, `MessageType` for hierarchical inference

**Files**:
- `generative_model.py`: Generative model classes and utilities
- `free_energy.py`: Free energy calculations (VFE, EFE)
- `precision.py`: Precision parameters and weighting

**See**: [core/README.md](src/active_inference/core/README.md) | [core/AGENTS.md](src/active_inference/core/AGENTS.md)

---

### `inference/`
**Purpose**: State inference engines for perception.

**Key Components**:
- `infer_states`: Variational inference via fixed-point iteration
- `variational_message_passing`: Forward-backward inference over sequences
- `ThrmlInferenceEngine`: THRML-based sampling inference (template)
- Batch inference utilities

**Files**:
- `state_inference.py`: Variational inference algorithms
- `thrml_inference.py`: THRML integration for sampling-based inference

**THRML Integration**: Uses THRML blocks, sampling schedules, and factors for energy-efficient inference.

**See**: [inference/README.md](src/active_inference/inference/README.md) | [inference/AGENTS.md](src/active_inference/inference/AGENTS.md)

---

### `agents/`
**Purpose**: Active inference agents combining perception and action.

**Key Components**:
- `ActiveInferenceAgent`: Complete perception-action loop
- `AgentState`: Internal state tracking (beliefs, history, free energy)
- Planning functions: `plan_action`, `plan_with_tree_search`, `evaluate_policy`

**Files**:
- `base_agent.py`: Core agent implementation
- `planning.py`: Planning algorithms for action selection

**Agent Cycle**:
1. Perceive: Infer states from observations (minimize VFE)
2. Act: Select actions minimizing expected free energy
3. Predict: Update beliefs based on selected actions
4. Update: Track history and metrics

**See**: [agents/README.md](src/active_inference/agents/README.md) | [agents/AGENTS.md](src/active_inference/agents/AGENTS.md)

---

### `environments/`
**Purpose**: Test environments for evaluating agents.

**Key Components**:
- `GridWorld`: 2D grid navigation with configurable obstacles and goals
- `GridWorldConfig`: Configuration for grid world setup
- `TMaze`: T-maze for epistemic foraging and information-seeking

**Files**:
- `grid_world.py`: Grid world environment
- `tmaze.py`: T-maze environment

**Usage**: Discrete state-action-observation spaces for testing agent behavior.

**See**: [environments/README.md](src/active_inference/environments/README.md) | [environments/AGENTS.md](src/active_inference/environments/AGENTS.md)

---

### `models/`
**Purpose**: Automatic generative model builders from environment specifications.

**Key Components**:
- `build_grid_world_model`: Create model matching GridWorld dynamics
- `build_tmaze_model`: Create model matching TMaze dynamics

**Files**:
- `discrete_mdp.py`: Model builders for discrete MDPs

**Usage**: Automatically construct properly structured generative models from environment configs.

**See**: [models/README.md](src/active_inference/models/README.md) | [models/AGENTS.md](src/active_inference/models/AGENTS.md)

---

## Utility Modules

### `utils/`
**Purpose**: Evaluation metrics and analysis utilities.

**Key Components**:
- Metrics: `calculate_kl_divergence`, `calculate_prediction_accuracy`, `calculate_policy_entropy`
- Statistical analysis: `StatisticalAnalysis` class with regression, correlation, hypothesis tests
- Validation: `ModelValidator`, `DataValidator` for correctness checks
- Resource tracking: `ResourceTracker` for monitoring memory/CPU/GPU usage

**Files**:
- `metrics.py`: Evaluation metrics
- `visualization.py`: Basic plotting utilities
- `statistical_analysis.py`: Advanced statistical methods
- `validation.py`: Model and data validation
- `resource_tracking.py`: Performance monitoring

**See**: [utils/README.md](src/active_inference/utils/README.md) | [utils/AGENTS.md](src/active_inference/utils/AGENTS.md)

---

### `visualization/`
**Purpose**: Comprehensive visualization suite for active inference and THRML.

**Key Components**:
- Active inference plots: Beliefs, free energy, actions, agent performance
- THRML plots: Sampling trajectories, energy landscapes, convergence diagnostics
- Network plots: Graphical models, factor graphs, Markov blankets
- Statistical plots: Distributions, convergence diagnostics, R-hat, ESS
- Environment plots: Grid worlds, agent trajectories, heatmaps
- Comparison plots: Multi-agent, learning curves, parameter sweeps
- Animations: Dynamic visualizations of beliefs, trajectories, sampling

**Files**:
- `core.py`: Configuration and utilities
- `active_inference_plots.py`: Active inference visualizations
- `thrml_plots.py`: THRML-specific plots
- `network_plots.py`: Graph visualizations
- `statistical_plots.py`: Statistical analysis plots
- `environment_plots.py`: Environment visualizations
- `comparison_plots.py`: Comparison and ablation plots
- `animation.py`: Animation utilities

**See**: [visualization/README.md](src/active_inference/visualization/README.md)

---

## Support Directories

### `docs/`
**Purpose**: Comprehensive documentation for all modules.

**Contents**:
- Getting started guide
- API reference
- Theoretical background
- Module documentation (7 modules)
- THRML integration guide
- Workflows and patterns
- Architecture overview

**See**: [docs/README.md](docs/README.md) | [docs/AGENTS.md](docs/AGENTS.md)

---

### `examples/`
**Purpose**: Example scripts demonstrating library usage.

**Contents**:
- 13 comprehensive examples (01-13)
- Basic inference to advanced THRML integration
- Statistical validation and meta-analysis
- Complete with logging, data saving, visualizations
- Analysis scripts for coin flip demonstrations

**See**: [examples/README.md](examples/README.md) | [examples/AGENTS.md](examples/AGENTS.md)

---

### `tests/`
**Purpose**: Comprehensive test suite following TDD principles.

**Contents**:
- Unit tests for all components
- Integration tests for workflows
- THRML integration tests
- Property-based tests with Hypothesis
- >90% coverage target

**Files**:
- `test_core.py`: Core component tests
- `test_inference.py`: Inference engine tests
- `test_agents.py`: Agent behavior tests
- `test_environments.py`: Environment tests
- `test_integration.py`: End-to-end tests
- `test_thrml_integration.py`: THRML-specific tests
- `conftest.py`: Shared fixtures

**See**: [tests/README.md](tests/README.md) | [tests/AGENTS.md](tests/AGENTS.md)

---

### `scripts/`
**Purpose**: Development and automation scripts.

**Contents**:
- `setup.sh`: Setup script
- `run_tests.sh`: Test runner
- `run_all_examples.sh`: Example runner
- `check.sh`: Code quality checks
- `format.sh`: Code formatting
- `validate_examples.py`: Example validation

**See**: [scripts/README.md](scripts/README.md)

---

### `archive/`
**Purpose**: Historical analysis scripts and demonstrations.

**Contents**:
- `analyze_seed_42.py`: Detailed seed analysis
- `coin_flip_seed_comparison.py`: Multi-seed comparison
- `coin_flip_variance_demo.py`: Sampling variance demonstration

**See**: [archive/README.md](archive/README.md)

---

## THRML Integration

### Current Status

**Implemented**:
- ✅ THRML types imported (`Block`, `CategoricalNode`, `BlockGibbsSpec`)
- ✅ THRML sampling infrastructure (`sample_states`, `SamplingSchedule`)
- ✅ Template `ThrmlInferenceEngine` with block structure
- ✅ Comprehensive THRML integration documentation

**In Progress**:
- ⚠️ Full factor-based inference (using `AbstractFactor`, `FactorSamplingProgram`)
- ⚠️ THRML observer integration for monitoring
- ⚠️ GPU-accelerated batch inference

**Future**:
- 🔄 Hierarchical THRML inference
- 🔄 Continuous state space THRML inference
- 🔄 Hardware-accelerated inference (Extropic)

### Integration Points

1. **Inference**: `ThrmlInferenceEngine` for sampling-based perception
2. **Agents**: THRML sampling for policy evaluation
3. **Models**: Convert `GenerativeModel` to THRML factors
4. **Environments**: THRML factors for transition dynamics
5. **Visualization**: THRML sampling trajectory visualization

**See**: [docs/thrml_integration.md](docs/thrml_integration.md)

---

## Design Principles

1. **Modularity**: Each component is independently usable
2. **Real Operations**: No mocks, actual inference and computation
3. **JAX Compatibility**: All functions JIT-compatible and vectorizable
4. **Type Safety**: Full type annotations with jaxtyping
5. **THRML Ready**: Designed for THRML integration throughout
6. **TDD**: Test-driven development with comprehensive tests
7. **Professional**: Structured logging, error handling, metrics tracking

---

## Quick Reference

### Create Model
```python
from active_inference.core import GenerativeModel
model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
```

### Infer States
```python
from active_inference.inference import infer_states
posterior, fe = infer_states(observation=0, prior_belief=model.D, model=model)
```

### Create Agent
```python
from active_inference.agents import ActiveInferenceAgent
agent = ActiveInferenceAgent(model=model, planning_horizon=3)
```

### Run Agent-Environment Loop
```python
from active_inference.environments import GridWorld
from active_inference.models import build_grid_world_model

env = GridWorld(size=5, goal_location=(4, 4))
model = build_grid_world_model(env.config)
agent = ActiveInferenceAgent(model=model)

agent_state = agent.reset()
observation = env.reset(key)

for step in range(100):
    action, agent_state, fe = agent.step(key, observation, agent_state)
    observation, reward, done = env.step(key, action)
    if done:
        break
```

### Visualize Results
```python
from active_inference import visualization as viz

viz.plot_belief_trajectory(beliefs, true_states=true_states)
viz.plot_free_energy(free_energies)
viz.plot_agent_performance(rewards, free_energies, entropies)
```

---

## Documentation Navigation

- **Getting Started**: [docs/getting_started.md](docs/getting_started.md)
- **API Reference**: [docs/api.md](docs/api.md)
- **Theory**: [docs/theory.md](docs/theory.md)
- **Module Index**: [docs/module_index.md](docs/module_index.md)
- **THRML Integration**: [docs/thrml_integration.md](docs/thrml_integration.md)
- **Workflows**: [docs/workflows_patterns.md](docs/workflows_patterns.md)
- **Architecture**: [docs/architecture.md](docs/architecture.md)

---

## Contributing

When adding new components:

1. Follow modular design principles
2. Add comprehensive tests
3. Document in appropriate AGENTS.md and README.md
4. Update main documentation
5. Add examples if appropriate
6. Ensure THRML compatibility

---

## License

See LICENSE file in root directory.
