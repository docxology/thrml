# Test Suite: Test Functions and Fixtures

## Test Fixtures (`conftest.py`)

### `simple_generative_model`
**Purpose**: Basic generative model for testing.

**Structure**:
- 4 states, 4 observations, 2 actions
- Uniform distributions (A, B, D)
- Flat preferences (C)

**Usage**: Default model for component tests.

### `grid_world_env`
**Purpose**: Grid world environment fixture.

**Configuration**: 5x5 grid with default goal.

**Usage**: Environment tests.

### `tmaze_env`
**Purpose**: T-maze environment fixture.

**Configuration**: Default T-maze setup.

**Usage**: T-maze tests.

### `simple_agent`
**Purpose**: Active inference agent fixture.

**Configuration**: Uses `simple_generative_model` with default precision.

**Usage**: Agent behavior tests.

## Test Files

### `test_core.py`
**Purpose**: Tests for core components.

**Test Classes**:
- `TestGenerativeModel`: Model initialization, normalization, predictions
- `TestFreeEnergy`: VFE and EFE calculations, mathematical correctness
- `TestPrecision`: Precision weighting, softmax, updates

**Key Tests**:
- Model normalization (distributions sum to 1)
- Free energy decomposition (accuracy + complexity)
- EFE calculation (pragmatic + epistemic)
- Precision weighting correctness

### `test_inference.py`
**Purpose**: Tests for inference engines.

**Test Classes**:
- `TestStateInference`: Variational inference correctness
- `TestThrmlInference`: THRML inference engine (template tests)

**Key Tests**:
- Convergence of iterative inference
- Posterior correctness (Bayes rule)
- Batch inference
- THRML component integration

### `test_agents.py`
**Purpose**: Tests for agent behavior.

**Test Classes**:
- `TestActiveInferenceAgent`: Agent perception-action cycle
- `TestPlanning`: Planning algorithm correctness

**Key Tests**:
- Perception correctness
- Action selection (greedy and planned)
- State tracking
- Free energy minimization

### `test_environments.py`
**Purpose**: Tests for environments.

**Test Classes**:
- `TestGridWorld`: Grid world dynamics
- `TestTMaze`: T-maze dynamics

**Key Tests**:
- State transitions
- Observation generation
- Reward calculation
- Boundary conditions

### `test_integration.py`
**Purpose**: End-to-end integration tests.

**Test Classes**:
- `TestAgentEnvironment`: Agent-environment interaction
- `TestModelBuilding`: Model construction from environments

**Key Tests**:
- Complete agent-environment loops
- Model-environment consistency
- Agent performance metrics

### `test_thrml_integration.py`
**Purpose**: THRML-specific integration tests.

**Test Classes**:
- `TestThrmlComponents`: THRML component usage
- `TestThrmlInference`: THRML-based inference

**Key Tests**:
- THRML imports and usage
- THRML factor creation
- THRML sampling (when implemented)
- THRML observer integration

## Usage Patterns

### Running Specific Tests
```bash
# Single test file
pytest tests/test_core.py -v

# Specific test class
pytest tests/test_core.py::TestGenerativeModel -v

# Specific test
pytest tests/test_core.py::TestGenerativeModel::test_normalization -v
```

### With Coverage
```bash
pytest tests/ --cov=active_inference --cov-report=html
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(st.integers(2, 10), st.integers(2, 10))
def test_generative_model_init(n_states, n_obs):
    """Test model initialization with various sizes."""
    model = GenerativeModel(n_states=n_states, n_observations=n_obs)
    assert model.n_states == n_states
    assert model.n_observations == n_obs
```

## THRML Integration Testing

### Current Tests
- THRML component imports
- THRML inference engine structure
- THRML factor creation (basic)

### Future Tests
- Full THRML sampling integration
- THRML factor conversion correctness
- THRML observer functionality
- THRML performance benchmarks

## Test Organization Principles

1. **One Test File Per Module**: Matches source structure
2. **Fixture Sharing**: Common fixtures in `conftest.py`
3. **Clear Naming**: Test names describe what they test
4. **Isolation**: Tests don't depend on each other
5. **Real Data**: No mocks, real operations

## Future Enhancements

1. More THRML integration tests
2. Performance benchmarks
3. Stress tests
4. Continuous integration
5. Coverage reporting automation
