# Test Suite

## Overview

The `tests` module contains comprehensive test suites for all active inference components. Tests follow test-driven development (TDD) principles with no mocks—all tests use real operations and data.

## Test Structure

### Test Files
- **test_core.py**: Tests for core components (generative models, free energy, precision)
- **test_inference.py**: Tests for inference engines (variational, THRML)
- **test_agents.py**: Tests for agent behavior (perception, action, planning)
- **test_environments.py**: Tests for environments (grid world, T-maze)
- **test_integration.py**: End-to-end integration tests
- **test_thrml_integration.py**: THRML-specific integration tests

### Test Fixtures (`conftest.py`)
- `simple_generative_model`: Basic generative model for testing
- `grid_world_env`: Grid world environment fixture
- `tmaze_env`: T-maze environment fixture
- `simple_agent`: Active inference agent fixture

## Testing Philosophy

1. **No Mocks**: All tests use real JAX operations
2. **Comprehensive Coverage**: Target >90% coverage
3. **Property-Based**: Use Hypothesis for robust testing
4. **Real Data**: Test with actual probability distributions
5. **Mathematical Correctness**: Verify formulas and properties

## THRML Integration Testing

### Current Tests
- Basic THRML component imports
- THRML inference engine template tests
- THRML factor structure tests

### Future Tests
- Full THRML sampling integration
- THRML factor conversion tests
- THRML observer tests
- THRML performance benchmarks

## Running Tests

### Basic Test Run
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ --cov=active_inference --cov-report=html
```

### Specific Test File
```bash
pytest tests/test_core.py -v
```

### Specific Test
```bash
pytest tests/test_core.py::TestGenerativeModel::test_normalization -v
```

## Test Organization

### Unit Tests
- Each component tested in isolation
- Fast, focused tests
- Verify individual function correctness

### Integration Tests
- Test component interactions
- End-to-end workflows
- Verify system behavior

### Property Tests
- Use Hypothesis for generative testing
- Test mathematical properties
- Test edge cases automatically

## Design Principles

1. **Isolation**: Each test is independent
2. **Clarity**: Clear test names and assertions
3. **Realism**: Use realistic test data
4. **Speed**: Tests should run quickly
5. **Maintainability**: Easy to update when code changes

## Dependencies

### Testing Framework
- `pytest >= 7.2.0`: Test framework
- `pytest-cov >= 4.0.0`: Coverage reporting
- `hypothesis >= 6.90.0`: Property-based testing

### Test Data
- Real JAX arrays
- Valid probability distributions
- Realistic environment configurations

## Future Enhancements

1. More THRML integration tests
2. Performance benchmarks
3. Stress tests for large models
4. Continuous integration setup
5. Test coverage reporting
