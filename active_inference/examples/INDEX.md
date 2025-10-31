# Examples Index

**For comprehensive documentation, navigation, and guides, see [active_inference/docs/](../docs/)**.

## Quick Reference

### By Topic

**Getting Started**
- Example 00: Probabilistic computing (Potts model) - **NOTEBOOK TRANSLATION**
- Example 01: Comprehensive THRML (Gaussian PGM) - **NOTEBOOK TRANSLATION**
- Example 02: Spin models (Ising/Boltzmann) - **NOTEBOOK TRANSLATION**
- Example 14: Basic THRML inference
- Example 15: Grid world agent (THRML-based)
- See [Getting Started Guide](../docs/getting_started.md)

**Core Concepts**
- Example 04: MDP (THRML-based)
- Example 05: POMDP (THRML-based)
- Example 10: Active inference fundamentals (THRML-based)
- See [Theory Documentation](../docs/theory.md)

**Statistical Methods**
- Example 06: Bayesian coin flip with THRML comparison (**USES THRML**)
- Example 12: Statistical validation (**USES THRML**)
- Example 13: Meta-analysis with THRML comparison (**USES THRML**)
- See [Workflows & Patterns](../docs/workflows_patterns.md)

**Performance & Optimization**
- Example 07: Matrix performance
- Example 08: Signal processing
- Example 09: Control theory with THRML state estimation (**USES THRML**)

**THRML Integration**
- **ALL Examples Use THRML**: Examples 00-06, 09-15 use THRML sampling-based inference
- **Examples 00-02**: Direct notebook translations - complete THRML from working notebooks
- Example 11: **PRIMARY THRML EXAMPLE** - Comprehensive real THRML methods
  - All methods from `examples/01_all_of_thrml.ipynb`, `examples/00_probabilistic_computing.ipynb`, `examples/02_spin_models.ipynb`
  - `ThrmlInferenceEngine`, `Block`, `BlockGibbsSpec`, `CategoricalNode`, `SpinNode`
  - `sample_states`, `sample_with_observation`, `FactorSamplingProgram`, `CategoricalEBMFactor`
  - `AbstractNode`, `AbstractFactor`, `AbstractConditionalSampler` for custom types
  - `InteractionGroup`, `MomentAccumulatorObserver`, state management utilities
  - `IsingEBM`, `estimate_kl_grad`, `hinton_init` for Boltzmann machines
  - Custom nodes, factors, samplers, and interactions
  - Heterogeneous graphs, clamping/conditioning, higher-order interactions
  - GPU-accelerated, energy-efficient inference
- Examples 03-06, 09-10, 12, 14-15: Direct THRML inference
- Examples 03-05, 15: Agents use THRML internally
- Example 06: THRML sampling comparison with analytical Beta-Binomial
- Example 09: THRML state estimation with noisy observations
- Example 13: THRML sampling comparison on subset of meta-analysis trials
- See [THRML Integration Guide](../docs/thrml_integration.md)

**Advanced Topics**
- Example 03: Precision control
- Example 12: Validation & resource tracking

### By Module

**Core**
- Examples: 00-02, 03-05, 10, 14-15
- Docs: [module_core.md](../docs/module_core.md)

**Inference**
- Examples: 00-02 (direct THRML), 06 (THRML + analytical), 10 (THRML), 11 (THRML), 13 (THRML + analytical), 14-15 (THRML)
- Docs: [module_inference.md](../docs/module_inference.md)

**Agents**
- Examples: 03 (THRML), 04 (THRML), 05 (THRML), 15 (THRML)
- Docs: [module_agents.md](../docs/module_agents.md)

**Environments**
- Examples: 03, 15
- Docs: [module_environments.md](../docs/module_environments.md)

**Utils & Visualization**
- Examples: 12 (validation, stats, resources)
- Docs: [module_utils.md](../docs/module_utils.md), [module_visualization.md](../docs/module_visualization.md)

## THRML Methods Cross-Reference

Complete index of THRML methods from working notebooks and their usage in active_inference examples.

### Node Types
- `CategoricalNode` - Examples 00-02, 10-11, 14 | Notebooks: 00_probabilistic_computing.ipynb, 01_all_of_thrml.ipynb
- `SpinNode` - Examples 01-02, 11 | Notebooks: 01_all_of_thrml.ipynb, 02_spin_models.ipynb
- `AbstractNode` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb (ContinuousNode)

### Block Management
- `Block` - Examples 00-02, 10-11, 14 | All notebooks
- `BlockSpec` - Examples 01, 11 | Notebook: 01_all_of_thrml.ipynb
- `BlockGibbsSpec` - Examples 00-02, 10-11, 14 | All notebooks

### Sampling
- `sample_states` - Examples 00-02, 10-11, 14 | All notebooks
- `sample_with_observation` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb
- `SamplingSchedule` - Examples 00-02, 10-11, 14 | All notebooks

### Conditionals/Samplers
- `CategoricalGibbsConditional` - Examples 00, 10-11, 14 | Notebooks: 00_probabilistic_computing.ipynb, 01_all_of_thrml.ipynb
- `SpinGibbsConditional` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `AbstractConditionalSampler` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb (GaussianSampler)

### Factors
- `CategoricalEBMFactor` - Examples 00, 10-11, 14 | Notebooks: 00_probabilistic_computing.ipynb, 01_all_of_thrml.ipynb
- `SpinEBMFactor` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `AbstractFactor` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb

### Interactions
- `InteractionGroup` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb

### Programs
- `FactorSamplingProgram` - Examples 00-02, 10-11, 14 | All notebooks
- `BlockSamplingProgram` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb

### Observers
- `MomentAccumulatorObserver` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb
- `StateObserver` - Examples 01, 11 (via documentation) | Notebook: 01_all_of_thrml.ipynb

### State Management
- `make_empty_block_state` - Examples 01, 11 | Notebook: 01_all_of_thrml.ipynb
- `block_state_to_global` - Examples 01, 11 | Notebook: 01_all_of_thrml.ipynb
- `from_global_state` - Examples 01, 11 | Notebook: 01_all_of_thrml.ipynb
- `DEFAULT_NODE_SHAPE_DTYPES` - Examples 01, 11 | Notebook: 01_all_of_thrml.ipynb

### Ising/Boltzmann Models
- `IsingEBM` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `IsingSamplingProgram` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `IsingTrainingSpec` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `estimate_kl_grad` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb
- `hinton_init` - Examples 02, 11 (via documentation) | Notebook: 02_spin_models.ipynb

### Active Inference Integration
- `ThrmlInferenceEngine` - Examples 03-06, 09-10, 12, 14-15 | Custom wrapper for active inference

**Legend**:
- "via documentation" = mentioned in docstrings/comments as reference, not directly invoked in code
- All methods are real THRML methods from the working notebooks, not mocks

## Configuration

All examples use `examples_config.yaml` for centralized configuration.

See [Configuration Documentation](../docs/CONFIGURATION_SUMMARY.md) for details.

## Complete Documentation

- **[📚 Documentation Hub](../docs/README.md)** - Start here
- **[📖 Module Index](../docs/module_index.md)** - Complete API reference
- **[🗺️ Documentation Navigation](../docs/NAVIGATION.md)** - Site map
- **[📝 Workflows Guide](../docs/workflows_patterns.md)** - Patterns and best practices

---

**This is a brief index. Comprehensive documentation is in [active_inference/docs/](../docs/)**
