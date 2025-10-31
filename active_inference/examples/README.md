# Active Inference Examples

Quick-start examples demonstrating active inference concepts and THRML integration.

## Quick Start

```bash
# Run an example
python3 examples/00_probabilistic_computing.py

# View example outputs
ls output/00_probabilistic_computing/
```

## Example Scripts

| Example | Description |
|---------|-------------|
| 00 | Probabilistic computing (Potts model) - **NOTEBOOK TRANSLATION** |
| 01 | Comprehensive THRML (Gaussian PGM) - **NOTEBOOK TRANSLATION** |
| 02 | Spin models and EBM training - **NOTEBOOK TRANSLATION** |
| 03 | Precision control (**USES THRML**) |
| 04 | Markov Decision Process - MDP (**USES THRML**) |
| 05 | Partially Observable MDP - POMDP (**USES THRML**) |
| 06 | Bayesian coin flip inference with THRML comparison (**USES THRML**) |
| 07 | THRML performance benchmarking (Potts model from notebook 00) (**USES THRML**) |
| 08 | THRML temporal inference / signal processing (**USES THRML**) |
| 09 | Control theory with THRML state estimation (**USES THRML**) |
| 10 | Active inference fundamentals (**USES THRML**) |
| 11 | **Comprehensive THRML integration** (PRIMARY THRML EXAMPLE - **USES THRML**) |
| 12 | Statistical validation & analysis (**USES THRML**) |
| 13 | Coin flip meta-analysis with THRML comparison (**USES THRML**) |
| 14 | Basic THRML sampling inference (**USES THRML**) |
| 15 | Grid world agent navigation (**USES THRML**) |

## Configuration

Examples use centralized configuration from `examples_config.yaml`. See [Configuration Documentation](../docs/CONFIGURATION_SUMMARY.md) for details.

## Comprehensive Documentation

For complete documentation, theory, and implementation details, see:

- **[📚 Main Documentation](../docs/README.md)** - Start here
- **[🎯 Getting Started Guide](../docs/getting_started.md)** - Installation and first steps
- **[📖 Module Documentation](../docs/module_index.md)** - Complete API reference
- **[🔬 Theory & Concepts](../docs/theory.md)** - Active inference background
- **[⚙️ Configuration Guide](../docs/CONFIGURATION_SUMMARY.md)** - Examples configuration
- **[🔗 THRML Integration](../docs/thrml_integration.md)** - THRML methods and usage
- **[🔄 Workflows & Patterns](../docs/workflows_patterns.md)** - Best practices

## THRML Integration

**ALL 16 Examples Use Real THRML Methods**: Every example (00-15) now uses real THRML methods.

- **Examples 00-02**: Direct notebook translations - complete THRML methods from working notebooks
- **Examples 03-15**: Use THRML via `ThrmlInferenceEngine` or direct THRML sampling
- **Example 07**: Performance benchmarking using exact Potts model from notebook 00
- **Example 08**: Temporal inference / signal processing with THRML
- **Example 11**: PRIMARY comprehensive THRML example

### THRML Methods Comprehensive Reference

All THRML methods from working notebooks (`examples/01_all_of_thrml.ipynb`, `examples/00_probabilistic_computing.ipynb`, `examples/02_spin_models.ipynb`) are available in active_inference:

| Category | Methods | Notebook Source |
|----------|---------|-----------------|
| **Node Types** | `CategoricalNode`, `SpinNode`, `AbstractNode` | All notebooks |
| **Block Management** | `Block`, `BlockSpec`, `BlockGibbsSpec` | All notebooks |
| **Sampling** | `sample_states`, `sample_with_observation`, `SamplingSchedule` | All notebooks |
| **Conditionals** | `CategoricalGibbsConditional`, `SpinGibbsConditional`, `AbstractConditionalSampler` | All notebooks |
| **Factors** | `CategoricalEBMFactor`, `SpinEBMFactor`, `AbstractFactor` | All notebooks |
| **Interactions** | `InteractionGroup`, custom interaction types | 01_all_of_thrml.ipynb |
| **Programs** | `FactorSamplingProgram`, `BlockSamplingProgram` | All notebooks |
| **Observers** | `MomentAccumulatorObserver`, `StateObserver` | 01_all_of_thrml.ipynb |
| **State Management** | `make_empty_block_state`, `block_state_to_global`, `from_global_state` | 01_all_of_thrml.ipynb |
| **Ising Models** | `IsingEBM`, `IsingSamplingProgram`, `IsingTrainingSpec`, `estimate_kl_grad`, `hinton_init` | 02_spin_models.ipynb |

### Primary THRML Example

**Example 11** (`11_thrml_comprehensive.py`) is the PRIMARY THRML example demonstrating comprehensive usage including:
- Real THRML methods (no mocks): `Block`, `BlockGibbsSpec`, `CategoricalNode`, `SpinNode`, `sample_states`, etc.
- `ThrmlInferenceEngine` for GPU-accelerated inference
- Block-based state organization for efficient sampling
- Energy-efficient inference via block Gibbs sampling
- Custom nodes, factors, samplers, and interactions
- Heterogeneous graphs with mixed node types
- Observer patterns for monitoring and data collection
- Complete workflow mirroring notebook patterns

### THRML Usage Across Examples

| Example | THRML Methods Used | Notes |
|---------|-------------------|-------|
| 00 | `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`, `sample_states` | Direct notebook translation (Potts) |
| 01 | Custom nodes/factors/samplers, `MomentAccumulatorObserver` | Direct notebook translation (Gaussian PGM) |
| 02 | `IsingEBM`, `estimate_kl_grad`, `hinton_init`, `SpinEBMFactor` | Direct notebook translation (Spin models) |
| 03-05 | Agent uses `ThrmlInferenceEngine` internally | THRML-based perception |
| 06 | THRML sampling comparison | Analytical vs sampling |
| 07 | Exact Potts model from notebook 00 with performance benchmarks | **Performance/scaling analysis** |
| 08 | `ThrmlInferenceEngine` for HMM filtering | **Temporal inference** |
| 09 | THRML state estimation | Noisy observations |
| 10 | Direct THRML inference | Fundamentals demo |
| **11** | **ALL comprehensive methods** | **Primary THRML example** |
| 12 | Direct THRML inference | Statistical validation |
| 13 | THRML subset comparison | Meta-analysis |
| 14 | Direct THRML inference | Basic inference |
| 15 | Agent uses `ThrmlInferenceEngine` internally | Grid world navigation |

### Advanced THRML Features

**Custom Node Types** (from 01_all_of_thrml.ipynb):
- Inherit from `AbstractNode` to define custom variable types
- Example: `ContinuousNode` for Gaussian variables
- Enables modeling beyond discrete categorical and spin variables

**Custom Factors** (from 01_all_of_thrml.ipynb):
- Inherit from `AbstractFactor` to define custom energy functions
- Implement `to_interaction_groups()` to specify conditional dependencies
- Examples: `QuadraticFactor`, `LinearFactor`, `CouplingFactor` for Gaussian PGM

**Custom Samplers** (from 01_all_of_thrml.ipynb):
- Inherit from `AbstractConditionalSampler` for custom conditionals
- Implement `sample()` method defining update rule
- Example: `GaussianSampler` for continuous variables

**Heterogeneous Graphs** (from 01_all_of_thrml.ipynb):
- Mix different node types (e.g., `SpinNode` + `ContinuousNode`)
- Define cross-type interactions
- Efficient GPU sampling despite heterogeneity

**Clamping/Conditioning** (from 01_all_of_thrml.ipynb):
- Use `sample_with_observation` to fix observed nodes
- Enables P(latent|observed) conditional sampling
- Essential for EBM training workflows

**Higher-Order Interactions** (from 02_spin_models.ipynb):
- `SpinEBMFactor` supports cubic, quartic, and arbitrary-order terms
- More expressive energy functions beyond pairwise
- Critical for advanced spin models

**EBM Training** (from 02_spin_models.ipynb):
- `IsingEBM` for complete Boltzmann machine models
- `estimate_kl_grad` for KL divergence gradient estimation
- `hinton_init` for proper initialization
- `IsingTrainingSpec` for training configuration

**Performance** (from 02_spin_models.ipynb):
- JAX vmap for batched parallel chains
- JAX jit for compilation
- JAX sharding for multi-GPU
- Benchmarking: flips/ns metrics

### THRML Benefits

- **GPU Acceleration**: Efficient parallel sampling on GPUs
- **Energy Efficient**: Block Gibbs sampling optimized for hardware
- **Scalable**: Handles large state spaces efficiently
- **Flexible**: Custom nodes, factors, samplers, and interactions
- **Hardware Ready**: Designed for future Extropic probabilistic computers

See **[THRML Integration Guide](../docs/thrml_integration.md)** for complete documentation.

## Example Structure

All examples follow a standard pattern:

1. **Import** - Load dependencies (THRML imports for examples 00-06, 09-15)
2. **Configure** - Load parameters from `examples_config.yaml` (including THRML parameters)
3. **Setup** - Create models, environments, agents (THRML engines for examples 03-06, 09-10, 12, 14-15)
4. **Execute** - Run inference/agent loops (THRML sampling for all THRML examples)
5. **Analyze** - Calculate metrics, visualize results
6. **Output** - Save data, plots, logs to `output/`

## Requirements

Requires [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package with examples dependencies
cd active_inference
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[examples]"
```

## Support Files

- `example_utils.py` - Shared utilities for all examples
- `examples_config.yaml` - Centralized configuration
- `.cursorrules` - Cursor AI coding rules for examples

---

**For detailed documentation, always refer to [active_inference/docs/](../docs/)**
