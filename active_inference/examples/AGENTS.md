# Example Scripts Reference

Quick reference for example scripts. **For comprehensive documentation, see [active_inference/docs/](../docs/)**.

## Example Scripts

### 00_probabilistic_computing.py
Probabilistic computing with THRML - direct from notebook.
- **Exact translation** of `examples/00_probabilistic_computing.ipynb`
- Potts model on grid graph
- Domain formation visualization
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`, `sample_states`

### 01_all_of_thrml.py
Comprehensive THRML methods - direct from notebook.
- **Exact translation** of `examples/01_all_of_thrml.ipynb`
- Custom nodes, factors, and samplers (Gaussian PGM)
- `ContinuousNode`, `QuadraticFactor`, `LinearFactor`, `CouplingFactor`, `GaussianSampler`
- Moment accumulation and covariance validation
- Complete THRML workflow demonstration

### 02_spin_models.py
Ising/Boltzmann models and EBM training - direct from notebook.
- **Exact translation** of `examples/02_spin_models.ipynb`
- `IsingEBM`, `IsingSamplingProgram`, `IsingTrainingSpec`
- KL divergence gradient estimation (`estimate_kl_grad`)
- Hinton initialization (`hinton_init`)
- Higher-order spin interactions

### 03_precision_control.py
Exploration vs exploitation with precision parameters.
- THRML-based agent perception (uses `ThrmlInferenceEngine`)
- Policy entropy analysis
- Precision sweeps
- Behavioral comparisons

### 04_mdp_example.py
Fully observable Markov Decision Process.
- THRML-based agent perception (uses `ThrmlInferenceEngine`)
- Deterministic transitions
- Reward-driven behavior
- Policy optimization

### 05_pomdp_example.py
Partially Observable MDP (Tiger problem).
- THRML-based agent perception (uses `ThrmlInferenceEngine`)
- Information-seeking behavior
- Belief state maintenance
- Epistemic foraging

### 06_coin_flip_inference.py
Bayesian parameter estimation.
- Beta-Binomial inference
- Sequential belief updating
- Credible intervals

### 07_matrix_performance.py
THRML performance benchmarking using Potts model.
- Uses exact code from `00_probabilistic_computing.ipynb`
- Grid size scaling analysis
- Parallel batch scaling
- Real THRML methods: `CategoricalNode`, `Block`, `sample_states`
- Throughput and efficiency metrics

### 08_signal_processing.py
THRML temporal inference for signal processing.
- Hidden Markov Model filtering
- Sequential state estimation
- Signal denoising with THRML sampling
- Uses `ThrmlInferenceEngine`
- Real THRML sampling-based inference

### 09_control_theory.py
Control theory demonstrations.
- LQR and PID controllers
- State space models
- Setpoint tracking

### 10_active_inference_fundamentals.py
Core active inference concepts.
- THRML sampling-based inference (`ThrmlInferenceEngine`)
- Free energy minimization
- Expected free energy components
- Precision control effects

### 11_thrml_comprehensive.py
**PRIMARY THRML INTEGRATION EXAMPLE** - demonstrates comprehensive real THRML methods mirroring the working notebooks:
- **Block management**: Block, BlockGibbsSpec, BlockSpec
- **Node types**: CategoricalNode, SpinNode, custom nodes via AbstractNode
- **Factor-based sampling**: FactorSamplingProgram, CategoricalEBMFactor, custom factors via AbstractFactor
- **Block Gibbs sampling**: sample_states, sample_with_observation, SamplingSchedule
- **Conditional samplers**: CategoricalGibbsConditional, custom samplers via AbstractConditionalSampler
- **Interactions**: InteractionGroup for custom interaction patterns
- **Observers**: MomentAccumulatorObserver, StateObserver for monitoring
- **State management**: make_empty_block_state, block_state_to_global, from_global_state
- **Energy-based modeling**: Real THRML sampling as demonstrated in notebooks
- **ThrmlInferenceEngine**: GPU-accelerated inference wrapper
- **Mirrors**: `examples/01_all_of_thrml.ipynb`, `examples/00_probabilistic_computing.ipynb`, `examples/02_spin_models.ipynb`
- See [THRML Integration Docs](../docs/thrml_integration.md)

**All Examples Now Use THRML**:
- Examples 00-02: Direct notebook translations with complete THRML methods
- Examples 03-06, 09-10, 12, 14-15: All use THRML via `ThrmlInferenceEngine`
- Agents in Examples 03-05, 15 use THRML internally for perception
- Example 11 shows comprehensive THRML methods in detail, mirroring notebook patterns

### 12_statistical_validation_demo.py
Statistical analysis and validation tools.
- THRML sampling-based inference (`ThrmlInferenceEngine`)
- Regression, correlation, effect sizes
- Model and trajectory validation
- Resource tracking
- Enhanced visualization

### 13_coin_flip_meta_analysis.py
Meta-analysis of Bayesian coin flip inference (optimized configuration).
- Parameter sweeps (bias, flip count, priors)
- Convergence analysis
- Statistical validation
- Configurable THRML comparison (disabled by default for speed)

### 14_basic_inference.py
THRML-based sampling inference demonstration.
- Config-driven parameters from `examples_config.yaml`
- THRML sampling inference (`ThrmlInferenceEngine`)
- State inference, free energy calculation
- Belief trajectory visualization

### 15_grid_world_agent.py
Agent-environment interaction in grid world.
- THRML-based agent perception (uses `ThrmlInferenceEngine`)
- Perception-action loops
- Goal-directed navigation
- Performance metrics

## Documentation Links

- **[Complete Examples Guide](../docs/getting_started.md#examples)** - Detailed walkthrough
- **[Module Documentation](../docs/module_index.md)** - API reference for all modules used in examples
- **[Theory Background](../docs/theory.md)** - Active inference concepts
- **[Configuration Guide](../docs/CONFIGURATION_SUMMARY.md)** - Using `examples_config.yaml`
- **[Workflows & Patterns](../docs/workflows_patterns.md)** - Best practices

## Usage Patterns

### Basic Inference
```python
from active_inference.core import GenerativeModel
from active_inference.inference import infer_states

model = GenerativeModel(n_states=4, n_observations=4, n_actions=2)
posterior, fe = infer_states(observation=0, prior_belief=model.D, model=model)
```

### Agent-Environment Loop
```python
from active_inference.agents import ActiveInferenceAgent
from active_inference.environments import GridWorld
from active_inference.models import build_grid_world_model

env = GridWorld(size=5)
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

---

## THRML Methods Reference

Complete reference of THRML methods demonstrated in the working notebooks and integrated into active_inference examples.

### Core Components

**Node Types**:
- `CategoricalNode` - Discrete categorical variables (softmax sampling)
- `SpinNode` - Binary ±1 spin variables (Ising/Boltzmann machines)
- `AbstractNode` - Base class for custom node types (see ContinuousNode example in 01_all_of_thrml.ipynb)

**Block Management**:
- `Block` - Container for homogeneous nodes sampled in parallel
- `BlockSpec` - Defines block structure with shape/dtype information
- `BlockGibbsSpec` - Specifies free blocks (sampled) and clamped blocks (fixed)

**Sampling Functions**:
- `sample_states` - Main THRML sampling function for block Gibbs sampling
- `sample_with_observation` - Sampling with observed/clamped nodes for conditioning
- `SamplingSchedule` - Configuration for warmup, number of samples, steps per sample

### Samplers and Conditionals

**Built-in Samplers**:
- `CategoricalGibbsConditional` - Softmax conditional for categorical nodes
- `SpinGibbsConditional` - Conditional for spin variables (extendable)

**Custom Samplers**:
- `AbstractConditionalSampler` - Base class for custom conditional samplers
- Implement `sample()` method to define custom conditional distributions
- Example: `GaussianSampler` in 01_all_of_thrml.ipynb for continuous variables

### Factors and Interactions

**Built-in Factors**:
- `CategoricalEBMFactor` - Energy-based factor for categorical variables
- `SpinEBMFactor` - Spin factor supporting arbitrary-order interactions (quadratic, cubic, etc.)

**Custom Factors**:
- `AbstractFactor` - Base class for defining custom energy functions
- Implement `to_interaction_groups()` to specify interactions
- Examples in 01_all_of_thrml.ipynb: `QuadraticFactor`, `LinearFactor`, `CouplingFactor`

**Interactions**:
- `InteractionGroup` - Specifies head nodes, tail nodes, and parameters
- Defines what information each node needs for conditional updates
- Custom interaction types (e.g., `LinearInteraction`, `QuadraticInteraction`)

### Program Types

**Sampling Programs**:
- `FactorSamplingProgram` - High-level program built from factors
- `BlockSamplingProgram` - Lower-level program using InteractionGroups directly
- Both compile to efficient GPU-accelerated sampling routines

### Observers

**Monitoring and Data Collection**:
- `MomentAccumulatorObserver` - Accumulates first/second moments during sampling
- `StateObserver` - Records state trajectories during sampling
- Custom observers for domain-specific metrics

### State Management

**State Utilities**:
- `make_empty_block_state` - Initialize block state structures
- `block_state_to_global` - Convert block-organized state to global array
- `from_global_state` - Extract block state from global representation
- `DEFAULT_NODE_SHAPE_DTYPES` - Default shape/dtype mappings for standard nodes

### Ising/Boltzmann Models

**Specialized Components** (from 02_spin_models.ipynb):
- `IsingEBM` - Complete Ising/Boltzmann machine model
- `IsingSamplingProgram` - Optimized sampling for Ising models
- `IsingTrainingSpec` - Training configuration (free/clamped blocks, schedules)
- `estimate_kl_grad` - KL divergence gradient estimation for EBM training
- `hinton_init` - Hinton initialization for Boltzmann machines

### Advanced Patterns

**Custom Node Types** (01_all_of_thrml.ipynb):
- Inherit from `AbstractNode` to define new variable types
- Example: `ContinuousNode` for Gaussian variables

**Custom Factors and Interactions** (01_all_of_thrml.ipynb):
- Define factors by implementing `to_interaction_groups()`
- Create custom interaction types for specialized energy functions
- Example: Gaussian PGM with quadratic and linear interactions

**Heterogeneous Graphs** (01_all_of_thrml.ipynb):
- Mix different node types (e.g., SpinNode + ContinuousNode)
- Define interactions between different node types
- Efficient parallel sampling despite heterogeneity

**Graph Integration** (all notebooks):
- Use NetworkX to construct graph topologies
- Compute graph coloring for parallel block updates
- Map graph edges to THRML factors/interactions

**Clamping and Conditioning** (01_all_of_thrml.ipynb):
- Use `sample_with_observation` to fix subset of nodes
- Enables conditional sampling P(latent|observed)
- Essential for EBM training (data vs model gradients)

**Higher-Order Interactions** (02_spin_models.ipynb):
- `SpinEBMFactor` supports interactions beyond pairwise
- Cubic, quartic, and arbitrary-order terms
- Enables more expressive energy functions

**Performance Optimization** (02_spin_models.ipynb):
- JAX vmap for batched parallel sampling chains
- JAX jit for compilation to optimized kernels
- JAX sharding for multi-GPU parallelization
- Benchmarking: flips/ns metrics for hardware comparison

**Visualization** (01_all_of_thrml.ipynb):
- PCA for visualizing high-dimensional sample distributions
- Graph plotting with NetworkX
- Energy landscape visualization

### Notebook References

**Comprehensive Tutorial**: `examples/01_all_of_thrml.ipynb`
- Custom nodes, factors, samplers, and interactions
- Gaussian PGM implementation from scratch
- Heterogeneous graphs with mixed node types
- State management and observer patterns
- Complete end-to-end workflow

**Potts Models**: `examples/00_probabilistic_computing.ipynb`
- Introduction to probabilistic computing
- Categorical nodes and factors
- Grid graphs and domain formation
- Basic THRML workflow

**Ising/Boltzmann Machines**: `examples/02_spin_models.ipynb`
- Spin models and binary variables
- EBM training with KL divergence gradients
- Higher-order interactions
- GPU performance benchmarking
- Comparison with FPGA implementations

---

**For comprehensive documentation, see [active_inference/docs/](../docs/)**
