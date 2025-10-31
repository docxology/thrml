# Performance Guide

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [Module Index](module_index.md) | [Workflows](workflows_patterns.md)

Optimization strategies and performance considerations for active inference systems.

## Overview

This guide provides practical strategies for optimizing active inference implementations, from basic JAX optimization to advanced THRML sampling strategies.

## Performance Characteristics

### Computational Complexity

| Component | Complexity | Bottleneck |
|-----------|-----------|------------|
| State inference (variational) | O(n_iterations × n_states) | Matrix operations |
| State inference (THRML) | O(n_samples × n_states) | Sampling operations |
| Expected free energy | O(n_actions × n_states²) | Transition predictions |
| Tree search planning | O(n_actions^horizon) | Exponential growth |
| Observation likelihood | O(n_states × n_obs) | Matrix multiplication |

### Memory Usage

```python
from active_inference.utils import estimate_resources

# Estimate requirements
estimates = estimate_resources(
    n_states=100,
    n_observations=50,
    n_actions=10,
    n_steps=1000
)

print(f"Estimated time: {estimates['time_seconds']:.1f}s")
print(f"Estimated memory: {estimates['memory_mb']:.1f} MB")
```

## Optimization Strategies

### 1. JAX JIT Compilation

**Purpose**: Compile functions for faster execution

**Usage**:
```python
import jax

# JIT compile inference
from active_inference.inference import infer_states

jit_infer = jax.jit(infer_states, static_argnums=(2, 3))

# Use compiled version
posterior, fe = jit_infer(observation, prior, model, 16)
```

**Benefits**:
- 10-100x speedup for repeated calls
- Optimized memory access patterns
- Reduced Python overhead

**When to Use**:
- Functions called repeatedly in loops
- Performance-critical sections
- After profiling identifies bottlenecks

---

### 2. Vectorization with vmap

**Purpose**: Batch operations over multiple samples

**Usage**:
```python
import jax

# Vectorize over observations
def infer_single(obs):
    return infer_states(obs, model.D, model)[0]

# Process batch
observations = jnp.array([0, 1, 2, 3])
posteriors = jax.vmap(infer_single)(observations)
# posteriors.shape = (4, n_states)
```

**Benefits**:
- Process multiple inputs in parallel
- Utilize SIMD operations
- Efficient memory usage

---

### 3. Reduce Inference Iterations

**Purpose**: Trade accuracy for speed

**Strategy**:
```python
# Default (high accuracy)
posterior, fe = infer_states(obs, prior, model, n_iterations=16)

# Fast (reduced accuracy)
posterior, fe = infer_states(obs, prior, model, n_iterations=4)
```

**Guidelines**:
- Monitor convergence: track FE decrease
- Small state spaces: fewer iterations needed
- Real-time control: prioritize speed
- Offline analysis: use more iterations

---

### 4. Caching Model Predictions

**Purpose**: Avoid redundant computations

**Implementation**:
```python
from functools import lru_cache

class CachedGenerativeModel:
    def __init__(self, model):
        self.model = model
        self._transition_cache = {}

    def get_state_transition(self, action):
        if action not in self._transition_cache:
            self._transition_cache[action] = self.model.B[:, :, action]
        return self._transition_cache[action]
```

**When to Use**:
- Deterministic models
- Repeated action evaluations
- Planning algorithms

---

### 5. Precision Tuning

**Purpose**: Balance exploration vs computational cost

**Strategy**:
```python
from active_inference.core import Precision

# Fast but less precise
fast_precision = Precision(
    sensory_precision=0.5,
    action_precision=1.0
)

# Slower but more precise
accurate_precision = Precision(
    sensory_precision=2.0,
    action_precision=5.0
)
```

**Trade-offs**:
- Lower precision → Faster convergence, more exploration
- Higher precision → Slower convergence, better accuracy

---

### 6. THRML Sampling Optimization

**Purpose**: Optimize sampling-based inference

**Strategies**:

**a) Reduce Sample Count**
```python
from active_inference.inference import ThrmlInferenceEngine

# Fast sampling
engine = ThrmlInferenceEngine(
    model=model,
    n_samples=100,    # Reduced from 1000
    n_warmup=10       # Reduced from 100
)
```

**b) Use Efficient Samplers**
```python
from thrml import BlockGibbsSpec, SamplingSchedule

# Optimize sampling schedule
schedule = SamplingSchedule(
    block_update_order="sequential",  # vs "random"
    adaptation_enabled=True
)
```

---

### 7. Model Size Reduction

**Purpose**: Reduce state space dimensionality

**Techniques**:

**a) State Aggregation**
```python
# Original: 100 fine-grained states
# Reduced: 10 abstract states
def aggregate_states(fine_states):
    return fine_states.reshape(10, 10).sum(axis=1)
```

**b) Observation Aliasing**
```python
# Fewer observations than states
config = GridWorldConfig(
    size=5,                # 25 states
    n_observations=9       # Only 9 observations
)
```

---

### 8. Planning Optimization

**Purpose**: Reduce planning computational cost

**Strategies**:

**a) Reduce Planning Horizon**
```python
# Long-term (slow)
agent = ActiveInferenceAgent(model, planning_horizon=5)

# Short-term (fast)
agent = ActiveInferenceAgent(model, planning_horizon=1)
```

**b) Greedy Planning**
```python
from active_inference.agents import plan_action

# Deterministic greedy (fastest)
action = plan_action(belief, model, precision)

# vs tree search (slower but better)
action = plan_with_tree_search(key, belief, model, horizon=3)
```

---

## Profiling and Monitoring

### Resource Tracking

```python
from active_inference.utils import ResourceTracker

tracker = ResourceTracker()
tracker.start()

# Your code
for step in range(1000):
    action, state, fe = agent.step(key, obs, state)
    if step % 100 == 0:
        tracker.snapshot(f"step_{step}")

tracker.stop()
print(tracker.generate_report())
```

### JAX Profiling

```python
import jax.profiler

# Profile code section
with jax.profiler.trace("/tmp/jax-trace"):
    for _ in range(100):
        posterior, fe = infer_states(obs, prior, model)

# View: chrome://tracing
```

---

## Performance Comparison

### Variational vs THRML Inference

| State Space Size | Variational Time | THRML Time | Winner |
|-----------------|------------------|------------|---------|
| 10 states | 0.5 ms | 5 ms | Variational (10x) |
| 50 states | 2 ms | 15 ms | Variational (7x) |
| 100 states | 10 ms | 30 ms | Variational (3x) |
| 500 states | 100 ms | 80 ms | THRML (1.25x) |
| 1000 states | 500 ms | 150 ms | THRML (3x) |

**Recommendation**: Use variational for < 100 states, THRML for > 100 states.

### Planning Strategies

| Strategy | Time per Decision | Quality |
|----------|------------------|---------|
| Greedy (horizon=1) | 1 ms | Good |
| Short lookahead (horizon=3) | 5 ms | Better |
| Tree search (horizon=5) | 50 ms | Best |

---

## Memory Optimization

### 1. Use float32 Instead of float64

```python
import jax
jax.config.update("jax_enable_x64", False)

# All operations now use float32 (2x memory savings)
```

### 2. Batch Size Management

```python
# Process in chunks to avoid OOM
def process_large_batch(observations, batch_size=32):
    results = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i+batch_size]
        results.append(process_batch(batch))
    return jnp.concatenate(results)
```

### 3. Gradient Checkpointing

```python
from jax import checkpoint

# For training/optimization
@checkpoint
def expensive_forward(params, x):
    # Recompute instead of storing intermediates
    return forward_pass(params, x)
```

---

## Real-Time Performance

### Target Latencies

| Application | Target Latency | Strategy |
|-------------|---------------|----------|
| Robot control | < 10 ms | Variational, horizon=1, JIT |
| Interactive agents | < 100 ms | Variational, horizon=2, caching |
| Offline planning | < 1 s | THRML, tree search, full accuracy |
| Batch analysis | Unlimited | THRML, large samples, full precision |

### Real-Time Example

```python
import jax

# Compile for speed
@jax.jit
def fast_agent_step(key, obs, state):
    # Minimal inference iterations
    posterior, fe = infer_states(obs, state.belief, model, n_iterations=4)

    # Greedy action selection
    action = plan_action(posterior, model, precision)

    return action, posterior

# Use in control loop
for step in range(1000):
    action, belief = fast_agent_step(key, obs, belief)
    # Execute action in < 5ms
```

---

## GPU Acceleration

### Enable GPU

```python
import jax

# Check device
print(jax.devices())

# Force GPU
import os
os.environ['JAX_PLATFORMS'] = 'gpu'

# Verify
print(jax.default_backend())  # Should show 'gpu'
```

### GPU-Optimized Code

```python
# Large batch operations benefit most
batch_size = 1000
observations = jnp.array([...])  # Large batch

# GPU parallelizes across batch
posteriors = jax.vmap(lambda obs: infer_states(obs, prior, model)[0])(observations)
```

---

## Bottleneck Analysis

### Common Bottlenecks

1. **Large Transition Tensors**: O(n_states² × n_actions)
   - **Solution**: Sparse representations, state aggregation

2. **Repeated EFE Calculations**: O(n_actions × n_states²) per step
   - **Solution**: Cache transitions, reduce actions

3. **Deep Tree Search**: O(n_actions^horizon)
   - **Solution**: Reduce horizon, beam search, pruning

4. **THRML Sampling**: O(n_samples × n_states)
   - **Solution**: Reduce samples, early stopping

### Profiling Checklist

- [ ] Identify hottest functions (JAX profiler)
- [ ] Check for Python loops (vectorize with vmap)
- [ ] Verify JIT compilation (no recompiles)
- [ ] Monitor memory usage (ResourceTracker)
- [ ] Check device utilization (GPU usage)
- [ ] Profile sampling convergence (THRML plots)

---

## Cross-References

- [Utils Module](module_utils.md#resource-tracking) - Resource monitoring
- [Workflows](workflows_patterns.md#optimization-patterns) - Optimization patterns
- [THRML Integration](thrml_integration.md) - THRML performance
- [Architecture](architecture.md#performance-considerations) - System design

---

## Examples

- [Example 07: Matrix Performance](../examples/07_matrix_performance.py)
- [Example 11: THRML Comprehensive](../examples/11_thrml_comprehensive.py)

---

> **Next**: [Workflows](workflows_patterns.md) | [Optimization Patterns](workflows_patterns.md#optimization-patterns) | [Resource Tracking](module_utils.md#resource-tracking)
