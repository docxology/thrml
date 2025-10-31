# Active Inference Theory

> **Navigation**: [Home](README.md) | [Getting Started](getting_started.md) | [Architecture](architecture.md) | [API Reference](api.md) | [Module Index](module_index.md) | [Workflows](workflows_patterns.md)

Mathematical foundations of active inference and the Free Energy Principle.

## Overview

Active Inference is a theoretical framework for understanding perception, action, and learning in biological and artificial agents. It is based on the Free Energy Principle, which proposes that all adaptive systems minimize their variational free energy.

## The Free Energy Principle

The Free Energy Principle states that any self-organizing system at equilibrium with its environment must minimize its free energy. For an agent, this means:

$$F = E_Q[\log Q(s) - \log P(o, s)]$$

Where:
- $F$ is the variational free energy
- $Q(s)$ is the agent's belief about hidden states
- $P(o, s)$ is the joint probability of observations and states under the generative model
- $E_Q[\cdot]$ denotes expectation under $Q$

## Perception as Inference

Perception involves inferring hidden states of the world from observations. This is achieved by minimizing variational free energy with respect to beliefs:

$$\Delta Q \propto -\nabla_Q F$$

This gradient descent on free energy implements approximate Bayesian inference.

## Action as Inference

Action selection involves choosing actions that minimize expected free energy:

$$G(\pi) = E_{Q(o,s|\pi)}[\log Q(s|\pi) - \log P(o, s)]$$

Where:
- $\pi$ is a policy (action sequence)
- $G(\pi)$ is the expected free energy under that policy

Expected free energy can be decomposed into:

$$G = \text{Pragmatic value} + \text{Epistemic value}$$

- **Pragmatic value**: Preference satisfaction (exploitation)
- **Epistemic value**: Information gain (exploration)

## Generative Models

Active inference agents maintain a generative model of their environment, typically structured as a POMDP:

- **A matrix**: Observation likelihood $P(o|s)$
- **B tensor**: State transitions $P(s'|s,a)$
- **C vector**: Preferred observations (goals)
- **D vector**: Initial state prior $P(s_0)$

## Variational Message Passing

Inference is performed through message passing:

1. **Bottom-up messages**: Prediction errors from observations
2. **Top-down messages**: Predictions from higher levels
3. **Lateral messages**: Context from same level

Precision (inverse variance) weights the influence of each message type.

## Implementation in THRML

This implementation uses THRML's efficient block Gibbs sampling to perform:
- State inference via variational optimization
- Action selection via expected free energy minimization
- Model learning via gradient descent

The energy-based formulation of THRML naturally aligns with active inference's free energy minimization objective.

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

2. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29(1), 1-49.

3. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

4. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.

---

## Implementation References

### Core Implementations
- [Free Energy Calculations](module_core.md#free-energy) - `variational_free_energy()`, `expected_free_energy()`
- [State Inference](module_inference.md) - Variational and THRML-based inference
- [Action Selection](module_agents.md#action-selection) - EFE minimization

### Practical Guides
- [Getting Started](getting_started.md) - Apply theory to practice
- [Workflows](workflows_patterns.md) - Implementation patterns
- [Examples](../examples/INDEX.md) - Working code examples

### Advanced Topics
- [THRML Integration](thrml_integration.md) - Energy-based formulation
- [Hierarchical Models](hierarchical_models.md) - Multi-level inference
- [Precision Control](precision_control.md) - Precision weighting

---

> **Next**: [Core Module](module_core.md) | [Inference Module](module_inference.md) | [Getting Started](getting_started.md)
