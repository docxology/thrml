"""Example 02: Spin Models - Exact Notebook Translation.

This is a direct translation of examples/02_spin_models.ipynb.
All code cells are executed in the same order as the notebook.

**Notebook Reference**: examples/02_spin_models.ipynb
**Output**: Prints to console and saves plots to output/ directory
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# === Cell 4: Imports ===

import jax

# Try to import dwave_networkx, provide fallback
try:
    import dwave_networkx

    HAS_DWAVE = True
except ImportError:
    HAS_DWAVE = False
    print("Warning: dwave_networkx not available, using grid graph as fallback")

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# === Cell 5: THRML Imports ===
from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ising import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init
from thrml.pgm import SpinNode


def main():
    """Run the spin models example exactly as in the notebook."""
    print("=" * 70)
    print("Example 02: Spin Models and EBM Training")
    print("Executing notebook code cells in order")
    print("=" * 70)
    print()

    # === Cell 7: Make the graph ===
    pegasus_size = 14

    if HAS_DWAVE:
        # make the graph using DWave's code
        graph = dwave_networkx.pegasus_graph(pegasus_size)
        print(f"Created Pegasus graph (size {pegasus_size})")
    else:
        # Fallback to grid graph
        grid_size = pegasus_size * 3
        graph = nx.grid_graph(dim=(grid_size, grid_size), periodic=False)
        print(f"Created {grid_size}x{grid_size} grid graph (fallback)")

    coord_to_node = {coord: SpinNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)

    # === Cell 9: Define model ===
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    key = jax.random.key(4242)

    key, subkey = jax.random.split(key, 2)
    biases = jax.random.normal(subkey, (len(nodes),))

    key, subkey = jax.random.split(key, 2)
    weights = jax.random.normal(subkey, (len(edges),))

    beta = jnp.array(1.0)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    print(f"Created Ising model: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Model has {len(model.factors)} factors")

    # === Cell 11: Show factor types ===
    factor_types = [x.__class__ for x in model.factors]
    print(f"Factor types: {set(f.__name__ for f in factor_types)}")

    # === Cell 14: Choose data nodes ===
    n_data = 500

    np.random.seed(4242)

    data_inds = np.random.choice(len(graph.nodes), n_data, replace=False)
    data_nodes = [nodes[x] for x in data_inds]

    print(f"Data nodes: {len(data_nodes)}, Latent nodes: {len(nodes) - len(data_nodes)}")

    # === Cell 16: Compute coloring for free sampling ===
    coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    free_coloring = [[] for _ in range(n_colors)]
    # form color groups
    for node in graph.nodes:
        free_coloring[coloring[node]].append(node)

    free_blocks = [Block(x) for x in free_coloring]

    print(f"Free sampling: {n_colors} color groups")

    # === Cell 18: Compute coloring for clamped sampling ===
    # in this case we will just re-use the free coloring
    # you can always do this, but it might not be optimal

    # a graph without the data nodes
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(data_nodes)

    clamped_coloring = [[] for _ in range(n_colors)]
    for node in graph_copy.nodes:
        clamped_coloring[coloring[node]].append(node)

    clamped_blocks = [Block(x) for x in clamped_coloring]

    print(f"Clamped sampling: {n_colors} color groups")

    # === Cell 20: Define data and schedule ===
    # lets define some random "data" to use for our example
    # in real life this could be encoded images, text, video etc
    data_batch_size = 50

    key, subkey = jax.random.split(key, 2)
    data = jax.random.bernoulli(subkey, 0.5, (data_batch_size, len(data_nodes))).astype(jnp.bool)

    # we will use the same sampling schedule for both cases
    schedule = SamplingSchedule(5, 100, 5)  # warmup  # n_samples  # steps_per_sample

    # convenient wrapper for everything you need for training
    training_spec = IsingTrainingSpec(model, [Block(data_nodes)], [], clamped_blocks, free_blocks, schedule, schedule)

    # how many parallel sampling chains to run for each term
    n_chains_free = 50
    n_chains_clamped = 1

    # initial states for each sampling chain
    # THRML comes with simple code for implementing the hinton initialization, which is commonly used with boltzmann machines
    key, subkey = jax.random.split(key, 2)
    init_state_free = hinton_init(subkey, model, free_blocks, (n_chains_free,))
    key, subkey = jax.random.split(key, 2)
    init_state_clamped = hinton_init(subkey, model, clamped_blocks, (n_chains_clamped, data_batch_size))

    print(f"Data batch: {data.shape}")
    print(f"Schedule: warmup={schedule.n_warmup}, samples={schedule.n_samples}, steps={schedule.steps_per_sample}")
    print(f"Chains: free={n_chains_free}, clamped={n_chains_clamped}")

    # === Cell 21: Estimate gradients ===
    print("Estimating KL divergence gradients...")
    # now for gradient estimation!
    # this function returns the gradient estimators for the weights and edges of our model, along with the moment data that was used to estimate them
    # the moment data is also returned in case you want to use it for something else in your training loop
    key, subkey = jax.random.split(key, 2)
    weight_grads, bias_grads, clamped_moments, free_moments = estimate_kl_grad(
        subkey,
        training_spec,
        nodes,  # the nodes for which to compute bias gradients
        edges,  # the edges for which to compute weight gradients
        [data],
        [],
        init_state_clamped,
        init_state_free,
    )

    print("Gradient estimation complete")

    # === Cell 23: Print gradients ===
    print(f"Weight gradients shape: {weight_grads.shape}")
    print(f"Bias gradients shape: {bias_grads.shape}")
    print(
        f"Weight grads - min: {float(weight_grads.min()):.4f}, max: {float(weight_grads.max()):.4f}, mean: {float(weight_grads.mean()):.4f}"
    )
    print(
        f"Bias grads - min: {float(bias_grads.min()):.4f}, max: {float(bias_grads.max()):.4f}, mean: {float(bias_grads.mean()):.4f}"
    )

    # Visualize gradients
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(weight_grads, bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Gradient Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Weight Gradients")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(bias_grads, bins=50, alpha=0.7, color="coral")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Gradient Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Bias Gradients")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("KL Divergence Gradient Estimation", fontsize=14)

    # Save plot
    output_dir = Path(__file__).parent.parent / "output" / "02_spin_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "kl_gradients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Cell 34: Higher-order interactions ===
    print()
    print("Demonstrating higher-order interactions...")
    # this creates a cubic interaction s_1 * s_2 * s_3 between a subset of our nodes
    key, subkey = jax.random.split(key, 2)
    cubic_factor = SpinEBMFactor(
        [Block(nodes[:10]), Block(nodes[10:20]), Block(nodes[20:30])], jax.random.normal(subkey, (10,))
    )

    print("Created cubic SpinEBMFactor")
    print(f"Factor node groups: {len(cubic_factor.node_groups)}")
    print(f"Nodes per group: {len(cubic_factor.node_groups[0].nodes)}")

    # === Summary ===
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Executed all notebook cells in order")
    print(f"✓ Created Ising/Boltzmann model with {len(nodes)} nodes")
    print(f"✓ Set up EBM training with {len(data_nodes)} data nodes")
    print("✓ Estimated KL divergence gradients")
    print("✓ Demonstrated higher-order (cubic) spin interactions")
    print("")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
