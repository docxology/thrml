"""Example 00: Probabilistic Computing with THRML - Exact Notebook Translation.

This is a direct translation of examples/00_probabilistic_computing.ipynb.
All code cells are executed in the same order as the notebook.

**Notebook Reference**: examples/00_probabilistic_computing.ipynb
**Output**: Prints to console and saves plots to output/ directory
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# === Cell 5: Imports ===
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

# === Cell 6: THRML Imports ===
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode


def main():
    """Run the probabilistic computing example exactly as in the notebook."""
    print("=" * 70)
    print("Example 00: Probabilistic Computing - Potts Model")
    print("Executing notebook code cells in order")
    print("=" * 70)
    print()

    # === Cell 8: Define graph parameters ===
    side_length = 20
    print(f"Grid side length: {side_length}")

    # in this simple example we will just use a basic grid, although THRML is capable of dealing with arbitrary graph topologies
    G = nx.grid_graph(dim=(side_length, side_length), periodic=False)

    # label the nodes with something THRML recognizes for convenience
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    for coord, node in coord_to_node.items():
        G.nodes[node]["coords"] = coord

    # write down the color groups for later
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]

    # write down the edges in a different format for later
    u, v = map(list, zip(*G.edges()))

    # plot the graph
    pos = {n: G.nodes[n]["coords"][:2] for n in G.nodes}
    colors = ["black", "orange"]
    node_colors = [colors[bicol[n]] for n in G.nodes]

    fig, axs = plt.subplots()

    nx.draw(
        G,
        pos=pos,
        ax=axs,
        node_size=50.0,
        node_color=node_colors,
        edgecolors="k",
        with_labels=False,
    )

    # Save plot
    output_dir = Path(__file__).parent.parent / "output" / "00_probabilistic_computing"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "graph_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # === Cell 10: Define interactions ===
    # how many categories to use for each variable
    n_cats = 5

    # temperature parameter
    beta = 1.0

    # implements W^{2} for each edge
    # in this case we are just using an identity matrix, but this could be anything
    id_mat = jnp.eye(n_cats)
    weights = beta * jnp.broadcast_to(jnp.expand_dims(id_mat, 0), (len(u), *id_mat.shape))
    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    interactions = [coupling_interaction]

    print(f"Categories: {n_cats}, Beta: {beta}")
    print(f"Coupling weights shape: {weights.shape}")

    # === Cell 12: Build sampling program ===
    # first, we have to specify a division of the graph into blocks that will be updated in parallel during gibbs sampling
    # During gibbs sampling, we are only allowed to update nodes in parallel if they are not neighbours
    # Mathematically, this means we should choose our sampling blocks based on the "minimum coloring" of the graph
    # we already computed this earlier

    # a Block of nodes is simply a sequence of nodes that are all the same type
    # we only have one type of node here, so not important to understand yet
    blocks = [Block(color0), Block(color1)]

    # our grouping of the graph into blocks
    spec = BlockGibbsSpec(blocks, [])

    # we have to define how each node in our blocks should be updated during each iteration of Gibbs sampling
    # THRML comes with a sampler that will do this for the vanilla potts model we are using here, so lets use that
    sampler = CategoricalGibbsConditional(n_cats)

    # now we can make a sampling program, which combines our grouping with the interactions we defined earlier
    prog = FactorSamplingProgram(
        spec,  # our block decomposition of the graph
        [sampler for _ in spec.free_blocks],  # how to update the nodes in each block every iteration of Gibbs sampling
        interactions,  # the interactions present in our model
        [],
    )

    print("Sampling program created")

    # === Cell 14: Run sampling ===
    # rng seed
    key = jax.random.key(4242)

    # everything in THRML is completely compatible with standard jax functionality like jit, vmap, etc.
    # here we will use vmap to run a bunch of parallel instances of Gibbs sampling
    n_batches = 100

    # we need to initialize our Gibbs sampling instances
    init_state = []
    for block in spec.free_blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(
            jax.random.randint(subkey, (n_batches, len(block.nodes)), minval=0, maxval=n_cats, dtype=jnp.uint8)
        )

    # how we should schedule our sampling
    schedule = SamplingSchedule(
        # how many iterations to do before drawing the first sample
        n_warmup=0,
        # how many samples to draw in total
        n_samples=100,
        # how many steps to take between samples
        steps_per_sample=5,
    )

    keys = jax.random.split(key, n_batches)

    # now run sampling
    print("Running sampling (compiling with JIT)...")
    samples = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], [Block(G.nodes)])))(
        init_state, keys
    )

    print(f"Samples shape: {samples[0].shape}")

    # === Cell 16: Visualize domain formation ===
    to_plot = [0, 7, 21]

    fig, axs = plt.subplots(nrows=1, ncols=len(to_plot))

    for i, num in enumerate(to_plot):
        axs[i].imshow(samples[0][num, -1, :].reshape((side_length, side_length)))
        axs[i].set_title(f"Sample {num}")
        axs[i].axis("off")

    fig.suptitle("Domain Formation Over Sampling", fontsize=14)
    fig.savefig(output_dir / "domain_formation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Domain formation visualization saved")

    # === Summary ===
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Executed all notebook cells in order")
    print(f"✓ Created {side_length}x{side_length} grid with {len(G.nodes)} nodes")
    print(f"✓ Ran {n_batches} parallel Gibbs sampling chains")
    print("✓ Observed domain formation in Potts model")
    print("")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
