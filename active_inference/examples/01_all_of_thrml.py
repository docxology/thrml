"""Example 01: All of THRML - Exact Notebook Translation.

This is a direct translation of examples/01_all_of_thrml.ipynb.
All code cells are executed in the same order as the notebook.

**Notebook Reference**: examples/01_all_of_thrml.ipynb
**Output**: Prints to console and saves plots to output/ directory
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# === Cell 6: Imports ===
from typing import Hashable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from jaxtyping import Array, Key, PyTree

# === Cell 7: THRML Imports ===
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, BlockSamplingProgram, SamplingSchedule, sample_with_observation
from thrml.conditional_samplers import AbstractConditionalSampler, _SamplerState, _State
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode


def main():
    """Run the all of THRML example exactly as in the notebook."""
    print("=" * 70)
    print("Example 01: All of THRML - Gaussian PGM")
    print("Executing notebook code cells in order")
    print("=" * 70)
    print()

    # === Cell 9: Define custom node type ===
    class ContinuousNode(AbstractNode):
        pass

    print("Defined ContinuousNode")

    # === Cell 11: Generate grid graph ===
    def generate_grid_graph(
        *side_lengths: int,
    ) -> tuple[
        tuple[list[ContinuousNode], list[ContinuousNode]], tuple[list[ContinuousNode], list[ContinuousNode]], nx.Graph
    ]:
        G = nx.grid_graph(dim=side_lengths, periodic=False)

        coord_to_node = {coord: ContinuousNode() for coord in G.nodes}
        nx.relabel_nodes(G, coord_to_node, copy=False)

        for coord, node in coord_to_node.items():
            G.nodes[node]["coords"] = coord

        # an aperiodic grid is always 2-colorable
        bicol = nx.bipartite.color(G)
        color0 = [n for n, c in bicol.items() if c == 0]
        color1 = [n for n, c in bicol.items() if c == 1]

        u, v = map(list, zip(*G.edges()))

        return (bicol, color0, color1), (u, v), G

    def plot_grid_graph(
        G: nx.Graph,
        bicol: Mapping[Hashable, int],
        ax: plt.Axes,
        *,
        node_size: int = 300,
        colors: tuple[str, str] = ("black", "orange"),
        **draw_kwargs,
    ):
        pos = {n: G.nodes[n]["coords"][:2] for n in G.nodes}

        node_colors = [colors[bicol[n]] for n in G.nodes]

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="k",
            linewidths=0.8,
            width=1.0,
            with_labels=False,
            **draw_kwargs,
        )

    # === Cell 12: Create graph ===
    grid_size = 5
    colors, edges, g = generate_grid_graph(grid_size, grid_size)

    all_nodes = colors[1] + colors[2]

    node_map = dict(zip(all_nodes, list(range(len(all_nodes)))))

    fig, axs = plt.subplots()

    plot_grid_graph(g, colors[0], axs)

    # Save plot
    output_dir = Path(__file__).parent.parent / "output" / "01_all_of_thrml"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "grid_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Created {grid_size}x{grid_size} grid: {len(all_nodes)} nodes, {len(edges[0])} edges")

    # === Cell 14: Define distribution ===
    # Fixed RNG seed for reproducibility
    key = jax.random.key(4242)

    # diagonal elements of the inverse covariance matrix
    key, subkey = jax.random.split(key, 2)
    cov_inv_diag = jax.random.uniform(subkey, (len(all_nodes),), minval=1, maxval=2)

    # add an off-diagonal element to the inverse covariance matrix for each edge in the graph
    key, subkey = jax.random.split(key, 2)
    # make sure the covariance matrix is PSD
    cov_inv_off_diag = jax.random.uniform(subkey, (len(edges[0]),), minval=-0.25, maxval=0.25)

    def construct_inv_cov(diag: Array, all_edges: tuple[list[ContinuousNode], list[ContinuousNode]], off_diag: Array):
        inv_cov = np.diag(diag)

        for n1, n2, cov in zip(*all_edges, off_diag):
            inv_cov[node_map[n1], node_map[n2]] = cov
            inv_cov[node_map[n2], node_map[n1]] = cov

        return inv_cov

    # construct a matrix representation of the inverse covariance matrix for convenience
    inv_cov_mat = construct_inv_cov(cov_inv_diag, edges, cov_inv_off_diag)

    inv_cov_mat_jax = jnp.array(inv_cov_mat)

    # mean vector
    key, subkey = jax.random.split(key, 2)
    mean_vec = jax.random.normal(subkey, (len(all_nodes),))

    # bias vector
    b_vec = -1 * jnp.einsum("ij, i -> j", inv_cov_mat, mean_vec)

    print(f"Defined Gaussian distribution: {len(all_nodes)} variables")

    # === Cell 17: Define block spec ===
    # a Block is just a list of nodes that are all the same type
    # forcing the nodes in a Block to be of the same type is important for parallelization
    free_blocks = [Block(colors[1]), Block(colors[2])]

    # we won't be clamping anything here, but in principle this could be a list of Blocks just like above
    clamped_blocks = []

    # every node in the program has to be assigned a shape and datatype (or PyTree thereof).
    # this is so THRML can build an internal "global" representation of the state of the sampling program using a small number of jax arrays
    node_shape_dtypes = {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)}

    # our block specification
    spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_shape_dtypes)

    print("Created block specification")

    # === Cell 19: Define interaction types and factors ===
    # these are just arrays that we can identify by type, will be useful later

    class LinearInteraction(eqx.Module):
        r"""An interaction of the form $c_i x_i$."""

        weights: Array

    class QuadraticInteraction(eqx.Module):
        r"""An interaction of the form $d_i x_i^2$."""

        inverse_weights: Array

    # now we can set up our three different types of factors

    class QuadraticFactor(AbstractFactor):
        r"""A factor of the form $w x^2$"""

        # 1/A_{ii}
        inverse_weights: Array

        def __init__(self, inverse_weights: Array, block: Block):
            # in general, a factor is initialized via a list of blocks
            # these blocks should all have the same number of nodes, and represent groupings of nodes involved in the factor
            # for example, if a Factor involved 3 nodes, we would initialize it with 3 parallel blocks of equal length
            super().__init__([block])

            # this array has shape [n], where n is the number of nodes in block
            self.inverse_weights = inverse_weights

        def to_interaction_groups(self) -> list[InteractionGroup]:
            # based on our conditional update rule, we can see that we need this to generate a Quadratic interaction with no tail nodes (i.e this interaction has no dependence on the neighbours of x_i)

            # we create an InteractionGroup that implements this interaction

            interaction = InteractionGroup(
                interaction=QuadraticInteraction(self.inverse_weights),
                head_nodes=self.node_groups[0],
                # no tail nodes in this case
                tail_nodes=[],
            )

            return [interaction]

    class LinearFactor(AbstractFactor):
        r"""A factor of the form $w x$"""

        # b_i
        weights: Array

        def __init__(self, weights: Array, block: Block):
            super().__init__([block])
            self.weights = weights

        def to_interaction_groups(self) -> list[InteractionGroup]:
            # follows the same pattern as previous, still no tail nodes

            return [
                InteractionGroup(
                    interaction=LinearInteraction(self.weights), head_nodes=self.node_groups[0], tail_nodes=[]
                )
            ]

    class CouplingFactor(AbstractFactor):
        # A_{ij}
        weights: Array

        def __init__(self, weights: Array, blocks: tuple[Block, Block]):
            # in this case our factor involves two nodes, so it is initialized with two blocks
            super().__init__(list(blocks))
            self.weights = weights

        def to_interaction_groups(self) -> list[InteractionGroup]:
            # this factor produces interactions that impact both sets of nodes that it touches
            # i.e if this factor involves a term like w x_1 x_2, it should produce one interaction with weight w that has x_1 as a head node and x_2 as a tail node,
            # and another interaction with weight w that has x_2 as a head node and x_1 as a tail node

            # if we were sure that x_1 and x_2 were always the same type of node, the two interactions could be part of the same InteractionGroup
            # we won't worry about that here though
            return [
                InteractionGroup(LinearInteraction(self.weights), self.node_groups[0], [self.node_groups[1]]),
                InteractionGroup(LinearInteraction(self.weights), self.node_groups[1], [self.node_groups[0]]),
            ]

    print("Defined custom factors")

    # === Cell 21: Define custom sampler ===
    class GaussianSampler(AbstractConditionalSampler):
        def sample(
            self,
            key: Key,
            interactions: list[PyTree],
            active_flags: list[Array],
            states: list[list[_State]],
            sampler_state: _SamplerState,
            output_sd: PyTree[jax.ShapeDtypeStruct],
        ) -> tuple[Array, _SamplerState]:
            # this is where the rubber meets the road in THRML

            # this function gets called during block sampling, and must take in information about interactions and neighbour states and produce a state update

            # interactions, active_flags, and states are three parallel lists.

            # each item in interactions is a pytree, for which each array will have shape [n, k, ...].
            # this is generated by THRML from the set of InteractionGroups that are used to create a sampling program
            # n is the number of nodes that we are updating in parallel during this call to sample
            # k is the maximum number of times any node in the block that is being updated shows up as a head node for this interaction

            # each item in active_flags is a boolean array with shape [n, k].
            # this is padding that is generated internally by THRML based on the graphical structure of the model,
            # and serves to allow for heterogeneous graph sampling to be vectorized on accelerators that rely on homogeneous data structures

            # each item in states is a list of Pytrees that represents the state of the tail nodes that are relevant to this interaction.
            # for example, for an interaction with a single tail node that has a scalar dtype, states would be:
            # [[n, k],]

            bias = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)
            var = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)

            # loop through all of the available interactions and process them appropriately

            # here we are simply implementing the math of our conditional update rule

            for active, interaction, state in zip(active_flags, interactions, states):
                if isinstance(interaction, LinearInteraction):
                    # if there are tail nodes, contribute w * x_1 * x_2 * ..., otherwise contribute w
                    state_prod = jnp.array(1.0)
                    if len(state) > 0:
                        state_prod = jnp.prod(jnp.stack(state, -1), -1)
                    bias -= jnp.sum(interaction.weights * active * state_prod, axis=-1)

                if isinstance(interaction, QuadraticInteraction):
                    # this just sets the variance of the output distribution
                    # there should never be any tail nodes

                    var = active * interaction.inverse_weights
                    var = var[..., 0]  # there should only ever be one

            return (jnp.sqrt(var) * jax.random.normal(key, output_sd.shape)) + (bias * var), sampler_state

        def init(self) -> _SamplerState:
            return None

    print("Defined GaussianSampler")

    # === Cell 23: Create sampling program ===
    # our three types of factor
    lin_fac = LinearFactor(b_vec, Block(all_nodes))
    quad_fac = QuadraticFactor(1 / cov_inv_diag, Block(all_nodes))
    pair_quad_fac = CouplingFactor(cov_inv_off_diag, (Block(edges[0]), Block(edges[1])))

    # an instance of our conditional sampler
    sampler = GaussianSampler()

    # the sampling program itself. Combines the three main components we just built
    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        # one sampler for every free block in gibbs_spec
        samplers=[sampler, sampler],
        factors=[lin_fac, quad_fac, pair_quad_fac],
        other_interaction_groups=[],
    )

    print("Created FactorSamplingProgram")

    # === Cell 25: Equivalent BlockSamplingProgram ===
    groups = []
    for fac in [lin_fac, quad_fac, pair_quad_fac]:
        groups += fac.to_interaction_groups()

    prog_2 = BlockSamplingProgram(gibbs_spec=spec, samplers=[sampler, sampler], interaction_groups=groups)

    print("Created equivalent BlockSamplingProgram")

    # === Cell 27: Set up moment observer ===
    # we will estimate the covariances for each pair of nodes connected by an edge and compare against theory
    # to do this we will need to estimate first moments and second moments
    second_moments = [(e1, e2) for e1, e2 in zip(*edges)]
    first_moments = [[(x,) for x in y] for y in edges]

    # this will accumulate products of the node state specified by first_moments and second_moments
    observer = MomentAccumulatorObserver(first_moments + [second_moments])

    print("Set up moment observer")

    # === Cell 29: Set up sampling ===
    # how many parallel sampling chains will we run?
    n_batches = 1000

    schedule = SamplingSchedule(
        # how many iterations to do before drawing the first sample
        n_warmup=0,
        # how many samples to draw in total
        n_samples=10000,
        # how many steps to take between samples
        steps_per_sample=5,
    )

    # construct the initial state of the iterative sampling algorithm
    init_state = []
    for block in spec.free_blocks:
        key, subkey = jax.random.split(key, 2)
        init_state.append(
            0.1
            * jax.random.normal(
                subkey,
                (
                    n_batches,
                    len(block.nodes),
                ),
            )
        )

    # RNG keys to use for each chain in the batch
    keys = jax.random.split(key, n_batches)

    # memory to hold our moment values
    init_mem = observer.init()

    print(f"Sampling setup: {n_batches} chains, {schedule.n_samples} samples")

    # === Cell 31: Run sampling ===
    print("Running sampling (compiling with JIT and vmap)...")
    # we use vmap to run a bunch of parallel sampling chains
    moments, _ = jax.vmap(lambda k, s: sample_with_observation(k, prog, schedule, s, [], init_mem, observer))(
        keys, init_state
    )

    # Take a mean over the batch axis and divide by the total number of samples
    moments = jax.tree.map(lambda x: jnp.mean(x, axis=0) / schedule.n_samples, moments)

    # compute the covariance values from the moment data
    covariances = moments[-1] - (moments[0] * moments[1])

    print("Sampling complete")

    # === Cell 33: Validate ===
    cov = np.linalg.inv(inv_cov_mat)

    node_map = dict(zip(all_nodes, list(range(len(all_nodes)))))

    real_covs = []

    for edge in zip(*edges):
        real_covs.append(cov[node_map[edge[0]], node_map[edge[1]]])

    real_covs = np.array(real_covs)

    error = np.max(np.abs(real_covs - covariances)) / np.abs(np.max(real_covs))

    print(f"Validation error: {error:.6f}")
    assert error < 0.01, f"Error {error} exceeds threshold 0.01"
    print("✓ Validation PASSED")

    # Visualize validation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_covs, covariances, alpha=0.6)
    ax.plot([real_covs.min(), real_covs.max()], [real_covs.min(), real_covs.max()], "r--", label="y=x")
    ax.set_xlabel("Theoretical Covariance")
    ax.set_ylabel("Estimated Covariance")
    ax.set_title(f"Covariance Validation (error={error:.6f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "covariance_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Summary ===
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Executed all notebook cells in order")
    print("✓ Created custom factors and samplers for Gaussian PGM")
    print(f"✓ Ran {n_batches} parallel Gibbs chains")
    print(f"✓ Validated covariance estimates (error: {error:.6f})")
    print("")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
