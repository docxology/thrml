"""Network and graph visualization for probabilistic graphical models."""

from typing import Any, Dict, List, Optional

from .core import create_figure, ensure_matplotlib, ensure_networkx, get_config, save_figure


def plot_graphical_model(
    nodes: List[Any],
    edges: List[tuple],
    node_labels: Optional[Dict] = None,
    node_types: Optional[Dict] = None,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    layout: str = "spring",
    **kwargs,
):
    """Plot a probabilistic graphical model structure.

    **Arguments:**

    - `nodes`: List of nodes
    - `edges`: List of edges (node pairs)
    - `node_labels`: Optional dict mapping nodes to labels
    - `node_types`: Optional dict mapping nodes to types for coloring
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `layout`: Layout algorithm ('spring', 'circular', 'kamada_kawai')
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    nx = ensure_networkx()

    fig, ax = create_figure(figsize=figsize or get_config().large_figsize)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))

    # Add edges
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges_idx = [(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges]
    G.add_edges_from(edges_idx)

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Node colors based on type
    if node_types:
        unique_types = list(set(node_types.values()))
        color_map = {t: i for i, t in enumerate(unique_types)}
        node_colors = [color_map.get(node_types.get(nodes[i], "default"), 0) for i in range(len(nodes))]
    else:
        node_colors = [get_config().primary_color] * len(nodes)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax, cmap="tab10")
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    if node_labels:
        labels = {i: node_labels.get(nodes[i], str(i)) for i in range(len(nodes))}
    else:
        labels = {i: str(i) for i in range(len(nodes))}

    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

    ax.set_title("Graphical Model Structure")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="networks", close=False)

    return fig, ax


def plot_factor_graph(
    variable_nodes: List[Any],
    factor_nodes: List[Any],
    connections: List[tuple],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot a factor graph with variable and factor nodes.

    **Arguments:**

    - `variable_nodes`: List of variable nodes
    - `factor_nodes`: List of factor nodes
    - `connections`: List of (variable, factor) connections
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    nx = ensure_networkx()

    fig, ax = create_figure(figsize=figsize or get_config().large_figsize)

    # Create bipartite graph
    G = nx.Graph()

    # Add nodes with bipartite attribute
    for i, v in enumerate(variable_nodes):
        G.add_node(f"V{i}", bipartite=0)
    for i, f in enumerate(factor_nodes):
        G.add_node(f"F{i}", bipartite=1)

    # Add edges
    for v_idx, f_idx in connections:
        G.add_edge(f"V{v_idx}", f"F{f_idx}")

    # Bipartite layout
    pos = nx.bipartite_layout(G, [f"V{i}" for i in range(len(variable_nodes))])

    # Draw variable nodes (circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[f"V{i}" for i in range(len(variable_nodes))],
        node_color=get_config().primary_color,
        node_shape="o",
        node_size=500,
        alpha=0.8,
        ax=ax,
    )

    # Draw factor nodes (squares)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[f"F{i}" for i in range(len(factor_nodes))],
        node_color=get_config().secondary_color,
        node_shape="s",
        node_size=500,
        alpha=0.8,
        ax=ax,
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title("Factor Graph")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="networks", close=False)

    return fig, ax


def plot_interaction_graph(
    blocks: List[Any],
    interactions: List[Any],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot THRML interaction graph showing block connections.

    **Arguments:**

    - `blocks`: List of THRML blocks
    - `interactions`: List of interaction groups
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    nx = ensure_networkx()

    fig, ax = create_figure(figsize=figsize or get_config().large_figsize)

    G = nx.Graph()

    # Add blocks as nodes
    for i, block in enumerate(blocks):
        G.add_node(i, size=len(block.nodes))

    # Add interactions as edges
    for interaction in interactions:
        head_nodes = interaction.head_nodes if hasattr(interaction, "head_nodes") else []
        tail_nodes = interaction.tail_nodes if hasattr(interaction, "tail_nodes") else []

        # Connect blocks involved in interaction
        # This is simplified - actual implementation depends on THRML structure
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                G.add_edge(i, j, weight=1)

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Node sizes proportional to block size
    node_sizes = [G.nodes[i].get("size", 10) * 100 for i in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=get_config().primary_color, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    ax.set_title("Block Interaction Graph")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="networks", close=False)

    return fig, ax


def plot_generative_model_structure(
    n_observations: int,
    n_states: int,
    n_actions: int,
    show_matrices: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot structure of a generative model (POMDP).

    **Arguments:**

    - `n_observations`: Number of observation states
    - `n_states`: Number of hidden states
    - `n_actions`: Number of actions
    - `show_matrices`: Whether to show matrix dimensions
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()

    fig, ax = create_figure(figsize=figsize or get_config().wide_figsize)

    # Define node positions
    positions = {
        "prior": (0.2, 0.5),
        "state_t": (0.4, 0.5),
        "state_t+1": (0.6, 0.5),
        "obs_t": (0.4, 0.2),
        "action": (0.5, 0.8),
    }

    # Draw nodes
    for name, pos in positions.items():
        circle = plt.Circle(pos, 0.05, color=get_config().primary_color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(
            pos[0], pos[1], name.replace("_", "\n"), ha="center", va="center", fontsize=8, color="white", weight="bold"
        )

    # Draw arrows
    arrows = [
        ("prior", "state_t", "D"),
        ("state_t", "obs_t", f"A ({n_observations}×{n_states})" if show_matrices else "A"),
        ("state_t", "state_t+1", f"B ({n_states}×{n_states}×{n_actions})" if show_matrices else "B"),
        ("action", "state_t+1", ""),
    ]

    for start, end, label in arrows:
        start_pos = positions[start]
        end_pos = positions[end]

        ax.annotate("", xy=end_pos, xytext=start_pos, arrowprops=dict(arrowstyle="->", lw=2, color="black", alpha=0.6))

        if label:
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("Generative Model Structure (POMDP)")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="networks", close=False)

    return fig, ax


def plot_markov_blanket(
    node_idx: int,
    nodes: List[Any],
    edges: List[tuple],
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """Plot the Markov blanket of a node.

    **Arguments:**

    - `node_idx`: Index of the node to show Markov blanket for
    - `nodes`: List of all nodes
    - `edges`: List of edges
    - `figsize`: Figure size
    - `save_path`: If provided, save figure
    - `**kwargs`: Additional plotting options

    **Returns:**

    - Figure and axes
    """
    plt = ensure_matplotlib()
    nx = ensure_networkx()

    fig, ax = create_figure(figsize=figsize or get_config().large_figsize)

    # Create graph
    G = nx.Graph()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    G.add_nodes_from(range(len(nodes)))
    edges_idx = [(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges]
    G.add_edges_from(edges_idx)

    # Find Markov blanket (parents, children, and co-parents)
    neighbors = set(G.neighbors(node_idx))
    co_parents = set()
    for neighbor in neighbors:
        co_parents.update(G.neighbors(neighbor))

    markov_blanket = neighbors.union(co_parents)
    markov_blanket.discard(node_idx)

    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Node colors
    node_colors = []
    for i in range(len(nodes)):
        if i == node_idx:
            node_colors.append("red")
        elif i in markov_blanket:
            node_colors.append("orange")
        else:
            node_colors.append("lightgray")

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="Target Node"),
        Patch(facecolor="orange", label="Markov Blanket"),
        Patch(facecolor="lightgray", label="Other Nodes"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title(f"Markov Blanket of Node {node_idx}")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, category="networks", close=False)

    return fig, ax
