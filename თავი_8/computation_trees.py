import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. Original graph
# -----------------------------
G = nx.Graph()

edges = [
    (1, 2),
    (1, 5),
    (2, 5),
    (5, 4),
    (4, 3)
]

G.add_edges_from(edges)

pos = {
    1: (0, 2),
    2: (2, 2),
    5: (0, 0),
    4: (2, 0),
    3: (3, 1)
}

# -----------------------------
# 2. Build computation tree
# -----------------------------
def build_computation_tree(G, root, depth=2):
    T = nx.DiGraph()
    labels = {}
    counter = 0

    def add_layer(node, level, parent_id=None):
        nonlocal counter

        current_id = counter
        counter += 1

        T.add_node(current_id)
        labels[current_id] = str(node)

        if parent_id is not None:
            T.add_edge(current_id, parent_id)

        if level < depth:
            for nbr in sorted(G.neighbors(node)):
                add_layer(nbr, level + 1, current_id)

        return current_id

    add_layer(root, 0)
    return T, labels


# -----------------------------
# 3. Tree layout
# -----------------------------
def hierarchy_pos(T, root, width=1.0, vert_gap=1.0, vert_loc=0, xcenter=0.5):
    children = list(T.predecessors(root))

    if len(children) == 0:
        return {root: (xcenter, vert_loc)}

    dx = width / len(children)
    next_x = xcenter - width / 2 - dx / 2

    pos = {root: (xcenter, vert_loc)}

    for child in children:
        next_x += dx
        pos.update(
            hierarchy_pos(
                T,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=next_x
            )
        )

    return pos


# -----------------------------
# 4. Draw everything
# -----------------------------
fig = plt.figure(figsize=(18, 9))

# Original graph
ax = plt.subplot2grid((2, 5), (0, 0), colspan=2)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1300,
    node_color="gold",
    edge_color="black",
    font_size=16,
    font_weight="bold",
    ax=ax
)
ax.set_title("Original Graph", fontsize=18)
ax.axis("off")

# Computational graphs for all nodes
for idx, node in enumerate(sorted(G.nodes())):
    ax = plt.subplot2grid((2, 5), (1, idx))

    T, labels = build_computation_tree(G, node, depth=2)

    root = 0
    tree_pos = hierarchy_pos(T, root, width=2.0, vert_gap=1.2)

    nx.draw_networkx_edges(
        T,
        tree_pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="black",
        ax=ax
    )

    nx.draw_networkx_nodes(
        T,
        tree_pos,
        node_size=700,
        node_color="gold",
        edgecolors="black",
        ax=ax
    )

    nx.draw_networkx_labels(
        T,
        tree_pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    ax.set_title(f"Computation Graph of Node {node}", fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.show()