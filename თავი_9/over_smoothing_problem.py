import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1. პატარა გრაფის შექმნა
# -----------------------------
G = nx.Graph()

edges = [
    (0, 1), (0, 2),
    (1, 3), (1, 4),
    (2, 5), (2, 6),
    (3, 7), (4, 8),
    (5, 9)
]

G.add_edges_from(edges)

pos = nx.spring_layout(G, seed=42)

# -----------------------------
# 2. საწყისი node features
# თითო node-ს განსხვავებული მნიშვნელობა აქვს
# -----------------------------
num_nodes = G.number_of_nodes()
features = np.random.RandomState(42).rand(num_nodes, 2)

# -----------------------------
# 3. ნორმალიზებული adjacency matrix self-loop-ებით
# ეს ჰგავს GCN aggregation-ს
# -----------------------------
A = nx.to_numpy_array(G)
A = A + np.eye(num_nodes)  # self-loops

D = np.diag(A.sum(axis=1))
D_inv = np.linalg.inv(D)

# Mean aggregation matrix
P = D_inv @ A

# -----------------------------
# 4. Over-smoothing-ის სიმულაცია
# ყოველ layer-ზე:
# H^(l+1) = P H^(l)
# -----------------------------
embeddings_by_layer = [features]

H = features.copy()
num_layers = 100

for _ in range(num_layers):
    H = P @ H
    embeddings_by_layer.append(H.copy())

# -----------------------------
# 5. ვხატავთ embedding-ებს სხვადასხვა layer-ზე
# -----------------------------
layers_to_plot = [0, 1, 2, 5, 10, 20]

plt.figure(figsize=(15, 8))

for i, layer in enumerate(layers_to_plot):
    H_layer = embeddings_by_layer[layer]

    plt.subplot(2, 3, i + 1)

    node_colors = H_layer[:, 0]  # პირველი feature-ს მიხედვით ფერი

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=800,
        font_color="white",
        font_weight="bold",
        edge_color="gray"
    )

    plt.title(f"Layer {layer}")

plt.suptitle("Over-smoothing demonstration in a small graph", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. ვზომავთ embedding-ების მსგავსებას
# საშუალო მანძილი node embedding-ებს შორის
# -----------------------------
avg_distances = []

for H_layer in embeddings_by_layer:
    distances = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(H_layer[i] - H_layer[j])
            distances.append(dist)

    avg_distances.append(np.mean(distances))

print("Layer 0")
print(embeddings_by_layer[0])

print("\nLayer 20")
print(embeddings_by_layer[20])
plt.figure(figsize=(8, 5))
plt.plot(range(num_layers + 1), avg_distances, marker="o")
plt.xlabel("Number of GNN aggregation layers")
plt.ylabel("Average distance between node embeddings")
plt.title("Node embeddings become more similar after many layers")
plt.grid(True)
plt.show()