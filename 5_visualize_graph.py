import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np

INPUT = "output_graph/final_graph_knowledge_layer.gexf"
OUT = "output_graph/visuals"
os.makedirs(OUT, exist_ok=True)

print("Loading graph...")
G = nx.read_gexf(INPUT)

# Keep only largest component
UG = G.to_undirected()
largest = max(nx.connected_components(UG), key=len)
G = G.subgraph(largest).copy()

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Compute degrees
degree_dict = dict(G.degree())
max_degree = max(degree_dict.values())

# Scale node sizes
node_sizes = [300 + (deg / max_degree) * 4000 for deg in degree_dict.values()]

# Compute layout
pos = nx.spring_layout(G, k=0.45, seed=42)

# ----------------------------------------------------
# 1) GRAPH WITHOUT LABELS (BIG NODES)
# ----------------------------------------------------
plt.figure(figsize=(15, 14))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color="skyblue",
    alpha=0.85,
    linewidths=0.3,
    edgecolors="black"
)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.15)

plt.title("Knowledge Graph — Scaled Node Sizes (Degree)", fontsize=20)
plt.axis("off")
plt.savefig(f"{OUT}/graph_scaled_nodes.png", dpi=400)
plt.close()


# ----------------------------------------------------
# 2) GRAPH WITH LABELS (ONLY IMPORTANT NODES)
# ----------------------------------------------------
important_nodes = [n for n, d in G.degree() if d >= (0.30 * max_degree)]
label_dict = {n: n for n in important_nodes}

plt.figure(figsize=(16, 15))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color="lightgreen",
    alpha=0.90,
    edgecolors="black"
)
nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.25)
nx.draw_networkx_labels(
    G, pos,
    labels=label_dict,
    font_size=10,
    font_weight="bold"
)

plt.title("Knowledge Graph — Key Nodes With Labels", fontsize=20)
plt.axis("off")
plt.savefig(f"{OUT}/graph_with_labels_scaled.png", dpi=400)
plt.close()


# ----------------------------------------------------
# 3) DEGREE DISTRIBUTION HISTOGRAM
# ----------------------------------------------------
degrees = list(degree_dict.values())

plt.figure(figsize=(12, 7))
plt.hist(degrees, bins=30, color="purple", alpha=0.7)
plt.title("Degree Distribution", fontsize=18)
plt.xlabel("Degree (# Connections)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/degree_distribution.png", dpi=300)
plt.close()


# ----------------------------------------------------
# 4) TOP HUBS BAR CHART
# ----------------------------------------------------
sorted_deg = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]
nodes_top, deg_top = zip(*sorted_deg)

plt.figure(figsize=(12, 8))
plt.barh(nodes_top[::-1], deg_top[::-1], color="steelblue")
plt.title("Top 20 Most Connected Geological Concepts", fontsize=18)
plt.xlabel("Degree (# Relations)")
plt.tight_layout()
plt.savefig(f"{OUT}/top_hubs.png", dpi=300)
plt.close()


# ----------------------------------------------------
# 5) PARETO PLOT (80/20 structure)
# ----------------------------------------------------
sorted_all = sorted(degree_dict.values(), reverse=True)
cum = np.cumsum(sorted_all) / sum(sorted_all)

plt.figure(figsize=(12, 7))
plt.plot(cum, color="darkred", linewidth=3)
plt.axhline(0.80, color="black", linestyle="--")
plt.axvline(int(len(cum)*0.20), color="black", linestyle="--")
plt.title("Pareto Curve — Structural Importance of Nodes", fontsize=18)
plt.xlabel("Top X% Nodes")
plt.ylabel("Cumulative % of Total Degree")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/pareto_curve.png", dpi=300)
plt.close()

print("\n✔ ALL VISUALS GENERATED:")
print(" - graph_scaled_nodes.png")
print(" - graph_with_labels_scaled.png")
print(" - degree_distribution.png")
print(" - top_hubs.png")
print(" - pareto_curve.png")
