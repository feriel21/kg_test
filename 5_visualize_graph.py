import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
INPUT = "output_graph/final_graph_knowledge_layer.gexf"
OUT = "output_graph/visuals"
os.makedirs(OUT, exist_ok=True)

print("Loading graph…")
G = nx.read_gexf(INPUT)

# Fix classes
for n in G.nodes():
    if G.nodes[n].get("class") is None:
        G.nodes[n]["class"] = "OTHER"

# Keep largest component
UG = G.to_undirected()
largest = max(nx.connected_components(UG), key=len)
G = G.subgraph(largest).copy()

print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

# ============================================
# COLOR MAPS
# ============================================
CLASS_COLORS = {
    "PROCESS":   "#E74C3C",
    "LOCATION":  "#3498DB",
    "FEATURE":   "#1ABC9C",
    "FACIES":    "#E67E22",
    "TRIGGER":   "#8E44AD",
    "MATERIAL":  "#8B4513",
    "OTHER":     "#7F8C8D"
}

# DISTINCT edge relation palette (10-color tableau)
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC"
]

RELATIONS = sorted({G.edges[u, v].get("label", "related_to") for u, v in G.edges()})
REL_COLORS = {rel: PALETTE[i % len(PALETTE)] for i, rel in enumerate(RELATIONS)}

# ============================================
# DEGREE & NODE SIZE
# ============================================
degree_dict = dict(G.degree())
max_degree = max(degree_dict.values())

node_colors = [
    CLASS_COLORS.get(G.nodes[n].get("class", "OTHER"), "#7F8C8D")
    for n in G.nodes()
]

node_sizes = [
    800 + (degree_dict[n] / max_degree) * 9000
    for n in G.nodes()
]

# ============================================
# LAYOUT – HIGH REPELLING (AÉRATION MAXIMALE)
# ============================================
print("Computing high-repulsion layout…")

pos = nx.spring_layout(
    G,
    k=90,  
    iterations=600,
    seed=42,
    scale=1000
)

# ============================================
# FULL GRAPH VISUALIZATION
# ============================================
plt.figure(figsize=(46, 40))

edge_colors = [REL_COLORS[G.edges[u, v].get("label", "related_to")] for u, v in G.edges()]

nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=1.0,
    alpha=0.35
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="white",
    linewidths=1.2,
    alpha=0.97
)

for n, (x, y) in pos.items():
    plt.text(
        x, y, n,
        fontsize=16,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            pad=0.30
        )
    )

# Legends
# Node classes
for cls, color in CLASS_COLORS.items():
    plt.scatter([], [], color=color, label=cls, s=300)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=20,
    title="Node Classes",
    title_fontsize=24
)

# Edge relations
for rel, col in REL_COLORS.items():
    plt.plot([], [], color=col, linewidth=4, label=rel)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 0.48),
    fontsize=18,
    title="Edge Relations",
    title_fontsize=22
)

plt.title("Geological Knowledge Graph — Full Labels & Colored Relations",
          fontsize=34, fontweight="bold")

plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/full_graph_labels_relations.png", dpi=550, bbox_inches="tight")
plt.close()

print(" full_graph_labels_relations.png saved")

# ======================================================
# 2) DEGREE DISTRIBUTION
# ======================================================
plt.figure(figsize=(12, 7))
plt.hist(list(degree_dict.values()), bins=20, color="#9B59B6", alpha=0.8)
plt.title("Degree Distribution", fontsize=20)
plt.xlabel("Degree (# Relations)", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/degree_distribution.png", dpi=350)
plt.close()

print(" degree_distribution.png saved")

# ======================================================
# 3) TOP HUBS
# ======================================================
sorted_deg = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]
nodes_top, deg_top = zip(*sorted_deg)

plt.figure(figsize=(12, 8))
plt.barh(nodes_top[::-1], deg_top[::-1], color="#2980B9")
plt.title("Top 20 Most Connected Geological Concepts", fontsize=20)
plt.xlabel("Degree (# Relations)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUT}/top_hubs.png", dpi=350)
plt.close()

print(" top_hubs.png saved")

# ======================================================
# 4) PARETO CURVE
# ======================================================
sorted_all = sorted(degree_dict.values(), reverse=True)
cum = np.cumsum(sorted_all) / sum(sorted_all)

plt.figure(figsize=(12, 7))
plt.plot(cum, color="#C0392B", linewidth=3)
plt.axhline(0.80, color="black", linestyle="--")
plt.axvline(int(len(cum) * 0.20), color="black", linestyle="--")
plt.title("Pareto Curve — 80/20 Structure", fontsize=20)
plt.xlabel("Top X% Nodes", fontsize=16)
plt.ylabel("Cumulative Contribution", fontsize=16)
plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/pareto_curve.png", dpi=350)
plt.close()

print(" pareto_curve.png saved")

# ======================================================
# 5) SUBGRAPH OF TOP 25 GEO-NODES
#    → Processes / Features / Trigger / Facies / Location / Material
#    → Colored edges by relation type + legend
#    → Colored nodes by geological class + legend
# ======================================================

TARGET_CLASSES = {"PROCESS", "FEATURE", "TRIGGER", "LOCATION", "FACIES", "MATERIAL"}

# --- Filter geologic nodes ---
filtered_nodes = [
    n for n in G.nodes()
    if G.nodes[n].get("class", "OTHER") in TARGET_CLASSES
]

# --- Sort by degree (importance) ---
sorted_deg_filtered = sorted(
    [(n, degree_dict[n]) for n in filtered_nodes],
    key=lambda x: x[1],
    reverse=True
)

# --- Select top 25 ---
top25 = [n for n, _ in sorted_deg_filtered[:25]]

# Subgraph
G_small = G.subgraph(top25).copy()

print(f"Top-25 subgraph: {len(G_small.nodes())} nodes, {len(G_small.edges())} edges")

# Layout
pos_small = nx.spring_layout(
    G_small,
    k=2.2,
    iterations=350,
    seed=42
)

# Sizes
node_sizes_small = [
    900 + (degree_dict[n] / max_degree) * 9000
    for n in G_small.nodes()
]

# Colors (class-based)
node_colors_small = [
    CLASS_COLORS.get(G_small.nodes[n].get("class", "OTHER"), "#7F8C8D")
    for n in G_small.nodes()
]

# Edge colors
edge_colors_small = [
    REL_COLORS[G_small.edges[u, v].get("label", "related_to")]
    for u, v in G_small.edges()
]

# ======================================================
# PLOT
# ======================================================
plt.figure(figsize=(30, 26))

# Edges
nx.draw_networkx_edges(
    G_small, pos_small,
    edge_color=edge_colors_small,
    width=2.3,
    alpha=0.85
)

# Nodes
nx.draw_networkx_nodes(
    G_small, pos_small,
    node_size=node_sizes_small,
    node_color=node_colors_small,
    edgecolors="white",
    linewidths=1.5,
    alpha=0.96
)

# Labels (big bold)
nx.draw_networkx_labels(
    G_small, pos_small,
    font_size=18,
    font_weight="bold",
    font_color="black"
)

# ======================================================
# LEGENDS
# ======================================================

# --- Legend 1 : Edge relations ---
for rel, col in REL_COLORS.items():
    plt.plot([], [], color=col, linewidth=5, label=rel)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=15,
    title="Edge Relations",
    title_fontsize=18
)

# --- Legend 2 : Node classes (geoscience) ---
for cls in TARGET_CLASSES:
    plt.scatter([], [], color=CLASS_COLORS[cls], s=350, label=cls)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 0.55),
    fontsize=15,
    title="Node Classes",
    title_fontsize=18
)

# ======================================================
# SAVE
# ======================================================
plt.title("Top 25 Geological Concepts — Processes, Features, Triggers, Facies, Location, Material",
          fontsize=30, fontweight="bold")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{OUT}/subgraph_top25_geo.png", dpi=520, bbox_inches="tight")
plt.close()

print(" subgraph_top25_geo.png saved (geoscience-aware)")
