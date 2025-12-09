#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 4 — GEOSCIENCE-AWARE GRAPH VISUALIZATION (STYLE PAULINE LE BOUTEILLER)

Produces:
    - Full graph (clean, minimal, color-coded, readable)
    - One subgraph per main category for comparison
    - Additional bar charts / heatmaps (separate from graph)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import os

from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

INPUT = paths["graph_knowledge"]
OUT = paths["evaluation_plots"]
os.makedirs(OUT, exist_ok=True)

print("Loading enriched graph...")
G = nx.read_gexf(INPUT)

# --------------------------------------------------------
# FIX CATEGORIES (STEP2 categories, enforced)
# --------------------------------------------------------
VALID_CATS = {
    "MTD_descriptors",
    "Mass_movement_properties",
    "Environmental_controls",
    "UNCLASSIFIED",
}

for n, data in G.nodes(data=True):
    cat = data.get("category", "UNCLASSIFIED")
    if cat not in VALID_CATS:
        cat = "UNCLASSIFIED"
    data["category"] = cat

# --------------------------------------------------------
# COLORS — Pauline style
# --------------------------------------------------------
CAT_COLORS = {
    "MTD_descriptors": "#1f78b4",      # blue
    "Mass_movement_properties": "#e31a1c",   # red
    "Environmental_controls": "#33a02c",      # green
    "UNCLASSIFIED": "#bdbdbd",                # grey
}

REL_COLORS = {
    "FORMS": "#e31a1c",         # red
    "INFLUENCES": "#33a02c",    # green
    "SHAPES": "#1f78b4",        # blue
    "RESPONDS_TO": "#fb9a99",   # pink
    "RELATED_TO": "#636363",    # grey
}

# --------------------------------------------------------
# KEEP LARGEST COMPONENT
# --------------------------------------------------------
UG = G.to_undirected()
largest = max(nx.connected_components(UG), key=len)
G = G.subgraph(largest).copy()

print(f"Graph nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

# --------------------------------------------------------
# LAYOUT — Clean and airy (similar to Pauline’s publication)
# --------------------------------------------------------
pos = nx.spring_layout(
    G,
    k=65,              # high repulsion
    iterations=500,
    seed=42,
)

# Node sizes by degree
deg = dict(G.degree())
max_deg = max(deg.values()) if deg else 1
node_sizes = [700 + (deg[n] / max_deg) * 5000 for n in G.nodes()]

# Node colors by category
node_colors = [CAT_COLORS[G.nodes[n]["category"]] for n in G.nodes()]

# Edge colors
edge_colors = [
    REL_COLORS.get(G.edges[u, v].get("label", "RELATED_TO"), "#636363")
    for u, v in G.edges()
]

# --------------------------------------------------------
# FULL GRAPH (STYLE PAULINE)
# --------------------------------------------------------
plt.figure(figsize=(48, 40))

nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=1.2,
    alpha=0.35
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="black",
    linewidths=0.8,
    alpha=0.95
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=18,
    font_weight="bold",
    font_color="black"
)

plt.title("Full Geological Knowledge Graph (Pauline-style)", fontsize=40, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/full_graph_pauline_style.png", dpi=520)
plt.close()
print("✔ Saved: full_graph_pauline_style.png")

# --------------------------------------------------------
# POSTER SUBGRAPH — ONE SINGLE CLEAN, READABLE GRAPH
# --------------------------------------------------------
print("Building poster-ready subgraph...")

# Centrality metrics
deg = dict(G.degree())
num_nodes = G.number_of_nodes()
bet = nx.betweenness_centrality(
    G,
    k=min(200, max(10, num_nodes)),  # min(200, nb de nodes), mais au moins 10
    normalized=True,
    seed=42
)


freq = {n: G.nodes[n].get("frequency", 1) for n in G.nodes()}

def get_top_nodes(category, k=10):
    nodes = [n for n in G.nodes() if G.nodes[n]["category"] == category]
    if not nodes:
        return []
    scored = [
        (n, 0.45*deg[n] + 0.35*freq[n] + 0.20*bet[n])
        for n in nodes
    ]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored_sorted[:k]]

# Select 10 best from each category
top_desc = get_top_nodes("MTD_descriptors", 10)
top_prop = get_top_nodes("Mass_movement_properties", 10)
top_ctrl = get_top_nodes("Environmental_controls", 10)

poster_nodes = set(top_desc + top_prop + top_ctrl)

# Build subgraph
PG = G.subgraph(poster_nodes).copy()

# Remove isolated nodes
PG.remove_nodes_from([n for n, d in PG.degree() if d == 0])

print(f"Poster subgraph → {PG.number_of_nodes()} nodes | {PG.number_of_edges()} edges")

# Layout
pos_p = nx.spring_layout(PG, k=4.5, iterations=350, seed=42)

# Node styling
node_colors_p = [CAT_COLORS[PG.nodes[n]["category"]] for n in PG.nodes()]
node_sizes_p = [
    1600 + (deg[n] / max(deg.values())) * 8000
    for n in PG.nodes()
]

# Edge colors
edge_colors_p = [
    REL_COLORS.get(PG.edges[u, v].get("label", "RELATED_TO"), "#636363")
    for u, v in PG.edges()
]

plt.figure(figsize=(40, 32))

nx.draw_networkx_edges(
    PG, pos_p,
    width=3.0,
    alpha=0.65,
    edge_color=edge_colors_p
)

nx.draw_networkx_nodes(
    PG, pos_p,
    node_size=node_sizes_p,
    node_color=node_colors_p,
    edgecolors="black",
    linewidths=1.5,
    alpha=0.98
)

nx.draw_networkx_labels(
    PG, pos_p,
    font_size=22,
    font_weight="bold",
    font_color="black"
)

plt.title("Poster Subgraph — Key Geological Concepts & Relations",
          fontsize=38, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/poster_subgraph.png", dpi=520)
plt.close()

print("✔ Saved poster_subgraph.png")

# --------------------------------------------------------
# EXTRA FIGURES (BAR CHARTS & HEATMAPS)
# --------------------------------------------------------
# Frequency bar chart
freq = {n: G.nodes[n].get("frequency", 1) for n in G.nodes()}
top30 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30]

names, values = zip(*top30)
plt.figure(figsize=(16, 12))
plt.barh(names[::-1], values[::-1], color="#1f78b4")
plt.title("Top 30 Concepts by Frequency", fontsize=24)
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT}/freq_top30.png", dpi=350)
plt.close()

# Category distribution
cats = [G.nodes[n]["category"] for n in G.nodes()]
count_map = {c: cats.count(c) for c in VALID_CATS}

plt.figure(figsize=(10,7))
plt.bar(count_map.keys(), count_map.values(), color=[CAT_COLORS[c] for c in count_map.keys()])
plt.title("Category Distribution", fontsize=24)
plt.tight_layout()
plt.savefig(f"{OUT}/category_distribution.png", dpi=350)
plt.close()

# Category-to-category relations heatmap
mat = np.zeros((len(VALID_CATS), len(VALID_CATS)))
cat_list = list(VALID_CATS)

for u, v in G.edges():
    cu = G.nodes[u]["category"]
    cv = G.nodes[v]["category"]
    i = cat_list.index(cu)
    j = cat_list.index(cv)
    mat[i][j] += 1

plt.figure(figsize=(12,10))
sns.heatmap(mat, annot=True, cmap="coolwarm", xticklabels=cat_list, yticklabels=cat_list)
plt.title("Category-to-Category Relation Matrix", fontsize=24)
plt.tight_layout()
plt.savefig(f"{OUT}/category_relation_heatmap.png", dpi=350)
plt.close()

print("🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY.")
