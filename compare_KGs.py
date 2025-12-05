import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer, util
from matplotlib_venn import venn2

# -----------------------------------------------------------
# CONFIG (dynamic, no hard-coded paths)
# -----------------------------------------------------------
from utils.config_loader import load_config
cfg = load_config()

paths = cfg["paths_expanded"]
DATASET = cfg["dataset"]  # "small_corpus" or "full_corpus"

KG_FULL = paths["graph_modular_full"]      # full corpus KG
KG_SMALL = paths["graph_modular_small"]    # small corpus KG
OUT_DIR = f"comparison/{DATASET}"

os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

print(">> Loading Knowledge Graphs...")
G_full = nx.read_gexf(KG_FULL)
G_small = nx.read_gexf(KG_SMALL)

nodes_full = set(G_full.nodes())
nodes_small = set(G_small.nodes())

# -----------------------------------------------------------
# 1. NODE OVERLAP
# -----------------------------------------------------------
shared_nodes = nodes_full.intersection(nodes_small)
unique_full = nodes_full - nodes_small
unique_small = nodes_small - nodes_full

coverage_small = len(shared_nodes) / len(nodes_small)
coverage_full = len(shared_nodes) / len(nodes_full)

# -----------------------------------------------------------
# 2. DEGREE DISTRIBUTION
# -----------------------------------------------------------
degrees_full = [d for _, d in G_full.degree()]
degrees_small = [d for _, d in G_small.degree()]

# -----------------------------------------------------------
# 3. HUB COMPARISON (Top-20)
# -----------------------------------------------------------
def top_hubs(G, n=20):
    deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return deg[:n]

top_full = top_hubs(G_full)
top_small = top_hubs(G_small)

# -----------------------------------------------------------
# 4. SEMANTIC SIMILARITY KG_SMALL → KG_FULL
# -----------------------------------------------------------
small_list = list(nodes_small)
full_list = list(nodes_full)

small_emb = model.encode(small_list, convert_to_tensor=True)
full_emb  = model.encode(full_list, convert_to_tensor=True)

sim_matrix = util.cos_sim(small_emb, full_emb).cpu().numpy()

# -----------------------------------------------------------
# 5. SAVE METRICS
# -----------------------------------------------------------
results = {
    "nodes": {
        "full": len(nodes_full),
        "small": len(nodes_small),
        "shared": len(shared_nodes),
        "unique_full": len(unique_full),
        "unique_small": len(unique_small),
        "coverage_small_to_full": float(coverage_small),
        "coverage_full_to_small": float(coverage_full)
    },
    "top_hubs_full": top_full,
    "top_hubs_small": top_small
}

with open(f"{OUT_DIR}/comparison_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(">> Metrics saved.")

# -----------------------------------------------------------
# 6. FIGURES
# -----------------------------------------------------------

# --- Figure 1: Venn diagram ---
plt.figure(figsize=(6, 5))
venn2(subsets=(len(unique_full), len(unique_small), len(shared_nodes)),
     set_labels=("Full KG", "Small KG"))
plt.title("Node Overlap Between KG_full and KG_small")
plt.savefig(f"{OUT_DIR}/venn_nodes.png", dpi=200)
plt.close()

# --- Figure 2: Degree distribution ---
plt.figure(figsize=(7, 5))
sns.kdeplot(degrees_full, label="Full KG", linewidth=2)
sns.kdeplot(degrees_small, label="Small KG", linewidth=2)
plt.title("Degree Distribution Comparison")
plt.xlabel("Node Degree")
plt.legend()
plt.savefig(f"{OUT_DIR}/degree_dist.png", dpi=200)
plt.close()

# --- Figure 3: Coverage bar chart ---
plt.figure(figsize=(6, 4))
plt.bar(["Small→Full", "Full→Small"], [coverage_small, coverage_full],
        color=["#4a90e2", "#50e3c2"])
plt.ylim(0, 1)
plt.title("Ontological Coverage Comparison")
plt.ylabel("Coverage")
plt.savefig(f"{OUT_DIR}/coverage.png", dpi=200)
plt.close()

# --- Figure 4: Top-20 hubs ---
def plot_hubs(top_full, top_small):
    labels_full = [x[0] for x in top_full]
    labels_small = [x[0] for x in top_small]
    values_full = [x[1] for x in top_full]
    values_small = [x[1] for x in top_small]

    plt.figure(figsize=(10, 5))
    plt.barh(labels_full, values_full, color="#4a90e2", alpha=0.7, label="Full KG")
    plt.barh(labels_small, values_small, color="#f5a623", alpha=0.7, label="Small KG")
    plt.gca().invert_yaxis()
    plt.title("Top-20 Hubs Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top20_hubs.png", dpi=200)
    plt.close()

plot_hubs(top_full, top_small)

# --- Figure 5: Similarity heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix[:40, :40], cmap="viridis")
plt.title("Semantic Similarity KG_small → KG_full (First 40 nodes)")
plt.savefig(f"{OUT_DIR}/similarity_heatmap.png", dpi=200)
plt.close()

print(">> All figures generated in:", OUT_DIR)
