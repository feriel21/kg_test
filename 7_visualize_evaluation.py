# ============================================================
# 7_visualize_evaluation_clustered.py
# Full evaluation visualization suite for Geo-KG
# ============================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------
EVAL_FILE = "output_graph/evaluation_results.json"
KG_FILE = "output_graph/final_graph_knowledge_layer.gexf"
OUT = "output_graph/visuals_evaluation"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------
# LOAD EVALUATION DATA
# ------------------------------------------------------------
with open(EVAL_FILE, "r") as f:
    D = json.load(f)

sim_matrix = np.array(D["similarity_matrix"])
ref_terms = D["ref_terms"]
auto_terms = D["auto_terms"]
sim_scores = np.array(D["similarity_scores"])
degrees = np.array(D["degrees"])

coverage_exact = D["coverage"]["exact"]
coverage_sem = D["coverage"]["semantic"]
hall_count = D["hallucinations"]["count"]
redundancy_count = D["redundancy"]["count"]
weak_nodes_count = D["weak_nodes"]["count"]
cohesion_mean = D["cohesion"]["mean"]
cohesion_max = D["cohesion"]["max"]

# ------------------------------------------------------------
# LOAD AUTO-KG (for node classes)
# ------------------------------------------------------------
G = nx.read_gexf(KG_FILE)
node_classes = {n: G.nodes[n].get("class", "OTHER") for n in auto_terms}

CLASSES = ["PROCESS", "FEATURE", "TRIGGER", "FACIES", "LOCATION", "MATERIAL", "OTHER"]
CLASS_COLORS = {
    "PROCESS": "#E74C3C",
    "FEATURE": "#1ABC9C",
    "TRIGGER": "#8E44AD",
    "FACIES": "#E67E22",
    "LOCATION": "#3498DB",
    "MATERIAL": "#8B4513",
    "OTHER": "#7F8C8D"
}

# ------------------------------------------------------------
# Helper: shorten text labels
# ------------------------------------------------------------
def short(t):
    t = t.replace("(", "").replace(")", "")
    return t[:35] + "…" if len(t) > 35 else t

# ============================================================
# 1) GLOBAL HEATMAP (REFERENCE vs AUTO)
# ============================================================

plt.figure(figsize=(28, 16))
sns.heatmap(
    sim_matrix,
    cmap="viridis",
    xticklabels=[short(x) for x in auto_terms],
    yticklabels=[short(x) for x in ref_terms],
    vmin=0, vmax=1,
    cbar_kws={"label": "Cosine Similarity"},
)

plt.xticks(rotation=60, fontsize=9)
plt.yticks(rotation=0, fontsize=11)
plt.title("GLOBAL — Semantic Similarity (Reference → Auto KG)", fontsize=20, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/1_global_heatmap.png", dpi=350)
plt.close()
print("1_global_heatmap.png saved")


# ============================================================
# 2) CLUSTERED HEATMAP (Ward)
# ============================================================

print("Clustering references...")

dist_ref = 1 - sim_matrix  # convert similarity → distance

n_clusters = 6
clust = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward"
).fit(dist_ref)


order = np.argsort(clust.labels_)
sim_clustered = sim_matrix[order, :]
ref_ordered = [ref_terms[i] for i in order]

plt.figure(figsize=(28, 16))
sns.heatmap(
    sim_clustered,
    cmap="viridis",
    xticklabels=[short(x) for x in auto_terms],
    yticklabels=[short(x) for x in ref_ordered],
    vmin=0, vmax=1,
    cbar_kws={"label": "Cosine Similarity"},
)

plt.xticks(rotation=60, fontsize=9)
plt.yticks(rotation=0, fontsize=11)
plt.title("CLUSTERED — Semantic Similarity (Reference grouped by themes)", fontsize=20, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/2_clustered_heatmap.png", dpi=350)
plt.close()
print(" 2_clustered_heatmap.png saved")


# ============================================================
# 3) CLASS-BY-CLASS HEATMAPS
# ============================================================

print("Generating class-by-class heatmaps...")

for cls in CLASSES:

    mask = [i for i, term in enumerate(auto_terms) if node_classes.get(term, "OTHER") == cls]
    if len(mask) == 0:
        continue

    sub_matrix = sim_matrix[:, mask]
    sub_auto_terms = [auto_terms[i] for i in mask]

    plt.figure(figsize=(20, 14))
    sns.heatmap(
        sub_matrix,
        cmap="magma",
        xticklabels=[short(x) for x in sub_auto_terms],
        yticklabels=[short(x) for x in ref_terms],
        vmin=0, vmax=1,
        cbar_kws={"label": "Cosine Similarity"},
    )

    plt.xticks(rotation=60, fontsize=9)
    plt.yticks(rotation=0, fontsize=11)
    plt.title(f"CLASS HEATMAP — {cls} ({len(sub_auto_terms)} nodes)", fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUT}/3_class_heatmap_{cls}.png", dpi=350)
    plt.close()

print(" Class heatmaps saved")


# ============================================================
# 4) HISTOGRAM OF SEMANTIC SIMILARITIES
# ============================================================

plt.figure(figsize=(12, 7))
plt.hist(sim_scores, bins=25, color="#8E44AD", alpha=0.85)
plt.title("Distribution of Auto KG Similarity to Reference", fontsize=17)
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/4_similarity_histogram.png", dpi=350)
plt.close()
print(" 4_similarity_histogram.png saved")


# ============================================================
# 5) SCATTER Similarity vs Degree
# ============================================================

plt.figure(figsize=(12, 8))
plt.scatter(sim_scores, degrees, s=70, alpha=0.7, color="#E67E22", edgecolors="black")
plt.title("Node Importance — Similarity vs Structural Degree", fontsize=18)
plt.xlabel("Cosine Similarity to Reference")
plt.ylabel("Degree (Connectivity Importance)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/5_similarity_vs_degree.png", dpi=350)
plt.close()
print(" 5_similarity_vs_degree.png saved")


# ============================================================
# 6) REPORT (.txt)
# ============================================================

with open(f"{OUT}/evaluation_report.txt", "w") as f:

    f.write("=== KNOWLEDGE GRAPH QUALITY REPORT ===\n\n")

    f.write(f"Exact Coverage: {coverage_exact*100:.2f}%\n")
    f.write(f"Semantic Coverage: {coverage_sem*100:.2f}%\n")
    f.write(f"Hallucination count: {hall_count}\n")
    f.write(f"Redundancy (merge candidates): {redundancy_count}\n")
    f.write(f"Weak nodes: {weak_nodes_count}\n")
    f.write(f"Semantic Cohesion (mean): {cohesion_mean:.3f}\n")
    f.write(f"Semantic Cohesion (max): {cohesion_max:.3f}\n")

    f.write("\n--- Top 20 Highest-Similarity Nodes ---\n")
    top_idx = np.argsort(sim_scores)[-20:]
    for idx in reversed(top_idx):
        f.write(f"{auto_terms[idx]} — sim={sim_scores[idx]:.3f}, degree={degrees[idx]}\n")

    f.write("\n--- Bottom 20 Lowest-Similarity Nodes (Potential noise) ---\n")
    bot_idx = np.argsort(sim_scores)[:20]
    for idx in bot_idx:
        f.write(f"{auto_terms[idx]} — sim={sim_scores[idx]:.3f}, degree={degrees[idx]}\n")

print(" evaluation_report.txt saved")

print("\n ALL EVALUATION VISUALS & REPORT GENERATED SUCCESSFULLY.")
