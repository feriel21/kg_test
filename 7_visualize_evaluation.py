import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

EVAL_FILE = "output_graph/evaluation_results.json"
KG_FILE = "output_graph/final_graph_knowledge_layer.gexf"
OUT_DIR = "output_graph/eval_plots_final"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    E = json.load(f)

G = nx.read_gexf(KG_FILE)

degrees = np.array(E["degrees"])
similarities = np.array(E["similarity_scores"])
sim_matrix = np.array(E["similarity_matrix"])
auto_terms = E["auto_terms"]
ref_terms = E["ref_terms"]

# =====================================================
# 1) COVERAGE (exact + semantic)
# =====================================================
def plot_coverage():
    exact = E["coverage"]["exact"]
    sem = E["coverage"]["semantic"]
    missing = len(E["coverage"]["missing_terms"])

    plt.figure(figsize=(7,4))
    sns.barplot(x=["Exact","Semantic"], y=[exact, sem], palette="Blues_d")
    plt.title("KG Coverage (Exact vs Semantic)")
    plt.ylabel("Coverage Score")
    plt.ylim(0,1)

    # annotation
    plt.text(0, exact+0.02, f"{exact:.2f}", ha="center")
    plt.text(1, sem+0.02, f"{sem:.2f}", ha="center")

    plt.savefig(f"{OUT_DIR}/coverage.png", dpi=200)
    plt.close()

# =====================================================
# 2) Similarity Histogram
# =====================================================
def plot_similarity_histogram():
    plt.figure(figsize=(8,4))
    sns.histplot(similarities, bins=40, kde=True, color="purple")
    plt.title("Similarity Distribution (Auto → Reference)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUT_DIR}/similarity_histogram.png", dpi=200)
    plt.close()

# =====================================================
# 3) Improved Heatmap (annotated, more meaning)
# =====================================================
def plot_similarity_heatmap_explicit():
    # keep 30×30 most frequent nodes
    top_idx = np.argsort(degrees)[-30:]
    sub_matrix = sim_matrix[:, top_idx][:30, :]

    plt.figure(figsize=(12,10))
    ax = sns.heatmap(
        sub_matrix,
        cmap="viridis",
        cbar=True,
        xticklabels=[auto_terms[i] for i in top_idx],
        yticklabels=ref_terms[:30],
        annot=False
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.title("Semantic Similarity Heatmap (Top Concepts)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/similarity_heatmap_explicit.png", dpi=300)
    plt.close()

# =====================================================
# 4) Hallucinations
# =====================================================
def plot_hallucinations():
    count = E["hallucinations"]["count"]

    plt.figure(figsize=(5,4))
    sns.barplot(x=["Hallucinations"], y=[count], color="red")
    plt.title("Non-Geological Concepts Detected")
    plt.ylim(0, max(2,count+1))
    plt.text(0, count+0.1, str(count), ha="center")
    plt.savefig(f"{OUT_DIR}/hallucinations.png", dpi=200)
    plt.close()

# =====================================================
# 5) Suspicious Edges Graph
# =====================================================
def plot_suspicious_edges():
    susp = E["suspicious_edges"]

    if len(susp)==0:
        plt.figure(figsize=(6,4))
        plt.text(0.3,0.5,"No suspicious relations found", fontsize=14)
        plt.axis("off")
        plt.savefig(f"{OUT_DIR}/suspicious_edges.png", dpi=200)
        plt.close()
        return

    H = nx.DiGraph()
    for u,v,rel,cu,cv,exp in susp:
        H.add_edge(u, v, label=rel)

    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(H, seed=42)
    nx.draw(H,pos,node_color="lightcoral",edge_color="black",with_labels=True,font_size=8)
    labels = nx.get_edge_attributes(H,'label')
    nx.draw_networkx_edge_labels(H,pos,edge_labels=labels,font_size=6)
    plt.title("Suspicious Relations in the KG")
    plt.savefig(f"{OUT_DIR}/suspicious_edges.png",dpi=200)
    plt.close()

# =====================================================
# 6) Redundancy (clusters to merge)
# =====================================================
def plot_redundancy():
    count = E["redundancy"]["count"]
    plt.figure(figsize=(5,4))
    sns.barplot(x=["Redundant Pairs"],y=[count],color="green")
    plt.title("Near-Duplicate Concepts (>0.85 similarity)")
    plt.ylim(0,max(2,count+1))
    plt.text(0,count+0.1,str(count),ha="center")
    plt.savefig(f"{OUT_DIR}/redundancy.png",dpi=200)
    plt.close()

# =====================================================
# 7) Cohesion
# =====================================================
def plot_cohesion():
    mean = E["cohesion"]["mean"]
    maxv = E["cohesion"]["max"]

    plt.figure(figsize=(6,4))
    sns.barplot(x=["Mean","Max"],y=[mean,maxv],palette="coolwarm")
    plt.ylim(0,1)
    plt.title("Global Semantic Cohesion")
    plt.savefig(f"{OUT_DIR}/cohesion.png",dpi=200)
    plt.close()

# =====================================================
# 8) Top-20 Hubs (important geological concepts)
# =====================================================
def plot_top_20_nodes():
    deg_dict = dict(G.degree())
    top = sorted(deg_dict.items(), key=lambda x: x[1], reverse=True)[:20]

    labels = [n for n,_ in top]
    values = [d for _,d in top]

    plt.figure(figsize=(10,8))
    sns.barplot(y=labels, x=values, palette="mako")
    plt.title("Top 20 Most Important Nodes (Hubs)")
    plt.xlabel("Degree")
    plt.ylabel("Node")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top20_nodes.png", dpi=300)
    plt.close()

# =====================================================
# 9) Top-20 Hallucinations
# =====================================================
def plot_top20_hallucinations():
    hall = E["hallucinations"]["nodes"]
    if len(hall)==0:
        return

    hall_sorted = sorted(hall, key=lambda x: x[1])[:20]

    labels = [n for n,_ in hall_sorted]
    values = [s for _,s in hall_sorted]

    plt.figure(figsize=(10,8))
    sns.barplot(y=labels, x=values, color="red")
    plt.title("Top 20 Hallucinations (Lowest Similarity)")
    plt.xlabel("Similarity")
    plt.ylabel("Node")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top20_hallucinations.png", dpi=300)
    plt.close()

# =====================================================
# RUN ALL
# =====================================================
plot_coverage()
plot_similarity_histogram()
plot_similarity_heatmap_explicit()
plot_hallucinations()
plot_suspicious_edges()
plot_redundancy()
plot_cohesion()
plot_top_20_nodes()
plot_top20_hallucinations()

print(f"\n>> All final visualizations saved in:\n   {OUT_DIR}\n")
