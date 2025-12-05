import networkx as nx
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import os

# ==========================================================
# CONFIG
# ==========================================================
KG_SMALL = "output/small_corpus/final_graph_clean.gexf"
KG_FULL  = "output/full_corpus/final_graph_clean.gexf"

TRIPLETS_SMALL = "output/small_corpus/knowledge_graph_triplets.json"
TRIPLETS_FULL  = "output/full_corpus/knowledge_graph_triplets.json"

REF_KG   = "reference/reference_kg.json"

OUT_DIR = "comparison_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N = 5  # Top-N new nodes to explain


# ==========================================================
# LOAD GRAPHS
# ==========================================================
print("Loading graphs...")
G_small = nx.read_gexf(KG_SMALL)
G_full  = nx.read_gexf(KG_FULL)

nodes_small = set(G_small.nodes())
nodes_full  = set(G_full.nodes())

shared = nodes_small.intersection(nodes_full)
only_small = nodes_small - nodes_full
only_full  = nodes_full - nodes_small


# ==========================================================
# LOAD TRIPLETS (to recover node → source articles)
# ==========================================================
def load_sources(triplet_file):
    with open(triplet_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for t in data:
        h = t["head"].lower()
        o = t["tail"].lower()
        src = t["provenance"]["source_doc"]

        mapping.setdefault(h, set()).add(src)
        mapping.setdefault(o, set()).add(src)

    return {k: sorted(list(v)) for k, v in mapping.items()}


sources_small = load_sources(TRIPLETS_SMALL)
sources_full  = load_sources(TRIPLETS_FULL)


# ==========================================================
# LOAD REFERENCE TERMS
# ==========================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(REF_KG, "r", encoding="utf-8") as f:
    ref = json.load(f)

ref_terms = [t.lower() for t in ref.get("canonical_terms", [])]
ref_emb = model.encode(ref_terms, convert_to_tensor=True)


# ==========================================================
# BUILD DIFF MATRIX
# ==========================================================
all_nodes = sorted(list(nodes_small.union(nodes_full)))

diff_matrix = pd.DataFrame(index=all_nodes, columns=[
    "in_small", "in_full",
    "degree_small", "degree_full",
    "sources_small", "sources_full",
    "explanation"
])

deg_small = dict(G_small.degree())
deg_full  = dict(G_full.degree())

for node in all_nodes:
    node_lower = node.lower()

    diff_matrix.loc[node, "in_small"] = int(node in nodes_small)
    diff_matrix.loc[node, "in_full"]  = int(node in nodes_full)
    diff_matrix.loc[node, "degree_small"] = deg_small.get(node, 0)
    diff_matrix.loc[node, "degree_full"]  = deg_full.get(node, 0)

    diff_matrix.loc[node, "sources_small"] = ", ".join(sources_small.get(node_lower, []))
    diff_matrix.loc[node, "sources_full"]  = ", ".join(sources_full.get(node_lower, []))

    # Generate explanation
    emb = model.encode(node_lower, convert_to_tensor=True)
    sims = util.cos_sim(emb, ref_emb)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims.max())
    closest_term = ref_terms[best_idx]

    if best_sim < 0.30:
        diff_matrix.loc[node, "explanation"] = "NEW concept — not in ontology"
    elif best_sim < 0.60:
        diff_matrix.loc[node, "explanation"] = f"Partial match to '{closest_term}'"
    else:
        diff_matrix.loc[node, "explanation"] = f"Strong match to '{closest_term}'"

diff_matrix["degree_change"] = diff_matrix["degree_full"] - diff_matrix["degree_small"]

# EXPORT UPDATED CSV
out_csv = f"{OUT_DIR}/diff_matrix_with_sources.csv"
diff_matrix.to_csv(out_csv)

print(f"\nSaved enriched diff matrix → {out_csv}")
print("\n>>> DONE.")
