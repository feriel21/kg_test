#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 6 — Knowledge Graph Evaluation (Pauline Ontology Version)

This evaluates:
    - Exact & semantic coverage of expert terms
    - Hallucinations (irrelevant concepts)
    - Redundancy (duplicate concepts)
    - Suspicious category-to-category relations
    - Graph cohesion
    - MTD geological signature score
"""

import json
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer, util

from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

AUTO_KG = paths["graph_knowledge"]
REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]
OUTPUT = paths["evaluation_json"]

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading graphs…")
G = nx.read_gexf(AUTO_KG)

# ============================================================
# LOAD REFERENCE CANONICAL TERMS (PAULINE)
# ============================================================
with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
    ref_data = json.load(f)

REF_TERMS = [t.lower() for t in ref_data.get("canonical_terms", [])]

print(f"> Auto KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"> Reference canonical terms: {len(REF_TERMS)}")

# ============================================================
# LOAD SBERT
# ============================================================
model = SentenceTransformer(MODEL_NAME)

auto_nodes = list(G.nodes())
auto_nodes_lower = [n.lower() for n in auto_nodes]

print("Encoding auto KG nodes...")
auto_emb = model.encode(auto_nodes_lower, convert_to_tensor=True)

print("Encoding reference terms...")
ref_emb = model.encode(REF_TERMS, convert_to_tensor=True)

# Similarities
sim_ref_to_auto = util.cos_sim(ref_emb, auto_emb)
sim_auto_to_ref = util.cos_sim(auto_emb, ref_emb)

# ============================================================
# 1 — Coverage (Exact & Semantic)
# ============================================================
exact_found = sum(1 for t in REF_TERMS if t in auto_nodes_lower)
exact_cov = exact_found / len(REF_TERMS)

best_sim_ref = sim_ref_to_auto.max(dim=1).values
semantic_hits = (best_sim_ref > 0.70).sum().item()
semantic_cov = semantic_hits / len(REF_TERMS)

missing_terms = [
    (REF_TERMS[i], float(best_sim_ref[i]))
    for i in range(len(REF_TERMS))
    if best_sim_ref[i] < 0.70
]

# ============================================================
# 2 — Hallucinations (Nodes far from reference domain)
# ============================================================
max_sim_auto = sim_auto_to_ref.max(dim=1).values

hallucinations = [
    (auto_nodes[i], float(max_sim_auto[i]))
    for i in range(len(auto_nodes))
    if max_sim_auto[i] < 0.40
]

# ============================================================
# 3 — Redundancy (Nodes almost duplicates)
# ============================================================
sim_auto_auto = util.cos_sim(auto_emb, auto_emb)
merge_pairs = []

for i in range(len(auto_nodes)):
    for j in range(i+1, len(auto_nodes)):
        score = float(sim_auto_auto[i][j])
        if score > 0.85:
            merge_pairs.append([auto_nodes[i], auto_nodes[j], score])

# ============================================================
# 4 — Suspicious relations (Pauline category logic)
# ============================================================

# Valid category mapping
PAULINE_CATS = {
    "MTD_descriptors",
    "Mass_movement_properties",
    "Environmental_controls",
    "UNCLASSIFIED"
}

def get_cat(n):
    c = G.nodes[n].get("category", "UNCLASSIFIED")
    return c if c in PAULINE_CATS else "UNCLASSIFIED"

# Valid inter-category edges (Pauline logic)
VALID_RELATIONS = {
    ("Environmental_controls", "Mass_movement_properties"): "INFLUENCES",
    ("Mass_movement_properties", "MTD_descriptors"): "FORMS",
    ("Environmental_controls", "MTD_descriptors"): "SHAPES",
    ("Mass_movement_properties", "Environmental_controls"): "RESPONDS_TO",
}

suspicious_edges = []

for u, v, d in G.edges(data=True):
    cu = get_cat(u)
    cv = get_cat(v)
    actual = d.get("label", "RELATED_TO")
    expected = VALID_RELATIONS.get((cu, cv), None)

    # If a specific rule exists but graph contradicts → suspicious
    if expected and actual != expected:
        suspicious_edges.append([u, v, actual, cu, cv, expected])

# ============================================================
# 5 — Weak nodes (degree <= 1)
# ============================================================
weak_nodes = [
    (n, G.degree(n))
    for n in G.nodes()
    if G.degree(n) <= 1
]

# ============================================================
# 6 — Cohesion metrics
# ============================================================
upper = []
for i in range(len(auto_nodes)):
    for j in range(i+1, len(auto_nodes)):
        upper.append(float(sim_auto_auto[i][j]))

cohesion_mean = float(np.mean(upper))
cohesion_max = float(np.max(upper))

# ============================================================
# 7 — MTD Signature Score (improved)
# ============================================================
MTD_SIGNATURE_TERMS = [
    "slump", "slide", "block", "raft", "chaotic facies",
    "transparent facies", "shear surface", "headwall",
    "toe", "runout", "mass transport", "failure surface"
]

signature_found = sum(
    1 for t in MTD_SIGNATURE_TERMS
    if any(t in n.lower() for n in auto_nodes)
)

signature_score = signature_found / len(MTD_SIGNATURE_TERMS)

# ============================================================
# BUILD JSON OUTPUT
# ============================================================

output = {
    "coverage": {
        "exact_coverage": exact_cov,
        "semantic_coverage": semantic_cov,
        "missing_terms": missing_terms,
    },
    "hallucinations": {
        "count": len(hallucinations),
        "nodes": hallucinations
    },
    "redundancy": {
        "count": len(merge_pairs),
        "pairs": merge_pairs
    },
    "suspicious_edges": suspicious_edges,
    "weak_nodes": {
        "count": len(weak_nodes),
        "nodes": weak_nodes
    },
    "cohesion": {
        "mean_similarity": cohesion_mean,
        "max_similarity": cohesion_max
    },
    "signature_test": {
        "signature_score": signature_score,
        "found": signature_found,
        "total": len(MTD_SIGNATURE_TERMS)
    },
    "metadata": {
        "ref_terms": REF_TERMS,
        "auto_terms": auto_nodes,
        "degrees": [int(G.degree(n)) for n in auto_nodes],
    }
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"\n✔ Evaluation saved → {OUTPUT}")
