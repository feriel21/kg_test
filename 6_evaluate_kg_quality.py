import json
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

# 1) Final knowledge graph to evaluate
AUTO_KG = paths["graph_knowledge"]

# 2) Expert reference KG
REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]


# 3) Output JSON for evaluation
OUTPUT = paths["evaluation_json"]

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading graphs...")


G = nx.read_gexf(AUTO_KG)

with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:

    ref = json.load(f)

# -------------------------------------------------
# Extract reference terms
# -------------------------------------------------
REF_TERMS = [t.lower() for t in ref.get("canonical_terms", [])]

print(f"- Auto KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"- Reference terms: {len(REF_TERMS)}")

# -------------------------------------------------
# Encode terms using SBERT
# -------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

auto_nodes = list(G.nodes())
auto_nodes_lower = [n.lower() for n in auto_nodes]

print("Encoding auto KG nodes...")
auto_emb = model.encode(auto_nodes_lower, convert_to_tensor=True)

print("Encoding reference terms...")
ref_emb = model.encode(REF_TERMS, convert_to_tensor=True)

# -------------------------------------------------
# Compute cosine similarities
# -------------------------------------------------
sim_ref_to_auto = util.cos_sim(ref_emb, auto_emb)  # [n_ref, n_auto]
sim_auto_to_ref = util.cos_sim(auto_emb, ref_emb)  # [n_auto, n_ref]

# -------------------------------------------------
# 1) Coverage test (exact + semantic)
# -------------------------------------------------
exact_found = sum(1 for t in REF_TERMS if t in auto_nodes_lower)
exact_cov = exact_found / len(REF_TERMS)

best_sim_ref = sim_ref_to_auto.max(dim=1).values
semantic_hits = (best_sim_ref > 0.70).sum().item()
semantic_cov = semantic_hits / len(REF_TERMS)

# -------------------------------------------------
# 2) Hallucinations (low similarity to reference)
# -------------------------------------------------
max_sim_auto = sim_auto_to_ref.max(dim=1).values
hallucinations = [
    (auto_nodes[i], float(max_sim_auto[i]))
    for i in range(len(auto_nodes))
    if max_sim_auto[i] < 0.40
]

# -------------------------------------------------
# 3) Redundancy (near-duplicate concepts)
# -------------------------------------------------
sim_auto_auto = util.cos_sim(auto_emb, auto_emb)
merge_pairs = []
for i in range(len(auto_nodes)):
    for j in range(i+1, len(auto_nodes)):
        score = float(sim_auto_auto[i][j])
        if score > 0.85:
            merge_pairs.append([auto_nodes[i], auto_nodes[j], score])

# -------------------------------------------------
# 4) Suspicious relations (class mismatch)
# -------------------------------------------------
def expected_relation(u_cls, v_cls):
    rules = {
        ("TRIGGER", "PROCESS"): "CAUSES",
        ("PROCESS", "LOCATION"): "LOCATED_IN",
        ("PROCESS", "MATERIAL"): "TRANSPORTS",
        ("PROCESS", "FEATURE"): "FORMS",
        ("FEATURE", "FACIES"): "EXHIBITS",
    }
    return rules.get((u_cls, v_cls), None)

suspicious_edges = []
for u, v, d in G.edges(data=True):
    rel = d.get("label", "RELATED_TO")
    cu = G.nodes[u].get("class", "OTHER")
    cv = G.nodes[v].get("class", "OTHER")
    exp = expected_relation(cu, cv)
    if exp and rel != exp:
        suspicious_edges.append([u, v, rel, cu, cv, exp])

# -------------------------------------------------
# 5) Weak nodes (degree <=1)
# -------------------------------------------------
weak_nodes = [(n, d) for n, d in G.degree() if d <= 1]

# -------------------------------------------------
# 6) Global cohesion
# -------------------------------------------------
upper_tri = []
for i in range(len(auto_nodes)):
    for j in range(i+1, len(auto_nodes)):
        upper_tri.append(float(sim_auto_auto[i][j]))

cohesion_mean = float(np.mean(upper_tri))
cohesion_max = float(np.max(upper_tri))

# -------------------------------------------------
# Save evaluation results
# -------------------------------------------------
output = {
    "coverage": {
        "exact": exact_cov,
        "semantic": semantic_cov,
        "missing_terms": [
            (REF_TERMS[i], float(best_sim_ref[i]))
            for i in range(len(REF_TERMS))
            if best_sim_ref[i] < 0.70
        ]
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
        "mean": cohesion_mean,
        "max": cohesion_max
    },
    "similarity_matrix": sim_ref_to_auto.tolist(),
    "similarity_scores": max_sim_auto.tolist(),
    "ref_terms": REF_TERMS,
    "auto_terms": auto_nodes,
    "degrees": [int(G.degree(n)) for n in auto_nodes]
}

# ============================================================
# EXTENDED GEOLOGICAL TESTS (5 ADVANCED METRICS)
# ============================================================

print("\nRunning extended geological evaluation tests...")

# ------------------------------------------------------------
# TEST 1 — Coverage by class
# ------------------------------------------------------------
ontology_classes = ["PROCESS", "FEATURE", "FACIES", "TRIGGER", "LOCATION", "MATERIAL"]

def get_class(n):
    return G.nodes[n].get("class", "OTHER")

coverage_by_class = {}
for cls in ontology_classes:
    ref_cls_terms = [t for t in REF_TERMS if ref.get("ontology", {}).get(t, None) == cls]
    if len(ref_cls_terms) == 0:
        coverage_by_class[cls] = 0
        continue
    found = sum(1 for t in ref_cls_terms if t in auto_nodes_lower)
    coverage_by_class[cls] = found / len(ref_cls_terms)


# ------------------------------------------------------------
# TEST 2 — Stability Test (if small corpus also exists)
# ------------------------------------------------------------
stability_score = None

try:
    # Try to load small graph automatically
    SMALL_G = nx.read_gexf(paths["graph_modular_small"])
    small_nodes = set(SMALL_G.nodes())
    full_nodes = set(G.nodes())

    shared_nodes = len(small_nodes.intersection(full_nodes))
    stability_score = shared_nodes / max(len(small_nodes), len(full_nodes))

except Exception:
    stability_score = "N/A"


# ------------------------------------------------------------
# TEST 3 — Density & Relationship Richness
# ------------------------------------------------------------
density = G.number_of_edges() / max(1, G.number_of_nodes())
relation_labels = [d.get("label", "RELATED_TO") for _,_,d in G.edges(data=True)]
relation_diversity = len(set(relation_labels))


# ------------------------------------------------------------
# TEST 4 — Provenance Test (how many articles mention each node)
# ------------------------------------------------------------
provenance_count = {}

for u, v, data in G.edges(data=True):
    sources = data.get("sources", "")
    if isinstance(sources, str):
        sources = sources.split("|")

    # count sources for both nodes
    provenance_count[u] = provenance_count.get(u, set()).union(set(sources))
    provenance_count[v] = provenance_count.get(v, set()).union(set(sources))

provenance_summary = {n: len(srcs) for n, srcs in provenance_count.items()}


# ------------------------------------------------------------
# TEST 5 — MTD Signature Score
# ------------------------------------------------------------
MTD_SIGNATURE_TERMS = [
    "slump", "slide", "debris flow", "mass transport", "headwall",
    "basal shear zone", "toe", "blocky facies", "chaotic facies",
    "transparent facies", "failure surface", "detachment surface"
]

signature_found = sum(1 for t in MTD_SIGNATURE_TERMS if t in auto_nodes_lower)
signature_score = signature_found / len(MTD_SIGNATURE_TERMS)


# ------------------------------------------------------------
# Inject extended metrics into output JSON
# ------------------------------------------------------------
output["extended_tests"] = {
    "coverage_by_class": coverage_by_class,
    "stability_score": stability_score,
    "density": density,
    "relation_diversity": relation_diversity,
    "provenance_count": provenance_summary,
    "signature_mtd_score": signature_score,
    "signature_terms_found": signature_found,
    "signature_terms_total": len(MTD_SIGNATURE_TERMS),
}

print("Extended tests added to evaluation JSON.")


with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"\n Evaluation saved → {OUTPUT}")
