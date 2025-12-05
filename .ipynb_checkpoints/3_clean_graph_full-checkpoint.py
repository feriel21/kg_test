import networkx as nx
import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
# ============================
# LOAD CONFIG
# ============================
from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

# Input graph (from step 2)
INPUT_GEXF = paths["graph_modular"]

# Output graph (cleaned)
OUTPUT_GEXF = paths["graph_clean"]
# ======================================================
# 1 — LOAD GEOLOGICAL REFERENCE TERMS
# ======================================================
REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]

if os.path.exists(REFERENCE_KG_PATH):
    with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    REF_TERMS = set(ref_data.get("canonical_terms", []))
else:
    print("⚠️ WARNING: reference_kg.json not found. Geological filter weakened.")
    REF_TERMS = set()

# ======================================================
# 2 — LOAD SBERT FOR FUZZY GEOLOGICAL SIMILARITY
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Pre-encode expert terms for faster similarity
REF_EMB = model.encode(list(REF_TERMS), convert_to_tensor=True) if len(REF_TERMS) > 0 else None


# ======================================================
# 3 — GEOLOGICAL VOCABULARY (fallback)
# ======================================================
GEO_KEYWORDS = {
    "mtd", "slide", "slump", "debris", "flow", "mass", "transport",
    "scar", "scarp", "failure", "avalanche", "collapse",
    "sediment", "deposit", "unit", "layer", "lodgment",
    "basin", "slope", "margin", "fan", "channel", "levee", "delta",
    "lobe", "toe", "headwall", "block", "raft",
    "surface", "boundary", "interface", "plane",
    "facies", "chaotic", "transparent", "amplitude",
    "reflection", "stratified", "structure", "geometry",
    "shear", "compression", "extension", "deformation",
    "ramp", "scour", "groove", "striations" , "mass transport deposit"
}





# Semantic noise lists
PRONOUNS = {
    "we", "they", "them", "their", "its", "it",
    "he", "she", "his", "her", "you", "your"
}

STOPWORDS = {
    "this", "that", "these", "those", "which", "what", "when",
    "something", "anything", "everything", "other",
    "one", "two", "three", "four", "five",
    "such", "some", "more", "most", "part", "parts"
}

GENERIC = {
    "area", "areas", "series", "case", "example",
    "value", "type", "group", "data", "set"
}

FORBIDDEN = PRONOUNS | STOPWORDS | GENERIC
MIN_LEN = 4


# ======================================================
# 5 — HELPER FUNCTIONS
# ======================================================
def is_semantic_noise(node: str):
    txt = node.lower().strip()
    if txt in FORBIDDEN:
        return True
    if len(txt) < MIN_LEN:
        return True
    clean = txt.replace(" ", "").replace("-", "")
    if not clean.isalpha():
        return True
    return False


def is_geological(node: str):
    """
    Flexible geological filter. A node is geological if:
    1) exact match with REF_TERMS
    2) fuzzy SBERT similarity match with expert terms
    3) contains a geological keyword
    4) connected to many geological nodes (structural fallback)
    """

    txt = node.lower()

    # --- 1) EXACT MATCH ---
    if txt in REF_TERMS:
        return True

    # --- 2) KEYWORD MATCH ---
    if any(k in txt for k in GEO_KEYWORDS):
        return True

    # --- 3) FUZZY SEMANTIC SBERT MATCH ---
    if REF_EMB is not None:
        emb = model.encode(txt, convert_to_tensor=True)
        sim = util.cos_sim(emb, REF_EMB).max().item()
        if sim > 0.55:  # soft threshold
            return True

    # otherwise → not geological
    return False

def is_geological_by_neighbors(G, node):
    count_geo_neighbors = 0
    for neigh in G.neighbors(node):
        if is_geological(neigh):
            count_geo_neighbors += 1

    return count_geo_neighbors >= 2  # keep if 2 geological neighbors

# ======================================================
# 6 — MAIN
# ======================================================
def main():
    print(f"Loading graph : {INPUT_GEXF}")
    G = nx.read_gexf(INPUT_GEXF)
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # -------------------------
    # 1 — STRUCTURAL CLEANING
    # -------------------------
    deg_rm = [n for n, d in G.degree() if d < 1]  # keep nodes with degree >= 1

    G.remove_nodes_from(deg_rm)
    print(f"After degree filter: {G.number_of_nodes()} nodes")

    if G.number_of_nodes() > 0:
        lcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
    print(f"After LCC: {G.number_of_nodes()} nodes")

    # -------------------------
    # 2 — SEMANTIC CLEANING
    # -------------------------
    noise = [n for n in G.nodes() if is_semantic_noise(n)]
    print(f"Semantic noise removed: {len(noise)}")
    G.remove_nodes_from(noise)

    if G.number_of_nodes() > 0:
        lcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    # -------------------------
    # 3 — GEOLOGICAL FILTER
    # -------------------------
    geo_remove = []
    for n in G.nodes():
        if not is_geological(n):
            if not is_geological_by_neighbors(G, n):
                geo_remove.append(n)
               
    
        
           


    print(f"Geological filter removed: {len(geo_remove)} nodes")
    G.remove_nodes_from(geo_remove)

    # final LCC
    if G.number_of_nodes() > 0:
        lcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    print(f"Final cleaned graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"\nSaved cleaned graph → {OUTPUT_GEXF}")


if __name__ == "__main__":
    main()
