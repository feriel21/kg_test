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

INPUT_GEXF = paths["graph_modular"]
OUTPUT_GEXF = paths["graph_clean"]

# ======================================================
# LOAD EXPERT TERMS
# ======================================================
REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]

if os.path.exists(REFERENCE_KG_PATH):
    with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    REF_TERMS = set(ref_data.get("canonical_terms", []))
else:
    print("⚠️ WARNING: reference_kg.json not found. Geological filtering weakened.")
    REF_TERMS = set()

# ======================================================
# SBERT MODEL
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

if len(REF_TERMS) > 0:
    REF_EMB = model.encode(list(REF_TERMS), convert_to_tensor=True)
else:
    REF_EMB = None

# ======================================================
# GEOLOGICAL VOCABULARY
# ======================================================
GEO_KEYWORDS = {
    "mtd", "slide", "slump", "debris", "flow", "mass transport",
    "headwall", "scarp", "toe", "failure", "shear",
    "basin", "slope", "margin", "fan", "channel", "lobe",
    "facies", "chaotic", "transparent", "deformation"
}

PRONOUNS = {"we", "they", "them", "their", "its", "it", "he", "she"}
STOPWORDS = {"this", "that", "these", "those"}
GENERIC = {"area", "case", "values", "type", "something"}

FORBIDDEN = PRONOUNS | STOPWORDS | GENERIC

# ======================================================
# FUNCTIONS
# ======================================================
def is_noise(n):
    n2 = n.lower().strip()
    if n2 in FORBIDDEN:
        return True
    if len(n2) < 3:
        return True
    if not any(c.isalpha() for c in n2):
        return True
    return False


def is_geological(n):
    txt = n.lower()

    if txt in REF_TERMS:
        return True

    if any(k in txt for k in GEO_KEYWORDS):
        return True

    if REF_EMB is not None:
        emb = model.encode(txt, convert_to_tensor=True)
        if util.cos_sim(emb, REF_EMB).max().item() > 0.55:
            return True

    return False


def is_geological_by_neighbors(G, n):
    count = sum(1 for neigh in G.neighbors(n) if is_geological(neigh))
    return count >= 2


# ======================================================
# MAIN
# ======================================================
def main():
    print(f"Loading graph: {INPUT_GEXF}")
    G = nx.read_gexf(INPUT_GEXF)
    print(f"Original: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    # 🔥 SAVE ORIGINAL METADATA
    node_meta = {n: dict(G.nodes[n]) for n in G.nodes()}

    # 1) Remove isolated nodes
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])

    # 2) Largest connected component
    if G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    # 3) Semantic noise cleanup
    noise = [n for n in G.nodes() if is_noise(n)]
    G.remove_nodes_from(noise)

    # 4) Geological filtering
    to_remove = []
    for n in G.nodes():
        if not is_geological(n) and not is_geological_by_neighbors(G, n):
            to_remove.append(n)

    G.remove_nodes_from(to_remove)

    # 5) Final LCC
    if G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    # 🔥 RESTORE METADATA (category, auto_category, source info…)
    for n in G.nodes():
        if n in node_meta:
            for key, val in node_meta[n].items():
                G.nodes[n][key] = val

    print(f"Cleaned graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"Saved → {OUTPUT_GEXF}")


if __name__ == "__main__":
    main()
