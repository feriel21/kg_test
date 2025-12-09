import networkx as nx
import os
import csv
import json

# ============================
# LOAD CONFIG
# ============================
from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

INPUT_GEXF = paths["graph_clean"]
OUTPUT_GEXF = paths["graph_knowledge"]
OUTPUT_FACTS = paths["facts_csv"]

REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]

# ============================
# LOAD EXPERT ONTOLOGY (OPTIONNEL)
# ============================
REF_MAP = {}
if os.path.exists(REFERENCE_KG_PATH):
    with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
        ref = json.load(f)
    REF_MAP = {k.lower(): v for k, v in ref.get("ontology", {}).items()}
    print(f"[STEP4] Loaded expert ontology ({len(REF_MAP)} terms).")
else:
    print("[STEP4] WARN: No reference ontology found, lexical only.")

# ============================
# PAULINE'S 3 MAIN CATEGORIES
# ============================
TARGET_CLASSES = {
    "MTD_descriptors",
    "Mass_movement_properties",
    "Environmental_controls"
}

# Lexical keywords approximating Pauline's tables
ONTOLOGY_KEYWORDS = {
    "MTD_descriptors": [
        "morphology", "basal surface", "upper surface", "internal facies",
        "headscarp", "headwall", "toe", "ridges", "ramp", "fault", "faults",
        "thickness", "thickening", "thinning", "geometry", "lobe", "tongue",
        "deformed facies", "chaotic facies", "transparent facies",
        "groove", "scour", "scar", "scarp"
    ],
    "Mass_movement_properties": [
        "trigger phase", "transport phase", "post-deposition phase",
        "runout", "run-out", "velocity", "flow velocity", "shear",
        "overpressure", "pore pressure", "failure", "instability",
        "slump", "slide", "debris flow", "mass movement", "mass transport",
        "shear zone", "basal shear", "detachment", "failure surface"
    ],
    "Environmental_controls": [
        "sea level", "relative sea level", "tectonics", "subsidence",
        "uplift", "compression", "extension", "basin geometry",
        "slope gradient", "topography", "sedimentation rate",
        "climate", "glacial", "evaporite", "salt tectonics"
    ]
}

COLORS = {
    "MTD_descriptors":  {"r": 52,  "g": 152, "b": 219},  # blue
    "Mass_movement_properties": {"r": 231, "g": 76,  "b": 60},   # red
    "Environmental_controls":   {"r": 46,  "g": 204, "b": 113},  # green
    "UNCLASSIFIED":             {"r": 149, "g": 165, "b": 166},  # grey
}

# ============================
# CATEGORY ASSIGNMENT (NOUVEAU)
# ============================
def assign_category(node_label: str, data: dict) -> str:
    """
    NEW STRATEGY:
    1) If already has a valid category from a previous step → keep it.
    2) Try exact match with expert KG ontology (REF_MAP).
    3) Try lexical partial match with Pauline keyword lists.
    4) Else → UNCLASSIFIED.
    """

    # 0 — preserve existing valid category if present
    existing = data.get("category")
    if isinstance(existing, str) and existing in TARGET_CLASSES:
        return existing

    text = (node_label or "").lower().strip()

    # 1 — Expert KG exact mapping
    if text in REF_MAP and REF_MAP[text] in TARGET_CLASSES:
        return REF_MAP[text]

    # 2 — Lexical heuristics (contains keyword)
    for cat, kw_list in ONTOLOGY_KEYWORDS.items():
        for kw in kw_list:
            if kw in text:
                return cat

    # 3 — Default
    return "UNCLASSIFIED"


# ============================
# RELATION ENRICHMENT RULES
# ============================
def enrich_relation(cat_u, cat_v):
    """
    Simple semantic layer aligned with Pauline logic:
    Env → Mass → Descriptors
    """
    if cat_u == "Environmental_controls" and cat_v == "Mass_movement_properties":
        return "INFLUENCES"

    if cat_u == "Mass_movement_properties" and cat_v == "MTD_descriptors":
        return "FORMS"

    if cat_u == "Environmental_controls" and cat_v == "MTD_descriptors":
        return "SHAPES"

    if cat_u == "Mass_movement_properties" and cat_v == "Environmental_controls":
        return "RESPONDS_TO"

    return "RELATED_TO"


# ============================
# MAIN SEMANTIC ENRICHMENT
# ============================
def semantic_enrich(G: nx.DiGraph):

    print("[STEP4] Assigning Pauline categories (lexical + expert)...")

    for node, data in G.nodes(data=True):

        cat = assign_category(node, data)

        data["category"] = cat
        data["class"] = cat
        data["viz"] = {"color": COLORS.get(cat, COLORS["UNCLASSIFIED"])}

    # Sanity check
    cats = {G.nodes[n]["category"] for n in G.nodes()}
    print(f"[STEP4] Category distribution: {cats}")

    print("[STEP4] Refining relations...")
    for u, v, edata in G.edges(data=True):
        cu = G.nodes[u]["category"]
        cv = G.nodes[v]["category"]
        edata["label"] = enrich_relation(cu, cv)

    return G


# ============================
# EXPORT FACTS
# ============================
def export_facts(G):
    print(f"[STEP4] Exporting facts → {OUTPUT_FACTS}")
    with open(OUTPUT_FACTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Subject", "Class_S", "Relation", "Object", "Class_O"])
        for u, v, edata in G.edges(data=True):
            w.writerow([
                u,
                G.nodes[u].get("category", "UNCLASSIFIED"),
                edata.get("label", "RELATED_TO"),
                v,
                G.nodes[v].get("category", "UNCLASSIFIED"),
            ])


# ============================
# MAIN
# ============================
def main():
    print(f"[STEP4] Loading graph: {INPUT_GEXF}")
    G = nx.read_gexf(INPUT_GEXF)

    G = semantic_enrich(G)

    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"[STEP4] Saved enriched graph → {OUTPUT_GEXF}")

    export_facts(G)
    print("[STEP4] DONE.")


if __name__ == "__main__":
    main()
