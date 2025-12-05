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

# Input graph (cleaned)
INPUT_GEXF = paths["graph_clean"]

# Output graph (knowledge layer)
OUTPUT_GEXF = paths["graph_knowledge"]

# CSV export path
OUTPUT_FACTS = paths["facts_csv"]

# Reference ontology path (MANDATORY)
REFERENCE_KG_PATH = cfg["reference"]["reference_kg"]



# ================================
# LOAD REFERENCE ONTOLOGY
# ================================
if os.path.exists(REFERENCE_KG_PATH):
    with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    REF_ONTOLOGY = ref_data.get("ontology", {})
    print(f"Loaded ontology from {REFERENCE_KG_PATH} with {len(REF_ONTOLOGY)} entries.")
else:
    print("⚠️ WARNING: reference_kg.json not found. Using heuristic ontology only.")
    REF_ONTOLOGY = {}

# ================================
# HEURISTIC ONTOLOGY (FALLBACK)
# ================================
ONTOLOGY_CLASSES = {
    "PROCESS": [
        "slide", "slump", "flow", "creep", "avalanche",
        "turbidite", "failure", "movement", "instability",
        "deformation", "transport"
    ],
    "FEATURE": [
        "scarp", "scar", "headwall", "toe", "block", "lobe",
        "lens", "channel", "levee", "mound", "geometry",
        "ramp", "surface", "boundary", "plane", "interface"
    ],
    "FACIES": [
        "chaotic", "transparent", "amplitude",
        "reflection", "seismic", "stratified",
        "hummocky", "continuous"
    ],
    "LOCATION": [
        "basin", "slope", "margin", "fan", "delta",
        "canyon", "sea", "gulf", "offshore", "continental",
        "zone", "platform", "flank"
    ],
    "MATERIAL": [
        "sediment", "sand", "clay", "mud", "debris",
        "clast", "rock", "granule"
    ],
    "TRIGGER": [
        "earthquake", "tectonic", "loading", "dissociation",
        "pressure", "overpressure", "pore", "storm", "climate"
    ]
}

# ================================
# UTILITIES
# ================================
def get_node_class(label: str) -> str:
    """
    Classify a node using:
    1) Reference ontology (expert KG)
    2) Heuristic ontology by keywords (fallback)
    """
    text = label.lower().strip()

    # 1) Expert ontology from reference_kg.json
    if text in REF_ONTOLOGY:
        return REF_ONTOLOGY[text]

    # 2) Heuristic match based on domain keywords
    for cls, keywords in ONTOLOGY_CLASSES.items():
        if any(k in text for k in keywords):
            return cls

    return "OTHER"


def refine_edge_relation(cls_u, cls_v, current_label=None):
    """
    Apply semantic rules to refine relationships.
    If no rule applies, keep current_label as-is.
    """

    # Trigger → Process
    if cls_u == "TRIGGER" and cls_v == "PROCESS":
        return "CAUSES"

    # Process → Location
    if cls_u == "PROCESS" and cls_v == "LOCATION":
        return "LOCATED_IN"

    # Process → Material
    if cls_u == "PROCESS" and cls_v == "MATERIAL":
        return "TRANSPORTS"

    # Process → Feature
    if cls_u == "PROCESS" and cls_v == "FEATURE":
        return "FORMS"

    # Feature → Facies
    if cls_u == "FEATURE" and cls_v == "FACIES":
        return "EXHIBITS"

    # default: keep existing label if provided
    return current_label


# ================================
# STEP 6 — SEMANTIC ENRICHMENT
# ================================
def semantic_enrich(G: nx.DiGraph):
    print("Classifying nodes (expert ontology + heuristics)...")

    # Assign classes + colors
    for node, data in G.nodes(data=True):
        label = node
        cls = get_node_class(label)
        data["class"] = cls  # main ontology class

        # Optional: also keep 'category' for compatibility with previous steps
        if "category" not in data:
            data["category"] = cls

        # Color for Gephi (viz attribute)
        if cls == "PROCESS":
            data["viz"] = {"color": {"r": 255, "g": 0, "b": 0}}       # Red
        elif cls == "LOCATION":
            data["viz"] = {"color": {"r": 0, "g": 0, "b": 255}}       # Blue
        elif cls == "FACIES":
            data["viz"] = {"color": {"r": 255, "g": 165, "b": 0}}     # Orange
        elif cls == "FEATURE":
            data["viz"] = {"color": {"r": 0, "g": 255, "b": 255}}     # Cyan
        elif cls == "TRIGGER":
            data["viz"] = {"color": {"r": 128, "g": 0, "b": 128}}     # Purple
        elif cls == "MATERIAL":
            data["viz"] = {"color": {"r": 139, "g": 69, "b": 19}}     # Brown

    print("Refining edges semantically...")

    enriched_count = 0
    for u, v, data in G.edges(data=True):
        cls_u = G.nodes[u].get("class", "OTHER")
        cls_v = G.nodes[v].get("class", "OTHER")

        current_label = data.get("label", "RELATED_TO")
        new_rel = refine_edge_relation(cls_u, cls_v, current_label=current_label)

        if new_rel != current_label:
            data["label"] = new_rel
            enriched_count += 1
        else:
            # Ensure there is at least a generic label
            data["label"] = current_label or "RELATED_TO"

    print(f"Semantic enrichment applied to {enriched_count} edges.")

    return G


# ================================
# STEP 7 — EXPORT FACTS
# ================================
def export_facts(G: nx.DiGraph):
    print(f"Exporting geological facts to {OUTPUT_FACTS}...")

    with open(OUTPUT_FACTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Class_S", "Relation", "Object", "Class_O"])

        for u, v, data in G.edges(data=True):
            cls_u = G.nodes[u].get("class", "OTHER")
            cls_v = G.nodes[v].get("class", "OTHER")
            rel = data.get("label", "RELATED_TO")

            # Export only meaningful facts (ignore double OTHER)
            if cls_u == "OTHER" and cls_v == "OTHER":
                continue

            writer.writerow([u, cls_u, rel, v, cls_v])

    print("Facts export complete.")


# ================================
# MAIN EXECUTION
# ================================
def main():
    print(f"Loading cleaned graph: {INPUT_GEXF}")

    if not os.path.exists(INPUT_GEXF):
        print("ERROR: Run 4.1_clean_graph_full.py first.")
        return

    G = nx.read_gexf(INPUT_GEXF)

    print("Applying semantic enrichment...")
    G = semantic_enrich(G)

    # Save enriched graph
    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"Saved enriched graph to: {OUTPUT_GEXF}")

    # Export facts
    export_facts(G)
    print("DONE.")


if __name__ == "__main__":
    main()
