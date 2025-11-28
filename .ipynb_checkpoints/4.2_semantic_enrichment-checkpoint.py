import networkx as nx
import os
import csv

# ================================
# CONFIG
# ================================
INPUT_GEXF = "output_graph/final_graph_clean_final.gexf"
OUTPUT_GEXF = "output_graph/final_graph_knowledge_layer.gexf"
OUTPUT_FACTS = "output_graph/geological_facts.csv"

# ================================
# ONTOLOGY (DOMAIN KNOWLEDGE)
# ================================
ONTOLOGY_CLASSES = {
    "PROCESS": [
        "slide", "slump", "flow", "creep", "avalanche",
        "turbidite", "failure", "movement", "instability",
        "deformation", "transport"
    ],
    "FEATURE": [
        "scarp", "headwall", "toe", "block", "lobe",
        "lens", "channel", "levee", "mound", "geometry",
        "ramp", "surface", "boundary"
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
    """Classify a node as PROCESS, LOCATION, etc."""
    text = label.lower()

    for cls, keywords in ONTOLOGY_CLASSES.items():
        if any(k in text for k in keywords):
            return cls

    return "OTHER"


def refine_edge_relation(cls_u, cls_v):
    """Apply semantic rules to refine relationships."""

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

    return None  # keep existing


# ================================
# STEP 6 — SEMANTIC ENRICHMENT
# ================================
def semantic_enrich(G):
    print("Classifying nodes...")

    # Assign classes + colors
    for node, data in G.nodes(data=True):
        label = node
        cls = get_node_class(label)
        data["class"] = cls

        # Color for Gephi
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
        cls_u = G.nodes[u]["class"]
        cls_v = G.nodes[v]["class"]

        new_rel = refine_edge_relation(cls_u, cls_v)

        if new_rel:
            data["label"] = new_rel
            enriched_count += 1

    print(f"Semantic enrichment applied to {enriched_count} edges.")

    return G


# ================================
# STEP 7 — EXPORT FACTS
# ================================
def export_facts(G):
    print(f"Exporting geological facts to {OUTPUT_FACTS}...")

    with open(OUTPUT_FACTS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Class_S", "Relation", "Object", "Class_O"])

        for u, v, data in G.edges(data=True):
            cls_u = G.nodes[u]["class"]
            cls_v = G.nodes[v]["class"]
            rel = data.get("label", "RELATED_TO")

            # export only meaningful facts
            if cls_u != "OTHER" and cls_v != "OTHER":
                writer.writerow([u, cls_u, rel, v, cls_v])


# ================================
# MAIN EXECUTION
# ================================
def main():
    print(f"Loading cleaned graph: {INPUT_GEXF}")

    if not os.path.exists(INPUT_GEXF):
        print("ERROR: Run clean_graph_full first.")
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
