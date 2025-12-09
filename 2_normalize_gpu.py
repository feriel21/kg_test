#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP 2 — Graph Construction with Ontology Filtering + Deduplication
===================================================================

Pipeline :

1) DEDUP — remove duplicate triplets BEFORE graph construction
2) FILTER A — remove triplets with irrelevant / UNCLASSIFIED nodes
3) FILTER B — enforce Pauline’s ontology (allowed transitions)
4) BUILD GRAPH — accumulate weights, sources, sentences
5) EXPORT — compress relations, sources, sentences
"""

import json
import os
import re
import networkx as nx
from utils.config_loader import load_config

cfg = load_config()
paths = cfg["paths_expanded"]

INPUT_FILE  = paths["triplets"]
OUTPUT_GEXF = paths["graph_modular"]
REFERENCE_KG = cfg["reference"]["reference_kg"]

os.makedirs(os.path.dirname(OUTPUT_GEXF), exist_ok=True)

# -------------------------------------------------------
# LOAD EXPERT KG
# -------------------------------------------------------
print("[STEP2] Loading expert Pauline KG…")
ref = json.load(open(REFERENCE_KG, "r", encoding="utf-8"))

EXPERT_NODES = ref["nodes"]  # dict: term → {sub_category, main_category}
print(f"[STEP2] Expert nodes available: {len(EXPERT_NODES)}")


# -------------------------------------------------------
# SANITIZATION
# -------------------------------------------------------
def clean(t):
    if not isinstance(t, str):
        t = str(t)
    t = "".join(ch for ch in t if ch.isprintable())
    t = re.sub(r"\s+", " ", t.strip())
    return t.lower()


# -------------------------------------------------------
# DEDUPLICATION
# -------------------------------------------------------
def deduplicate(triplets):
    """
    Remove redundant triplets:
    (head, relation, tail, source_doc)
    """
    seen = set()
    unique = []

    for tr in triplets:
        key = (
            clean(tr["head"]),
            clean(tr["relation"]),
            clean(tr["tail"]),
            tr["provenance"]["source_doc"]
        )
        if key not in seen:
            seen.add(key)
            unique.append(tr)

    print(f"[STEP2] Dedup: {len(triplets)} → {len(unique)} triplets")
    return unique


# -------------------------------------------------------
# FILTER B — Allowed ontology transitions
# -------------------------------------------------------
def allowed_relation(catA, catB):
    """
    ONLY transitions approved by Pauline’s ontology.
    """

    # Same category = valid
    if catA == catB:
        return True

    # Environmental → Mass Movement
    if catA.startswith("Environmental") and catB.startswith("Mass"):
        return True

    # Mass Movement → Descriptor
    if catA.startswith("Mass") and catB.startswith("MTD"):
        return True

    # Environmental → Descriptor
    if catA.startswith("Environmental") and catB.startswith("MTD"):
        return True

    return False


# -------------------------------------------------------
# LOAD TRIPLETS
# -------------------------------------------------------
def load_triplets():
    print(f"[STEP2] Loading triplets: {INPUT_FILE}")
    return json.load(open(INPUT_FILE, "r", encoding="utf-8"))


# -------------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------------
def build_graph(triplets):
    G = nx.DiGraph()
    print("[STEP2] Building ontology-filtered graph…")

    for tr in triplets:
        head = clean(tr["head"])
        tail = clean(tr["tail"])
        rel  = clean(tr["relation"])
        src  = tr["provenance"]["source_doc"]
        sent = clean(tr["provenance"]["sentence"])

        # FILTER A — basic cleaning
        if not head or not tail or head == tail:
            continue

        head_cat = EXPERT_NODES.get(head, {}).get("main_category", "UNCLASSIFIED")
        tail_cat = EXPERT_NODES.get(tail, {}).get("main_category", "UNCLASSIFIED")

        # FILTER A.2 : skip if both nodes are irrelevant
        if head_cat == "UNCLASSIFIED" and tail_cat == "UNCLASSIFIED":
            continue

        # FILTER B — ontology filtering
        if not allowed_relation(head_cat, tail_cat):
            continue

        # --- NODE CREATION ---
        if head not in G:
            G.add_node(
                head,
                category=head_cat,
                sub_category=EXPERT_NODES.get(head, {}).get("sub_category")
            )

        if tail not in G:
            G.add_node(
                tail,
                category=tail_cat,
                sub_category=EXPERT_NODES.get(tail, {}).get("sub_category")
            )

        # --- EDGE CREATION ---
        if G.has_edge(head, tail):
            G[head][tail]["weight"] += 1
            G[head][tail]["sources"].add(src)
            G[head][tail]["sentences"].add(sent)
            G[head][tail]["relations"][rel] = G[head][tail]["relations"].get(rel, 0) + 1

        else:
            G.add_edge(
                head, tail,
                weight=1,
                sources={src},
                sentences={sent},
                relations={rel: 1}
            )

    return G

# -------------------------------------------------------
# REPORT: Missing & Extra Concepts
# -------------------------------------------------------
def compare_with_reference(G, ref_nodes, out_path):
    auto_nodes = set(G.nodes())
    ref_nodes  = set(ref_nodes)

    missing_in_auto = sorted(list(ref_nodes - auto_nodes))
    extra_in_auto   = sorted(list(auto_nodes - ref_nodes))

    report = {
        "total_ref_nodes": len(ref_nodes),
        "total_auto_nodes": len(auto_nodes),
        "missing_from_auto": missing_in_auto,
        "extra_in_auto": extra_in_auto,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[STEP2] Coverage report saved → {out_path}")

# -------------------------------------------------------
# EXPORT
# -------------------------------------------------------
def finalize_export(G):
    print("[STEP2] Finalizing export…")

    for u, v, data in G.edges(data=True):

        data["sources"] = "|".join(sorted(data["sources"]))
        data["sentences"] = "|".join(list(data["sentences"]))

        # Most frequent relation = final label
        rel_types = data["relations"]
        data["label"] = max(rel_types, key=rel_types.get)

        data["relations_summary"] = "|".join(f"{k}({v})" for k, v in rel_types.items())
        del data["relations"]

    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"[STEP2] Saved → {OUTPUT_GEXF}")
def export_node_summary(G, out_path):
    summary = {}

    for n, data in G.nodes(data=True):

        in_deg  = G.in_degree(n)
        out_deg = G.out_degree(n)
        deg     = G.degree(n)

        # collect all sources from edges touching this node
        srcs = set()
        sentences = 0
        for u, v, edata in G.in_edges(n, data=True):
            srcs.update(edata["sources"].split("|"))
            sentences += len(edata["sentences"].split("|"))
        for u, v, edata in G.out_edges(n, data=True):
            srcs.update(edata["sources"].split("|"))
            sentences += len(edata["sentences"].split("|"))

        summary[n] = {
            "category": data.get("category", "UNCLASSIFIED"),
            "sub_category": data.get("sub_category", None),
            "degree_total": deg,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "num_sources": len(srcs),
            "num_sentences": sentences
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"[STEP2] Node summary saved → {out_path}")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    trip = load_triplets()
    trip = deduplicate(trip)
    G = build_graph(trip)
    finalize_export(G)
    summary_path = OUTPUT_GEXF.replace(".gexf", "_nodes.json")
    export_node_summary(G, summary_path)
    coverage_path = OUTPUT_GEXF.replace(".gexf", "_coverage.json")
    compare_with_reference(G, EXPERT_NODES.keys(), coverage_path)
    print(f"[STEP2] DONE — {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()
