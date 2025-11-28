import networkx as nx
import os

INPUT_GEXF = "output_graph/final_graph_modular.gexf"
OUTPUT_GEXF = "output_graph/final_graph_clean_final.gexf"

# -------- SEMANTIC BLOCKERS -------- #
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

MIN_LEN = 4  # skip small tokens like "top", "mid", "sea"?

def is_noise(node: str):
    text = node.lower().strip()

    # 1. forbidden list
    if text in FORBIDDEN:
        return True

    # 2. too short
    if len(text) < MIN_LEN:
        return True

    # 3. non alphabetic (remove weird chars)
    clean = text.replace(" ", "").replace("-", "")
    if not clean.isalpha():
        return True

    return False


def main():
    print(f"Loading graph : {INPUT_GEXF}")
    G = nx.read_gexf(INPUT_GEXF)
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ----------------------------------------------------------
    # 1 — STRUCTURAL CLEANING
    # ----------------------------------------------------------
    # remove degree < 3
    deg_rm = [n for n, d in G.degree() if d < 3]
    G.remove_nodes_from(deg_rm)
    print(f"After degree filter: {G.number_of_nodes()} nodes")

    # largest connected component
    if G.number_of_nodes() > 0:
        lcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
    print(f"After LCC: {G.number_of_nodes()} nodes")

    # ----------------------------------------------------------
    # 2 — SEMANTIC CLEANING
    # ----------------------------------------------------------
    noise = [n for n in G.nodes() if is_noise(n)]
    print(f"Semantic noise removed: {len(noise)}")
    G.remove_nodes_from(noise)

    # final LCC
    if G.number_of_nodes() > 0:
        lcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    print(f"Final cleaned graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"\nSaved cleaned graph → {OUTPUT_GEXF}")


if __name__ == "__main__":
    main()
