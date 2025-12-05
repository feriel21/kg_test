import networkx as nx
import pandas as pd
import json
import os
from collections import defaultdict, Counter

# ==========================================================
# CONFIG
# ==========================================================
KG_FILE = "output/full_corpus/final_graph_knowledge_layer.gexf"   # change “full” or “small”
OUT_DIR = "node_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading knowledge graph…")
G = nx.read_gexf(KG_FILE)

# ==========================================================
# 1) Extract node classes (MTD ontology)
# ==========================================================
print("Extracting node categories…")
node_classes = {n: G.nodes[n].get("class", "OTHER") for n in G.nodes()}

# ==========================================================
# 2) Extract article provenance for each node
# ==========================================================
print("Extracting provenance from edge sources…")

node_sources = defaultdict(list)

for u, v, data in G.edges(data=True):
    sources = data.get("sources", "").split("|")
    
    for s in sources:
        if s.strip():
            node_sources[u].append(s)
            node_sources[v].append(s)

# convert to count per article
node_sources_count = {
    n: Counter(node_sources[n]) for n in G.nodes()
}

# ==========================================================
# 3) Build main analysis table
# ==========================================================
rows = []

for n in G.nodes():
    cls = node_classes[n]
    src_count = node_sources_count[n]

    rows.append({
        "node": n,
        "class": cls,
        "total_occurrences": sum(src_count.values()),
        "num_articles": len(src_count),
        "articles": "; ".join([f"{k}({v})" for k, v in src_count.items()])
    })

df = pd.DataFrame(rows)
df = df.sort_values("total_occurrences", ascending=False)

df.to_csv(f"{OUT_DIR}/nodes_with_sources_and_classes.csv", index=False)
print("Saved → nodes_with_sources_and_classes.csv")

# ==========================================================
# 4) Group nodes by article
# ==========================================================
article_to_nodes = defaultdict(list)

for node, src_count in node_sources_count.items():
    for article in src_count:
        article_to_nodes[article].append(node)

rows2 = []

for article, nodes in article_to_nodes.items():
    rows2.append({
        "article": article,
        "num_nodes": len(nodes),
        "nodes": ", ".join(nodes)
    })

df2 = pd.DataFrame(rows2)
df2 = df2.sort_values("num_nodes", ascending=False)

df2.to_csv(f"{OUT_DIR}/nodes_grouped_by_article.csv", index=False)
print("Saved → nodes_grouped_by_article.csv")

# ==========================================================
# 5) Detect redundancy / synonym patterns
# ==========================================================
print("Computing redundancy groups…")

# Two nodes are considered "redundant" if they always appear together in the same articles
redundant_pairs = []

nodes_list = list(G.nodes())

for i in range(len(nodes_list)):
    for j in range(i + 1, len(nodes_list)):
        n1, n2 = nodes_list[i], nodes_list[j]
        
        a1 = set(node_sources_count[n1].keys())
        a2 = set(node_sources_count[n2].keys())

        # Appearing in exactly the same articles
        if a1 == a2 and len(a1) > 0:
            redundant_pairs.append((n1, n2))

df3 = pd.DataFrame(redundant_pairs, columns=["node1", "node2"])
df3.to_csv(f"{OUT_DIR}/redundant_node_pairs.csv", index=False)
print("Saved → redundant_node_pairs.csv")

print("\n===== DONE =====")
print(f"All outputs saved in: {OUT_DIR}")
