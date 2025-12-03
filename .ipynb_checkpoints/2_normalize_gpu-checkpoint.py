import json
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
import re
from collections import Counter

# ---------------------------------------------------------
# --- LOAD GEOLOGICAL REFERENCE GRAPH (NEW SECTION) -------
# ---------------------------------------------------------
REFERENCE_KG_PATH = "reference/reference_kg.json"

if os.path.exists(REFERENCE_KG_PATH):
    with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    REF_TERMS = set(ref_data.get("canonical_terms", []))
    REF_ONTOLOGY = ref_data.get("ontology", {})
else:
    print("⚠️ WARNING: reference_kg.json not found. Continuing without expert constraints.")
    REF_TERMS = set()
    REF_ONTOLOGY = {}
# ---------------------------------------------------------

# --- IMPORT DOMAIN CONFIGURATION ---
# This ensures consistency with the Extraction Step
from domain_config import CRITICAL_DESCRIPTORS, PROTECTED_TERMS, RELATION_MAP

# --- CONFIGURATION PATHS ---
INPUT_FILE = "output_graph/knowledge_graph_triplets.json"
OUTPUT_FOLDER = "output_graph"
OUTPUT_GRAPH_FILE = os.path.join(OUTPUT_FOLDER, "final_graph_modular.gexf")
CLUSTERS_LOG_FILE = os.path.join(OUTPUT_FOLDER, "clusters_log_modular.txt")

# Similarity Threshold (0.90 = High Precision)
# We want to distinguish "Sediment" from "Sedimentation"
SIMILARITY_THRESHOLD = 0.90
MODEL_NAME = 'all-MiniLM-L6-v2'


def sanitize_text(text):
    """
    Cleans text and removes leading determinants (The, A, An).
    """
    if not isinstance(text, str):
        return str(text)
    # Remove non-printable chars
    text = "".join(ch for ch in text if ch.isprintable())
    # Remove starting articles
    text = re.sub(r'^(the|a|an|these|those)\s+', '', text.strip(), flags=re.IGNORECASE)
    return text.strip()


def smart_simplify_label(text):
    """
    Applies intelligent simplification rules defined in domain_config.py.
    Goal: Keep technical terms precise (e.g. 'Chaotic Facies') but shorten generic ones.
    """
    text_lower = text.lower()

    # Rule 1: Protected Terms (Never change these)
    if text_lower in PROTECTED_TERMS:
        return text

    words = text.split()

    # Rule 2: Critical Descriptors (Allow longer names if technical)
    # Example: 'chaotic seismic facies' (3 words) -> Kept because 'facies' is critical
    is_critical = any(crit in text_lower for crit in CRITICAL_DESCRIPTORS)

    if is_critical:
        # If extremely long (>4 words), keep only the last 3 words
        if len(words) > 4:
            return " ".join(words[-3:])
        return text

    # Rule 3: General/Generic Terms (Cut to 1 word)
    # Example: 'large submarine landslide' -> 'landslide'
    if len(words) > 3:
        return words[-1]

    return text


def normalize_relation(verb):
    """
    Maps raw verbs to the Ontology defined in domain_config.py.
    """
    verb = verb.lower()
    for category, keywords in RELATION_MAP.items():
        if any(k in verb for k in keywords):
            return category
    return "RELATED_TO"  # Default fallback


def load_data():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run Step 2 first.")
        exit(1)


def get_canonical_entities(triplets, model, device):
    print("Counting frequencies & Filtering candidates...")
    entity_counts = Counter()
    candidates = set()

    for item in triplets:
        h = sanitize_text(item['head'].lower())
        t = sanitize_text(item['tail'].lower())

        if h:
            candidates.add(h)
            entity_counts[h] += 1
        if t:
            candidates.add(t)
            entity_counts[t] += 1

    unique_list = list(candidates)
    print(f"Vectorizing {len(unique_list)} entities on {device.upper()}...")

    # ---------------------------------------------------------
    # --- Reference-guided mapping (priority fusion) [PART C] -
    # ---------------------------------------------------------
    def map_to_reference(term):
        """
        Very simple expert mapping:
        if the term is already exactly a canonical expert term, keep it.
        (You can later extend this with similarity-based matching.)
        """
        if term in REF_TERMS:
            return term
        return None

    reference_map = {}
    for t in unique_list:
        ref = map_to_reference(t)
        if ref:
            reference_map[t] = ref
    # ---------------------------------------------------------

    # Compute Embeddings and cluster entities
    if unique_list:
        embeddings = model.encode(unique_list, convert_to_tensor=True, device=device)
        print(f"Clustering with threshold {SIMILARITY_THRESHOLD}...")
        clusters = util.community_detection(
            embeddings,
            min_community_size=1,
            threshold=SIMILARITY_THRESHOLD
        )
    else:
        clusters = []

    entity_map = {}
    log_lines = []

    for cluster in clusters:
        cluster_terms = [unique_list[i] for i in cluster]

        # Selection Strategy: Pick the most FREQUENT term in the cluster
        # Frequency is better than brevity for scientific accuracy.
        most_frequent = max(cluster_terms, key=lambda t: entity_counts[t])

        # Apply the Smart Simplification logic to the chosen representative
        canonical = smart_simplify_label(most_frequent)

        # Log merges for audit
        if len(cluster_terms) > 1:
            log_lines.append(f"Canonical: [{canonical}] merged: {cluster_terms}")

        for term in cluster_terms:
            entity_map[term] = canonical

    # ---------------------------------------------------------
    # Merge reference-guided mapping on top of clustering
    # (Expert terms override purely statistical canonical labels)
    # ---------------------------------------------------------
    entity_map.update(reference_map)
    # ---------------------------------------------------------

    # Save Log
    with open(CLUSTERS_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))
    print(f"Clustering Log saved to {CLUSTERS_LOG_FILE}")

    return entity_map


def build_graph(triplets, entity_map):
    G = nx.DiGraph()
    print("Building Graph...")
    for item in tqdm(triplets):
        head_raw = sanitize_text(item['head'].lower())
        tail_raw = sanitize_text(item['tail'].lower())

        # Map to canonical name OR simplify the raw name if no cluster found
        head = entity_map.get(head_raw, smart_simplify_label(head_raw))
        tail = entity_map.get(tail_raw, smart_simplify_label(tail_raw))

        raw_rel = sanitize_text(item['relation'])
        norm_rel = normalize_relation(raw_rel)
        source = sanitize_text(item['provenance']['source_doc'])

        if not head or not tail or head == tail:
            continue

        # ---------------------------------------------------------
        # --- Assign ontology if available [PART A] ---------------
        # ---------------------------------------------------------
        head_class = REF_ONTOLOGY.get(head, "UNKNOWN")
        tail_class = REF_ONTOLOGY.get(tail, "UNKNOWN")

        if head not in G:
            G.add_node(head, category=head_class)
        if tail not in G:
            G.add_node(tail, category=tail_class)
        # ---------------------------------------------------------

        # Add Edge with weight
        if G.has_edge(head, tail):
            G[head][tail]['weight'] += 1
            G[head][tail]['sources'].add(source)

            # Count relation types
            if norm_rel in G[head][tail].get('rel_types', {}):
                G[head][tail]['rel_types'][norm_rel] += 1
            else:
                G[head][tail]['rel_types'][norm_rel] = 1
        else:
            G.add_edge(
                head,
                tail,
                weight=1,
                sources={source},
                rel_types={norm_rel: 1}
            )

    return G


def save_graph(G):
    # Format attributes for Gephi export
    for u, v, data in G.edges(data=True):
        data['sources'] = "|".join(data['sources'])

        rel_types = data.get('rel_types', {})
        rel_list = [f"{k}({v})" for k, v in rel_types.items()]
        data['relations_summary'] = "|".join(rel_list)

        # Determine main label for the edge
        if rel_types:
            data['label'] = max(rel_types, key=rel_types.get)

        if 'rel_types' in data:
            del data['rel_types']

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    nx.write_gexf(G, OUTPUT_GRAPH_FILE)
    print(f"Graph Saved successfully: {OUTPUT_GRAPH_FILE}")


def main():
    # Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    triplets = load_data()

    # IMPORTANT: Pass 'device' to avoid crashes on non-GPU nodes
    entity_mapping = get_canonical_entities(triplets, model, device)

    G = build_graph(triplets, entity_mapping)
    save_graph(G)


if __name__ == "__main__":
    main()
