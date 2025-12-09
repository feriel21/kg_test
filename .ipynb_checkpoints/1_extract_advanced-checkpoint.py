#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP 1 — Ontology-guided Extraction
==================================================

"""

import os, json, spacy, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# =============================
# LOAD CONFIG
# =============================
from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

INPUT_FOLDER = paths["json_output"]
OUTPUT_FILE  = paths["triplets"]
OUTPUT_DIR   = os.path.dirname(OUTPUT_FILE)

REFERENCE_KG = cfg["reference"]["reference_kg"]

# =====================================================================
# 1. LOAD REFERENCE KG (nodes + categories)
# =====================================================================
print("Loading reference KG…")
with open(REFERENCE_KG, "r", encoding="utf-8") as f:
    ref = json.load(f)

REF_NODES = list(ref["nodes"].keys())
REF_LOWER = [n.lower() for n in REF_NODES]
REF_CLASSES = {n.lower(): ref["nodes"][n]["main_category"] for n in ref["nodes"]}

print(f">> {len(REF_NODES)} expert concepts loaded.")

# =====================================================================
# 2. LOAD SciBERT
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SciBERT on {device}…")
model = SentenceTransformer("allenai/scibert_scivocab_uncased", device=device)

REF_EMB = model.encode(REF_LOWER, convert_to_tensor=True)

# =====================================================================
# 3. LOAD SPACY
# =====================================================================
try:
    nlp = spacy.load("en_core_sci_sm")
    print("Using SciSpaCy.")
except:
    nlp = spacy.load("en_core_web_sm")
    print("Using spaCy default.")

# =====================================================================
# 4. ONTOLOGY RELATION RULES (FROM PAULINE)
# =====================================================================
def ontology_relation(catA, catB):
    """
    Relations définies par le graphe expert.
    """
    # Environmental control → Mass Movement
    if catA == "Environmental Control (Green)" and catB == "Mass Movement Property (Orange)":
        return "INFLUENCES"

    # Mass Movement → Descriptor
    if catA == "Mass Movement Property (Orange)" and catB == "MTD Descriptor (Blue)":
        return "FORMS"

    # Environmental → Descriptor
    if catA == "Environmental Control (Green)" and catB == "MTD Descriptor (Blue)":
        return "SHAPES"

    # Same category → generic relation
    if catA == catB:
        return "RELATED_TO"

    return "RELATED_TO"

# =====================================================================
# 5. FIND CONCEPTS IN SENTENCE
# =====================================================================
def detect_concepts(sentence_text):
    """
    Retourne la liste des concepts Pauline détectés dans une phrase.
    Détection par similitude SciBERT.
    """
    emb = model.encode(sentence_text.lower(), convert_to_tensor=True)
    sims = util.cos_sim(emb, REF_EMB)[0]

    detected = []
    for idx, score in enumerate(sims):
        if float(score) > 0.48:  # seuil souple
            detected.append(REF_NODES[idx])

    return detected

# =====================================================================
# 6. BUILD TRIPLETS ONTOLOGY-BASED
# =====================================================================
def extract_triplets_from_sentence(sent, fname, element_id):
    text = sent.text
    concepts = detect_concepts(text)

    if len(concepts) < 2:
        return []

    triplets = []

    # Create all pairs
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            A = concepts[i]
            B = concepts[j]

            catA = REF_CLASSES[A.lower()]
            catB = REF_CLASSES[B.lower()]

            rel = ontology_relation(catA, catB)

            triplets.append({
                "head": A,
                "relation": rel,
                "tail": B,
                "provenance": {
                    "source_doc": fname,
                    "element_id": element_id,
                    "sentence": text
                }
            })

    return triplets

# =====================================================================
# 7. STREAM DOCUMENTS
# =====================================================================
def stream_documents():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    for fname in files:
        path = os.path.join(INPUT_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            txt = item.get("text")
            if not txt: continue
            yield txt, fname, item.get("element_id")

# =====================================================================
# 8. MAIN
# =====================================================================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    triplets = []

    print("Starting ontology-guided extraction…")

    for text, fname, eid in tqdm(stream_documents()):
        doc = nlp(text)
        for sent in doc.sents:
            triplets.extend(extract_triplets_from_sentence(sent, fname, eid))

    # Deduplicate
    final = []
    seen = set()
    for t in triplets:
        key = (t["head"], t["relation"], t["tail"], t["provenance"]["source_doc"])
        if key not in seen:
            seen.add(key)
            final.append(t)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4)

    print(f"\nDONE → {len(final)} triplets saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
