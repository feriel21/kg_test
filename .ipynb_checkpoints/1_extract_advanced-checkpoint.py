import os
import json
import spacy
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

# ================================
# CONFIGURATION PATHS
# ================================
INPUT_FOLDER = "./output_json"
OUTPUT_FOLDER = "output_graph"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "knowledge_graph_triplets.json")
BATCH_SIZE = 32  # smaller batch, safer with SciBERT

REFERENCE_KG_PATH = "reference/reference_kg.json"

# ================================
# MODEL CONFIG
# ================================
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Similarity threshold to consider a span anchored to the reference KG
REF_SIM_THRESHOLD = 0.45

# Maximum number of tokens allowed in a span (head/tail)
MAX_SPAN_TOKENS = 10

# ================================
# LOAD REFERENCE KG
# ================================
if os.path.exists(REFERENCE_KG_PATH):
    try:
        with open(REFERENCE_KG_PATH, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        REF_TERMS = [t.lower() for t in ref_data.get("canonical_terms", [])]
        print(f">> Loaded {len(REF_TERMS)} reference terms from {REFERENCE_KG_PATH}")
    except Exception as e:
        print(f">> Error loading reference KG: {e}")
        REF_TERMS = []
else:
    print(">> WARNING: reference_kg.json not found. Extraction will not be guided by expert KG.")
    REF_TERMS = []

# ================================
# LOAD SPACY
# ================================
try:
    is_gpu_spacy = spacy.prefer_gpu()
    if is_gpu_spacy:
        print(">> GPU Activated for SpaCy!")
    else:
        print(">> GPU not available or CuPy missing. Using CPU for SpaCy.")
except Exception as e:
    print(f">> GPU Error: {e}. Using CPU for SpaCy.")

try:
    nlp = spacy.load("en_core_sci_sm")
    print("Info: Using SciSpacy model (Science-optimized).")
except Exception:
    nlp = spacy.load("en_core_web_sm")
    print("Info: Using Standard SpaCy model (General purpose).")

# ================================
# LOAD SCIBERT (SentenceTransformer)
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Loading SciBERT ({SCIBERT_MODEL_NAME}) on {device} ...")
sbert_model = SentenceTransformer(SCIBERT_MODEL_NAME, device=device)

if REF_TERMS:
    REF_EMB = sbert_model.encode(REF_TERMS, convert_to_tensor=True)
    print(">> Pre-encoded reference terms for fast similarity checks.")
else:
    REF_EMB = None

# ================================
# UTILITIES
# ================================
BAD_SPAN_KEYWORDS = [
    "figure", "fig.", "image", "photo", "photograph",
    "author", "paper", "article", "study", "section",
    "table", "data", "dataset", "value", "values",
    "result", "results", "method", "methods", "analysis",
    "diagram", "map", "chart", "graph", "model", "equation",
    "supplementary", "appendix", "table", "column", "row"
]


def sanitize_span(text: str) -> str:
    """Clean basic noise from spans."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    # remove leading determiners
    for det in ["the ", "a ", "an ", "this ", "these ", "those "]:
        if text.lower().startswith(det):
            text = text[len(det):]
    # strip punctuation
    text = text.strip(" ,.;:")
    return text


def is_candidate_span(text: str) -> bool:
    """Check if a span is structurally acceptable before semantic checks."""
    t = sanitize_span(text).lower()
    if not t or len(t) < 3:
        return False
    # filter obvious non-content
    if any(bad in t for bad in BAD_SPAN_KEYWORDS):
        return False
    # limit length (tokens)
    if len(t.split()) > MAX_SPAN_TOKENS:
        return False
    # must contain alphabetic chars
    if not any(ch.isalpha() for ch in t):
        return False
    return True


_span_cache = {}  # cache: span -> (canonical, max_sim, anchored_bool)


def anchor_to_reference(span: str):
    """
    Map a span to the closest reference term using SciBERT.
    Returns (canonical_span, max_sim, anchored_bool).
    If no reference KG or low similarity => return original.
    """
    span_clean = sanitize_span(span).lower()

    if span_clean in _span_cache:
        return _span_cache[span_clean]

    if REF_EMB is None or not REF_TERMS:
        # no reference available
        result = (span_clean, 0.0, False)
        _span_cache[span_clean] = result
        return result

    # embed span
    emb = sbert_model.encode(span_clean, convert_to_tensor=True)
    sims = util.cos_sim(emb, REF_EMB)  # [1, n_ref]
    max_sim = float(sims.max().item())
    best_idx = int(torch.argmax(sims).item())
    best_ref = REF_TERMS[best_idx]

    if max_sim >= REF_SIM_THRESHOLD:
        result = (best_ref, max_sim, True)
    else:
        result = (span_clean, max_sim, False)

    _span_cache[span_clean] = result
    return result


def resolve_chunk(token):
    """Return the noun chunk containing the token, or token text as fallback."""
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token.text


# Broader set of verbs potentially relevant in geoscience context
PRIMARY_GEO_VERBS = {
    "fail", "collapse", "transport", "erode", "deposit", "remobilize",
    "slide", "slump", "flow", "deform", "tilt", "uplift", "subside"
}

SECONDARY_GEO_VERBS = {
    "form", "generate", "produce", "create", "control", "influence",
    "characterize", "contain", "occur", "trigger", "initiate", "thicken",
    "thin", "overlie", "underlie", "stack", "fill", "charge"
}

ALLOWED_VERBS = PRIMARY_GEO_VERBS | SECONDARY_GEO_VERBS


def accept_relation_verb(lemma: str) -> bool:
    """Check if a verb lemma is acceptable as a relation."""
    lemma = lemma.lower()
    return lemma in ALLOWED_VERBS


# ================================
# EXTRACTION LOGIC
# ================================
def extract_svo_relations(doc, element_id, filename):
    """
    Extract subject-verb-object relations using dependency parsing,
    then filter and anchor with reference KG.
    """
    triplets = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        relation = token.lemma_.lower()

        if not accept_relation_verb(relation):
            continue

        subj, obj = None, None

        # children dependencies
        for child in token.children:
            # Active subjects
            if child.dep_ in ("nsubj", "csubj"):
                subj = resolve_chunk(child)
            # Direct object / attribute
            if child.dep_ in ("dobj", "attr", "acomp", "pobj"):
                obj = resolve_chunk(child)

            # Passive
            if child.dep_ == "nsubjpass":
                obj = resolve_chunk(child)

            # Agent phrase: "triggered by X"
            if child.dep_ == "agent":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        subj = resolve_chunk(gc)

        if not subj or not obj:
            continue

        subj_clean = sanitize_span(subj)
        obj_clean = sanitize_span(obj)

        if not is_candidate_span(subj_clean) or not is_candidate_span(obj_clean):
            continue

        # Anchor to expert KG (or keep original)
        head_canon, sim_h, anchored_h = anchor_to_reference(subj_clean)
        tail_canon, sim_t, anchored_t = anchor_to_reference(obj_clean)

        # At least one side should be anchored or high-quality
        if not (anchored_h or anchored_t):
            # if both are unanchored and similarity is low => skip
            if max(sim_h, sim_t) < 0.40:
                continue

        # Avoid trivial equality
        if head_canon == tail_canon:
            continue

        triplets.append({
            "head": head_canon,
            "relation": relation,
            "tail": tail_canon,
            "provenance": {
                "source_doc": filename,
                "element_id": element_id,
                "sentence": token.sent.text
            }
        })

    return triplets


def extract_pattern_relations(doc, element_id, filename):
    """
    Extract non-verbal patterns common in geoscience text:
      - X of Y  -> PART_OF
      - X in Y  -> LOCATED_IN
      - X characterized by Y -> CHARACTERIZED_BY
    """
    triplets = []

    # Pattern 1: X of Y  (toe of slope, headwall of failure, etc.)
    for token in doc:
        if token.dep_ == "prep" and token.text.lower() == "of":
            head_token = token.head
            pobj = None
            for child in token.children:
                if child.dep_ == "pobj":
                    pobj = child
                    break
            if head_token is None or pobj is None:
                continue

            span_head = resolve_chunk(head_token)
            span_tail = resolve_chunk(pobj)

            span_head = sanitize_span(span_head)
            span_tail = sanitize_span(span_tail)

            if not is_candidate_span(span_head) or not is_candidate_span(span_tail):
                continue

            h_canon, sim_h, anch_h = anchor_to_reference(span_head)
            t_canon, sim_t, anch_t = anchor_to_reference(span_tail)

            if not (anch_h or anch_t):
                if max(sim_h, sim_t) < 0.40:
                    continue

            if h_canon == t_canon:
                continue

            triplets.append({
                "head": h_canon,
                "relation": "PART_OF",
                "tail": t_canon,
                "provenance": {
                    "source_doc": filename,
                    "element_id": element_id,
                    "sentence": token.sent.text
                }
            })

    # Pattern 2: X in Y (process in location, deposit in basin)
    for token in doc:
        if token.dep_ == "prep" and token.text.lower() == "in":
            head_token = token.head
            pobj = None
            for child in token.children:
                if child.dep_ == "pobj":
                    pobj = child
                    break
            if head_token is None or pobj is None:
                continue

            span_head = resolve_chunk(head_token)
            span_tail = resolve_chunk(pobj)

            span_head = sanitize_span(span_head)
            span_tail = sanitize_span(span_tail)

            if not is_candidate_span(span_head) or not is_candidate_span(span_tail):
                continue

            h_canon, sim_h, anch_h = anchor_to_reference(span_head)
            t_canon, sim_t, anch_t = anchor_to_reference(span_tail)

            if not (anch_h or anch_t):
                if max(sim_h, sim_t) < 0.40:
                    continue

            if h_canon == t_canon:
                continue

            triplets.append({
                "head": h_canon,
                "relation": "LOCATED_IN",
                "tail": t_canon,
                "provenance": {
                    "source_doc": filename,
                    "element_id": element_id,
                    "sentence": token.sent.text
                }
            })

    # Pattern 3: "X is characterized by Y"
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if "characterized by" in sent_text:
            # crude but effective split
            parts = sent_text.split("characterized by")
            if len(parts) != 2:
                continue
            left, right = parts[0], parts[1]
            span_head = sanitize_span(left)
            span_tail = sanitize_span(right)

            if not is_candidate_span(span_head) or not is_candidate_span(span_tail):
                continue

            h_canon, sim_h, anch_h = anchor_to_reference(span_head)
            t_canon, sim_t, anch_t = anchor_to_reference(span_tail)

            if not (anch_h or anch_t):
                if max(sim_h, sim_t) < 0.40:
                    continue

            if h_canon == t_canon:
                continue

            triplets.append({
                "head": h_canon,
                "relation": "CHARACTERIZED_BY",
                "tail": t_canon,
                "provenance": {
                    "source_doc": filename,
                    "element_id": element_id,
                    "sentence": sent.text
                }
            })

    return triplets


def stream_documents():
    """
    Generator for nlp.pipe(): yields (text, context).
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder {INPUT_FOLDER} not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]

    for filename in files:
        try:
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if item.get("type") in ["NarrativeText", "Title"] and item.get("text"):
                    text = item["text"][:100000]
                    context = {
                        "element_id": item.get("element_id"),
                        "filename": filename
                    }
                    yield (text, context)

        except Exception as e:
            print(f"Error reading {filename}: {e}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_triplets = []

    file_count = len([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")])
    print(f"Starting SciBERT-guided extraction on ~{file_count} files...")

    doc_stream = stream_documents()

    for doc, context in tqdm(nlp.pipe(doc_stream, as_tuples=True, batch_size=BATCH_SIZE)):
        filename = context["filename"]
        element_id = context["element_id"]

        trip_svo = extract_svo_relations(doc, element_id, filename)
        trip_pat = extract_pattern_relations(doc, element_id, filename)

        all_triplets.extend(trip_svo)
        all_triplets.extend(trip_pat)

    # Optional: de-duplicate triplets
    unique = []
    seen = set()
    for t in all_triplets:
        key = (t["head"], t["relation"], t["tail"],
               t["provenance"]["source_doc"], t["provenance"]["element_id"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=4)

    print(f"\nExtraction Finished. {len(unique)} triplets saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
