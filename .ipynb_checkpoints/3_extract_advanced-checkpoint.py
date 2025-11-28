import os
import json
import spacy
from tqdm import tqdm

# --- IMPORT DOMAIN CONFIGURATION ---
# This imports the keyword list defined in domain_config.py
from domain_config import DOMAIN_KEYWORDS

# --- CONFIGURATION PATHS ---
INPUT_FOLDER = "./output_json"
OUTPUT_FOLDER = "output_graph"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "knowledge_graph_triplets.json")
BATCH_SIZE = 50  # Process 50 documents at a time (Optimization)

# --- GPU ACTIVATION ---
# We attempt to activate the GPU. If CuPy is not installed, it falls back to CPU safely.
try:
    is_gpu = spacy.prefer_gpu()
    if is_gpu:
        print(">> GPU Activated for Spacy!")
    else:
        print(">> GPU not available or CuPy missing. Using CPU.")
except Exception as e:
    print(f">> GPU Error: {e}. Using CPU.")

# Load Spacy Model
try:
    nlp = spacy.load("en_core_sci_sm")
    print("Info: Using SciSpacy model (Science-optimized).")
except:
    nlp = spacy.load("en_core_web_sm")
    print("Info: Using Standard Spacy model (General purpose).")

def is_domain_relevant(text):
    """
    Checks if the text contains any keyword from the external configuration file.
    """
    text_lower = text.lower()
    return any(k in text_lower for k in DOMAIN_KEYWORDS)

def resolve_references(token):
    """
    Captures the full Noun Chunk (e.g., 'The massive slump' instead of just 'slump').
    """
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    return token.text

def extract_relations_from_doc(doc, element_id, filename):
    """
    Core Logic: Dependency Parsing (Subject -> Verb -> Object).
    """
    triplets = []
    for token in doc:
        # We pivot around the VERB
        if token.pos_ == "VERB":
            subj, obj = None, None
            relation = token.lemma_ # Base form of the verb
            
            # Analyze children tokens to find Subject and Object
            for child in token.children:
                # Active Voice (Earthquake triggers MTD)
                if child.dep_ in ("nsubj", "csubj"):
                    subj = resolve_references(child)
                elif child.dep_ in ("dobj", "attr", "acomp"):
                    obj = resolve_references(child)
                
                # Passive Voice (MTD was triggered by Earthquake)
                if child.dep_ == "nsubjpass":
                    obj = resolve_references(child) # In passive, subj is the target (object)
                if child.dep_ == "agent":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            subj = resolve_references(grandchild)

            # Validate and Filter
            if subj and obj:
                subj = subj.strip()
                obj = obj.strip()
                
                # Check against Domain Keywords (from config file)
                if is_domain_relevant(subj) or is_domain_relevant(obj):
                    triplets.append({
                        "head": subj,
                        "relation": relation,
                        "tail": obj,
                        "provenance": {
                            "source_doc": filename,
                            "element_id": element_id,
                            "sentence": token.sent.text
                        }
                    })
    return triplets

def stream_documents():
    """
    Generator function to feed nlp.pipe().
    This is much more memory efficient than loading everything at once.
    Yields: (text, context_metadata)
    """
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder {INPUT_FOLDER} not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    
    for filename in files:
        try:
            with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if item.get("type") in ["NarrativeText", "Title"] and item.get("text"):
                    # Truncate to 100k chars for safety
                    text = item["text"][:100000]
                    # We pass metadata along with the text
                    context = {"element_id": item.get("element_id"), "filename": filename}
                    yield (text, context)
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    all_triplets = []
    
    # Count files for progress bar estimation (approximate)
    file_count = len([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')])
    print(f"Starting High-Performance Extraction on ~{file_count} files...")

    # --- BATCH PROCESSING PIPELINE ---
    # nlp.pipe is critical for GPU usage. 
    # 'as_tuples=True' allows us to pass the metadata (filename, id) through the pipeline.
    doc_stream = stream_documents()
    
    for doc, context in tqdm(nlp.pipe(doc_stream, as_tuples=True, batch_size=BATCH_SIZE)):
        filename = context["filename"]
        element_id = context["element_id"]
        
        found = extract_relations_from_doc(doc, element_id, filename)
        all_triplets.extend(found)

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=4)
        
    print(f"\nExtraction Finished. {len(all_triplets)} triplets saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()