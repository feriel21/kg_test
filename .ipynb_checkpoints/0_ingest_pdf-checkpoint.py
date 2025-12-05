import os
import json
import fitz  # PyMuPDF library
import hashlib # Pour générer l'ID unique
from tqdm import tqdm

# --- Configuration ---
from utils.config_loader import load_config
cfg = load_config()
paths = cfg["paths_expanded"]

INPUT_FOLDER = paths["pdf_folder"]        
OUTPUT_FOLDER = paths["json_output"]      

HEADER_MARGIN = 60
FOOTER_MARGIN = 60

# --- Helper Functions ---

def generate_element_id(filename, page_num, bbox):
    """
    Creates a deterministic unique ID based on location.
    Format: MD5 hash of "filename_page_x0_y0"
    """
    unique_string = f"{filename}_{page_num}_{int(bbox[0])}_{int(bbox[1])}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def infer_type(text):
    """
    Simple heuristic to guess if a block is a Title or Text.
    Real parsing would need font size analysis, but this is a good approximation.
    """
    clean_text = text.strip()
    # If short and no terminal punctuation, likely a Title/Header
    if len(clean_text) < 150 and clean_text[-1] not in ['.', ':', ';']:
        return "Title"
    return "NarrativeText"

def process_pdfs():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.pdf')]
    print(f"Starting processing of {len(pdf_files)} articles (Advanced Logic)...")

    for filename in tqdm(pdf_files):
        try:
            doc = fitz.open(os.path.join(INPUT_FOLDER, filename))
            structured_content = []

            for page_num, page in enumerate(doc):
                width = page.rect.width
                height = page.rect.height
                
                blocks = page.get_text("blocks")
                
                # --- IMPROVED SORTING (Hybrid 1-col / 2-col) ---
                def get_sort_key(b):
                    x0, y0, x1, y1 = b[:4]
                    block_width = x1 - x0
                    
                    # If block spans > 80% of page width, treat as "Full Width" (Column 0)
                    # Otherwise, split left/right
                    if block_width > (width * 0.8):
                        column = 0 
                    else:
                        column = 0 if x0 < (width / 2) else 1
                    
                    # Sort primarily by Column, then by Y position
                    return (column, y0)

                blocks.sort(key=get_sort_key)

                for b in blocks:
                    x0, y0, x1, y1, text, block_no, block_type = b
                    text = text.strip()
                    
                    # Filters
                    if not text: continue
                    if y1 < HEADER_MARGIN or y0 > (height - FOOTER_MARGIN): continue

                    # --- METADATA GENERATION ---
                    unique_id = generate_element_id(filename, page_num + 1, (x0, y0, x1, y1))
                    inferred_type = infer_type(text)

                    item = {
                        "element_id": unique_id,  # Traceability Key
                        "text": text,
                        "type": inferred_type,    # Heuristic Type
                        "metadata": {
                            "source_doc": filename,
                            "page_number": page_num + 1,
                            "languages": ["eng"], # Assumed default for science
                            "coordinates": [x0, y0, x1, y1],
                            "is_full_width": (x1 - x0) > (width * 0.8)
                        }
                    }
                    structured_content.append(item)

            output_path = os.path.join(OUTPUT_FOLDER, filename.replace('.pdf', '.json'))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_content, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nProcessing finished. JSON files are in {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_pdfs()