import pandas as pd
import json
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
EXCEL_FILE = 'reference/Table_Supplementary_1_V2.xlsx'
SHEET_EDGES = 'EDGES_TABLE'
SHEET_REFS = 'References'
SHEET_DESC = 'Edges_OrganizedByDescriptor'
OUTPUT_FILENAME = "reference/reference_kg.json"

# ==========================================
# 2. HIERARCHY DEFINITION (The Source of Truth)
# ==========================================
# This defines exactly which Sub-Category belongs to which Color Group.

HIERARCHY = {
    "MTD Descriptor (Blue)": [
        "Morphology",
        "Basal surface",
        "Upper surface",
        "Internal facies distributions",
        "Headscarp",
        "Position",
        "Global environment"
        
    ],
    "Mass Movement Property (Orange)": [
        "Trigger phase",
        "Transport phase",
        "Post-deposition phase"
        
    ],
    "Environmental Control (Green)": [
        "Environmental controls",
        
    ]
}

# --- Auto-generate Lookup Dictionary ---
# Creates a map: "Morphology" -> "MTD Descriptor (Blue)"
SUBCAT_TO_COLOR = {}
for main_group, sub_list in HIERARCHY.items():
    for sub in sub_list:
        SUBCAT_TO_COLOR[sub] = main_group

# ==========================================
# 3. CLASSIFICATION LOGIC
# ==========================================
def get_classification(concept_name, excel_subcat=None):
    """
    Determines the Sub-Category and Main Category (Color).
    """
    # 1. PRIORITY: Excel Classification (from the Descriptor sheet)
    if excel_subcat:
        # Look up the color for this sub-category
        if excel_subcat in SUBCAT_TO_COLOR:
            return {"sub_category": excel_subcat, "main_category": SUBCAT_TO_COLOR[excel_subcat]}
        
        # Fuzzy match (e.g., "Internal facies" matching "Internal facies distributions")
        for known_sub, color in SUBCAT_TO_COLOR.items():
            if known_sub in excel_subcat:
                return {"sub_category": excel_subcat, "main_category": color}

    # 2. DIRECT MATCH: Is the concept name ITSELF a category?
    # Example: The node is named "Slope angle"
    if concept_name in SUBCAT_TO_COLOR:
        return {"sub_category": concept_name, "main_category": SUBCAT_TO_COLOR[concept_name]}

    # 3. KEYWORD RULES (Fallback for unlisted terms)
    name_lower = concept_name.lower()
    
    # Orange Keywords (Processes)
    if any(x in name_lower for x in ["flow", "transport", "slide", "velocity", "trigger", "movement", "kinematic"]):
        return {"sub_category": "Transport phase", "main_category": "Mass Movement Property (Orange)"}
    
    # Green Keywords (Context)
    if any(x in name_lower for x in ["sea level", "basin", "tectonic", "sediment", "slope", "salt", "fault"]):
        return {"sub_category": "Environmental controls", "main_category": "Environmental Control (Green)"}

    # 4. DEFAULT FALLBACK
    # If it's a physical object not listed above, it's likely a Descriptor (Blue)
    return {"sub_category": "Morphology", "main_category": "MTD Descriptor (Blue)"}

# ==========================================
# 4. EXECUTION
# ==========================================
print(f"Loading {EXCEL_FILE}...")

try:
    xls = pd.ExcelFile(EXCEL_FILE)
    df_edges = pd.read_excel(xls, sheet_name=SHEET_EDGES)
    df_refs = pd.read_excel(xls, sheet_name=SHEET_REFS)
    # Load Descriptor sheet, keeping only first 2 cols
    df_desc = pd.read_excel(xls, sheet_name=SHEET_DESC, usecols=[0, 1])
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- A. Build References Dictionary ---
refs_lookup = {}
for _, row in df_refs.iterrows():
    if pd.notna(row.iloc[0]):
        rid = str(row.iloc[0]).split('.')[0].strip()
        refs_lookup[rid] = str(row.iloc[1])

# --- B. Build Concept Map from Excel ---
concept_map_from_excel = {}
df_desc.columns = ['Category', 'Concept']
# CRITICAL: Forward fill to handle merged cells in Excel
df_desc['Category'] = df_desc['Category'].ffill()

for _, row in df_desc.iterrows():
    sub = str(row['Category']).strip()
    concept = str(row['Concept']).strip()
    if concept and concept.lower() != 'nan':
        concept_map_from_excel[concept] = sub

# --- C. Build Graph ---
nodes_dict = {}
edges_list = []

print("Processing graph...")

for _, row in df_edges.iterrows():
    label = str(row.iloc[0])
    
    # Check valid line format
    if " - " not in label: continue
    
    parts = label.split(" - ")
    source = parts[0].strip()
    target = parts[1].strip()

    # Get References
    ref_val = row['Reference #'] if 'Reference #' in row else row.iloc[2]
    citations = []
    if pd.notna(ref_val):
        ids = [x.strip().split('.')[0] for x in str(ref_val).replace('"', '').split(',') if x.strip()]
        citations = [refs_lookup.get(i, "Unknown") for i in ids]

    # Add Edge
    edges_list.append({
        "source": source,
        "target": target,
        "relation": "related_to",
        "citations": citations
    })

    # Classify Nodes (Source & Target)
    for node in [source, target]:
        if node not in nodes_dict:
            # Check if this node was in the Descriptor sheet
            excel_sub = concept_map_from_excel.get(node, None)
            
            # Determine class
            classification = get_classification(node, excel_sub)
            nodes_dict[node] = classification

# ==========================================
# 5. SAVE OUTPUT
# ==========================================
final_data = {
    "project": "MTD Ontology",
    "hierarchy": HIERARCHY,
    "nodes": nodes_dict,
    "edges": edges_list
}

with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print(f"Success! Processed {len(nodes_dict)} nodes.")
print(f"Output saved to: {OUTPUT_FILENAME}")