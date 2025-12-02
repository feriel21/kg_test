import pandas as pd
import json
import os

INPUT_FILE = "reference/Table_Supplementary_1_V2.xlsx"
OUTPUT_FILE = "reference/reference_kg.json"

def main():
    print("Loading reference Excel…")
    df = pd.read_excel(INPUT_FILE)

    canonical_terms = set()
    relations = []
    ontology = {}

    for _, row in df.iterrows():
        label = str(row["Label"])
        rel_type = str(row["Type"])

        # Split "A - B"
        if " - " not in label:
            continue
        
        head, tail = label.split(" - ")
        head = head.strip().lower()
        tail = tail.strip().lower()

        canonical_terms.add(head)
        canonical_terms.add(tail)

        # Basic ontology inference (can be extended later)
        if "mtd" in head:
            ontology[head] = "FEATURE"
        if "mtd" in tail:
            ontology[tail] = "FEATURE"
        if "runout" in head or "runout" in tail:
            ontology[head] = ontology.get(head, "MEASURE")
            ontology[tail] = ontology.get(tail, "MEASURE")

        relations.append({
            "head": head,
            "tail": tail,
            "type": rel_type.lower()
        })

    data = {
        "canonical_terms": sorted(list(canonical_terms)),
        "relations": relations,
        "ontology": ontology
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Reference KG exported → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
