# 🌋 MTD Knowledge Graph Pipeline  
*A complete geoscience-aware pipeline for extracting, structuring, enriching and visualizing geological knowledge from scientific PDFs.*

---

## 📂 Dataset (Scientific Articles)

All PDFs used to build the example knowledge graph are publicly available:

👉 **Google Drive (Read-Only)**  
https://drive.google.com/drive/folders/1-Sy0SJQJ8Nq4fODSLoml4vgBWV_LmhU-?usp=sharing

---

# 🎯 Objective

This repository provides a fully automated pipeline that converts a collection of scientific PDF articles into a **clean**, **normalized**, **ontology-aware**, and **visually interpretable** geological Knowledge Graph (KG), specifically designed for:

- Mass Transport Deposits (MTDs)  
- Submarine landslides  
- Slumps / slides / debris flows  
- Slope instability processes  

The pipeline is meant for **geoscientists**, **ML researchers**, and **students**.

---

# 🧬 What the Pipeline Produces

Running the pipeline generates:

### ✔ Clean structured text extracted from PDFs  
### ✔ SciBERT-guided SVO triplets  
### ✔ Normalized geological concepts (SBERT clustering)  
### ✔ Cleaned & filtered KG  
### ✔ Ontology-aware nodes (PROCESS, FEATURE, TRIGGER...)  
### ✔ Semantically enriched relations (CAUSES, FORMS…)  
### ✔ Publication-ready visualizations  
### ✔ Full KG evaluation (coverage, redundancy, cohesion)  
### ✔ Clustered heatmaps + similarity statistics  

---

# 🚀 PipelineOverview 

Run the entire pipeline:

```bash
bash run_all.sh


-----

# 📂 All results are stored in:

    output_json/
    output_graph/
    output_graph/visuals/
    output_graph/visuals_evaluation/

---

### STEP 0 — PDF → JSON Extraction
**Script:** `0_ingest_pdf.py`

- Extracts scientific text blocks
- Cleans sections, paragraphs, and narrative sentences
- Removes figure/table captions, references, noise
- Produces structured JSON ready for NLP extraction

**Output:** `output_json/`

---

### STEP 1 — SciBERT-Guided Triplet Extraction
**Script:** `1_extract_advanced.py`

- Advanced dependency parsing (SciSpacy)
- SVO extraction (subject–verb–object)
- **Geoscience pattern detection:**
  - X of Y → PART_OF
  - X in Y → LOCATED_IN
  - “characterized by” → CHARACTERIZED_BY
- Semantic filtering with SciBERT
- Anchoring to the reference KG (ontology-guided extraction)
- Noise removal based on similarity thresholds

**Output:** `knowledge_graph_triplets.json`

---

### STEP 2 — SBERT Entity Normalization
**Script:** `2_normalize_gpu.py`

- Embeds all concepts with SBERT
- Clusters synonym groups (community detection)
- Canonicalizes geological terms
- **Applies geoscience domain rules:**
  - protected terms
  - critical descriptors
  - merging long forms → short canonical names
- Integrates reference ontology overrides

> **Example merges:**
> * slumping, slumps, slump blocks → **slump**
> * chaotic facies, chaotically bedded facies → **chaotic facies**

**Output:** `entity_map.json`, `clusters_log.txt`

---

### STEP 3 — Knowledge Graph Cleaning
**Script:** `3_clean_graph_full.py`

- **Removes:**
  - meaningless nodes
  - low-similarity nodes
  - weak nodes (degree ≤ 1)
  - isolated subgraphs
  - duplicate edges
- Ensures the KG is clean, compact, and geologically consistent.

**Output:** `final_graph_clean.gexf`

---

### STEP 4 — Semantic Enrichment (Ontology Integration)
**Script:** `4_semantic_enrichment.py`

Adds expert knowledge:
- **Node classification:** `PROCESS`, `FEATURE`, `TRIGGER`, `FACIES`, `LOCATION`, `MATERIAL`
- **Relation enrichment:** `CAUSES`, `FORMS`, `LOCATED_IN`, `EXHIBITS`, `TRANSPORTS`
- Integrates reference ontology categories + rules.

**Output:** `final_graph_semantic.gexf`

---

### STEP 5 — Knowledge Graph Visualization
**Script:** `5_visualize_graph.py`

Generates publication-ready graphs:
- Full KG with ontology color coding
- Top-25 geological subgraph
- Airy layout with readable labels
- Process–Feature–Trigger view
- Degree distribution

**Output:** `output_graph/visuals/`

---

### STEP 6 — KG Quality Evaluation
**Script:** `6_evaluate_kg_quality.py`

Runs your full validation protocol:
- **Semantic metrics:** Exact coverage, Semantic coverage, Similarity matrix, Hallucination detection
- **Structural metrics:** Degree distribution, Weak nodes, Redundancy detection, Cohesion score
- **Geoscience metrics:** Top-K hubs, Concept coherence

**Output:** `evaluation_results.json`

---

### STEP 7 — Evaluation Visualization
**Script:** `7_visualize_evaluation.py`

Creates the visual report:
- Coverage chart
- Similarity histogram
- Improved similarity heatmap
- Redundancy chart
- Hallucination chart
- Cohesion chart
- Top 20 hubs
- Top 20 hallucinations

**Output:** `output_graph/visuals_evaluation/` (Full evaluation report)

---

## 📝 Pipeline Summary

| Step | Script | Purpose |
| :--- | :--- | :--- |
| **0** | `0_ingest_pdf.py` | PDF → JSON |
| **1** | `1_extract_advanced.py` | SciBERT-guided extraction |
| **2** | `2_normalize_gpu.py` | SBERT normalization |
| **3** | `3_clean_graph_full.py` | Graph cleaning |
| **4** | `4_semantic_enrichment.py` | Ontology integration |
| **5** | `5_visualize_graph.py` | KG visualization |
| **6** | `6_evaluate_kg_quality.py` | KG evaluation |
| **7** | `7_visualize_evaluation.py` | Evaluation visualizations |

## 🧠 Technologies Used
* **Python 3**
* **PyMuPDF**
* **spaCy**
* **SciBERT** (`allenai/scibert_scivocab_uncased`)
* **SBERT** (`sentence-transformers`)
* **NetworkX**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

## 🛠 Installation

```bash
git clone <your_repo_url>
cd <your_repo_folder>
pip install -r requirements.txt
bash run_all.sh
