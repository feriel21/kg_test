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

#*** All results are stored in:**


output_json/
output_graph/
output_graph/visuals/
output_graph/visuals_evaluation/


###Scientific Workflow
1) PDF → Structured Text

ingest_pdf.py
Extracts sentences & narrative blocks using PyMuPDF.

2) SciBERT-Guided Relation Extraction

3_extract_advanced.py
Extracts reliable scientific relations using:

Dependency parsing

SVO rules

Geoscience phrase patterns

SciBERT semantic filtering

Redundancy reduction

Reference KG similarity checks

Produces 40k–70k high-quality triplets.

3) Concept Normalization (SBERT)

4_normalize_gpu.py
Merges synonyms using Sentence-BERT:

Example merges:

slumps, slumping, slump blocks → slump

chaotic facies, chaotically bedded → chaotic facies

Reduces noise and improves KG clarity.

4) Graph Cleaning

4.1_clean_graph_full.py
Removes:

meaningless nodes

weak nodes

low degree noise

isolated components

5) Ontology Classification + Semantic Enrichment

4.2_semantic_enrichment.py

Nodes labeled into:

Class	Examples
PROCESS	slump, slide, debris flow
FEATURE	headwall, toe, scarp
TRIGGER	earthquake, overpressure
FACIES	chaotic facies
LOCATION	slope, basin
MATERIAL	sand, clay

Edges refined into:

CAUSES

FORMS

LOCATED_IN

EXHIBITS

TRANSPORTS

6) High-Quality Visualizations

5_visualize_graph.py

Produces:

Full KG with:

class colors

relation colors

bold readable labels

ID-based graph (airier, publication-ready)

Top-25 geoscience subgraph (Process, Feature, Trigger, etc.)

Degree distribution

Pareto curve

Top hubs

7) Knowledge Graph Evaluation

6_evaluate_kg_quality.py
Generates evaluation_results.json containing:

Coverage vs reference ontology

Hallucinations

Redundancy

Suspicious relations

Weak nodes

Semantic cohesion

Similarity matrices

8) Evaluation Visualizations

7_visualize_evaluation.py

Generates:

Global similarity heatmap

Clustered heatmap

Per-class heatmaps

Similarity histogram

Similarity vs degree scatter

Full evaluation report

📁 Repository Structure
project/
├── data/                      
├── output_json/
├── output_graph/
│   ├── visuals/
│   └── visuals_evaluation/
│
├── ingest_pdf.py
├── 3_extract_advanced.py
├── 4_normalize_gpu.py
├── 4.1_clean_graph_full.py
├── 4.2_semantic_enrichment.py
├── 5_visualize_graph.py
├── 6_evaluate_kg_quality.py
├── 7_visualize_evaluation.py
│
└── run_all.sh

🧠 Technologies Used

Python 3

PyMuPDF

spaCy

SciBERT (allenai/scibert_scivocab_uncased)

SBERT (sentence-transformers)

NetworkX

Matplotlib

Seaborn

Scikit-learn

🛠 Installation
git clone <your_repo_url>
cd <your_repo_folder>
pip install -r requirements.txt
bash run_all.sh
