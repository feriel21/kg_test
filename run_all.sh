#!/bin/bash

echo "======================================================"
echo "    MTD KNOWLEDGE-GRAPH PIPELINE — FULL EXECUTION"
echo "======================================================"
echo ""

# Activate venv
if [ -d "venv" ]; then
    echo " Activating virtual environment..."
    source venv/bin/activate
fi


echo ">>> Running pipeline for dataset: $(jq -r .dataset_name config/pipeline_config.json)"

echo ""
echo "------------------------------------------------------"
echo " STEP 0 — Extract PDFs → JSON"
echo "------------------------------------------------------"
python 0_ingest_pdf.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 1 — SciBERT-guided Triplet Extraction"
echo "------------------------------------------------------"
python 1_extract_advanced.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 2 — Normalize Entities (SBERT embeddings)"
echo "------------------------------------------------------"
python 2_normalize_gpu.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 3 — Clean Knowledge Graph"
echo "------------------------------------------------------"
python 3_clean_graph_full.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 4 — Semantic Enrichment (Ontology)"
echo "------------------------------------------------------"
python 4_semantic_enrichment.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 5 — Graph Visualization"
echo "------------------------------------------------------"
python 5_visualize_graph.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 6 — KG Quality Evaluation"
echo "------------------------------------------------------"
python 6_evaluate_kg_quality.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 7 — Evaluation Visualization"
echo "------------------------------------------------------"
python 7_visualize_evaluation.py || exit 1

echo ""
echo "======================================================"
echo "     PIPELINE EXECUTED SUCCESSFULLY 🎉"
echo "======================================================"
