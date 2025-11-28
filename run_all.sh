#!/bin/bash

echo "======================================================"
echo "   🔵 MTD KNOWLEDGE-GRAPH PIPELINE — FULL EXECUTION"
echo "======================================================"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo " Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "------------------------------------------------------"
echo " STEP 0 — Parsing PDFs → JSON Blocks"
echo "------------------------------------------------------"
python ingest_pdf.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 1 — Domain Discovery from Corpus"
echo "------------------------------------------------------"
python 0_discover_domain_v2.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 2 — Auto-update domain_config.py"
echo "------------------------------------------------------"
python 1_auto_update_config.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 3 — Semantic Analytics (PMI / Heatmap)"
echo "------------------------------------------------------"
python 2_visualize_analytics.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 4 — Extract Triplets (SVO / Entities)"
echo "------------------------------------------------------"
python 3_extract_advanced.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 5 — Normalize Entities (SBERT)"
echo "------------------------------------------------------"
python 4_normalize_gpu.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 6 — Clean Knowledge Graph"
echo "------------------------------------------------------"
python 4.1_clean_graph_full.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 7 — Semantic Enrichment (Ontology Classification)"
echo "------------------------------------------------------"
python 4.2_semantic_enrichment.py || exit 1

echo ""
echo "------------------------------------------------------"
echo " STEP 8 — Graph Visualization (Degree + Communities)"
echo "------------------------------------------------------"
python 5_visualize_graph.py || exit 1

echo ""
echo "======================================================"
echo "    PIPELINE EXECUTED SUCCESSFULLY !"
echo "======================================================"
