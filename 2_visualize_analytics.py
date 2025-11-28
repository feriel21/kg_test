import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import nltk

# --- IMPORT DOMAIN CONFIGURATION ---
try:
    from domain_config import DOMAIN_KEYWORDS
except ImportError:
    print("Error: domain_config.py not found. Ensure it is in the same directory.")
    exit(1)

# --- CONFIGURATION ---
INPUT_FOLDER = "./output_json"
OUTPUT_FOLDER = "output_graph"
TOP_N_FOR_HEATMAP = 25 # Keep only the top 25 most frequent words for readable heatmap

def setup_nltk():
    """Ensure NLTK tokenizer is available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def load_corpus_sentences():
    """Reads JSON files and returns a flat list of sentences."""
    print(f"Loading corpus from {INPUT_FOLDER}...")
    sentences = []
    if not os.path.exists(INPUT_FOLDER): return []

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    for filename in tqdm(files):
        try:
            with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract narrative text blocks
                text_blocks = [i['text'] for i in data if i.get('text') and i.get('type') == "NarrativeText"]
                full_text = " ".join(text_blocks)
                # Split into sentences for context analysis (intra-sentence co-occurrence)
                sents = nltk.sent_tokenize(full_text.lower())
                sentences.extend(sents)
        except: pass
    return sentences

def calculate_pmi_matrix(sentences, keywords):
    """
    Calculates the Positive Pointwise Mutual Information (PPMI) matrix.
    PPMI measures how much more likely two words are to occur together than by chance.
    """
    print("Calculating PPMI Matrix (Semantic Associations)...")
    
    # 1. Count frequencies
    word_counts = Counter()
    co_occurrence_counts = defaultdict(int)
    total_windows = len(sentences)
    
    # Filter sentences to only keep relevant keywords
    keyword_set = set(keywords)
    
    for sent in tqdm(sentences):
        # Find which keywords appear in this sentence
        present_keywords = [kw for kw in keywords if kw in sent]
        
        # Update individual counts (Set ensures we count once per sentence)
        for kw in set(present_keywords): 
            word_counts[kw] += 1
            
        # Update co-occurrence counts (Pairs)
        # If 'slump' and 'failure' are in the same sentence -> +1 link
        for i in range(len(present_keywords)):
            for j in range(i + 1, len(present_keywords)):
                w1, w2 = sorted([present_keywords[i], present_keywords[j]])
                if w1 != w2:
                    co_occurrence_counts[(w1, w2)] += 1

    # 2. Build the PMI Matrix
    # PMI(x, y) = log2( P(x,y) / (P(x) * P(y)) )
    # PPMI = max(PMI, 0) -> We only care about positive associations
    
    matrix_data = pd.DataFrame(0.0, index=keywords, columns=keywords)
    
    for (w1, w2), count_xy in co_occurrence_counts.items():
        count_x = word_counts[w1]
        count_y = word_counts[w2]
        
        if count_x > 0 and count_y > 0:
            p_x = count_x / total_windows
            p_y = count_y / total_windows
            p_xy = count_xy / total_windows
            
            # Avoid log(0) error
            if p_xy > 0:
                pmi = math.log2(p_xy / (p_x * p_y))
                ppmi = max(pmi, 0) # Keep only positive associations
                
                matrix_data.at[w1, w2] = ppmi
                matrix_data.at[w2, w1] = ppmi # Symmetric matrix

    return matrix_data, word_counts

def plot_heatmap(matrix, filename):
    """Generates a heatmap of semantic associations."""
    plt.figure(figsize=(14, 12))
    # Mask the upper triangle (redundant in symmetric matrix)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    sns.heatmap(
        matrix, 
        mask=mask,
        annot=False, # Set to True to see numbers (might be cluttered)
        cmap="viridis", # Professional color map (Cold -> Hot)
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5, "label": "Semantic Strength (PPMI)"}
    )
    
    plt.title("Semantic Association Heatmap (PPMI)\nStrong connections between geological concepts", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f" -> Heatmap saved: {filename}")
    plt.close()

def plot_barchart(word_counts, top_n, filename):
    """Generates a bar chart of the most frequent concepts."""
    # Sort and pick top N
    common = word_counts.most_common(top_n)
    words = [x[0] for x in common]
    counts = [x[1] for x in common]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts, y=words, palette="magma")
    
    plt.title(f"Top {top_n} Concepts Detected in Corpus", fontsize=16)
    plt.xlabel("Number of Mentions (Sentence Frequency)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f" -> Bar Chart saved: {filename}")
    plt.close()

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    setup_nltk()
    
    sentences = load_corpus_sentences()
    if not sentences: 
        print("No text found in input folder.")
        return

    # Filter: We use the list from Domain Config
    # But we only keep the Top N most frequent for the Heatmap (otherwise unreadable)
    print("Analyzing frequencies for selection...")
    temp_counts = Counter()
    for sent in sentences:
        for kw in DOMAIN_KEYWORDS:
            if kw in sent: temp_counts[kw] += 1
    
    # Select top keywords for the matrix
    top_keywords = [x[0] for x in temp_counts.most_common(TOP_N_FOR_HEATMAP)]
    
    # Computations
    ppmi_matrix, final_counts = calculate_pmi_matrix(sentences, top_keywords)
    
    # Generate plots
    heatmap_path = os.path.join(OUTPUT_FOLDER, "viz_ppmi_heatmap.png")
    plot_heatmap(ppmi_matrix, heatmap_path)
    
    barchart_path = os.path.join(OUTPUT_FOLDER, "viz_concept_frequency.png")
    plot_barchart(final_counts, 30, barchart_path)

    print("\nVisualization complete. Please download PNG files to your local machine.")

if __name__ == "__main__":
    main()