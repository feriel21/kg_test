import os
import json
import nltk
import re
import math
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util


# ==========================
#  1. Expert Core Vocabulary
# ==========================
# These terms are ALWAYS included (domain experts).
EXPERT_KEYWORDS = [
    "mass transport", "mtd", "mass transport deposit",
    "slump", "slide", "debris flow", "turbidite", "landslide", "megaslide",
    "deposit", "sediment", "basin", "fault", "channel", "levee", "fan",
    "margin", "slope", "canyon", "drift", "contourite", "clinoform",
    "block", "lobe", "toe", "headwall", "scarp", "basal shear surface",
    "chaotic facies", "transparent facies", "seismic facies",
    "amplitude", "reflection", "reflector", "continuous", "discontinuous",
    "hummocky", "stratified", "acoustic", "echofacies",
    "overpressure", "instability", "tectonic", "erosion",
    "transport", "liquefaction", "loading"
]


# ==========================
#  2. Configuration
# ==========================
INPUT_FOLDER = "./output_json"
TOP_N_AI_KEYWORDS = 60
OUTPUT_CONFIG_FILE = "domain_config.py"

# Geological roots for filtering
GEO_ROOTS = [
    "seism", "fault", "fract", "fail", "shear", "stress", "strain",
    "veloc", "pressur", "stabil", "kinemat", "mechan",
    "lith", "strat", "sediment", "deposit", "turbid", "facies",
    "clast", "sand", "clay", "mud", "eros", "slump", "slide",
    "mass", "transport", "debris", "flow", "block", "lobe",
    "slope", "basin", "margin", "channel", "levee", "fan", "canyon",
    "toe", "headwall", "scarp", "basal", "surface", "ramp", "front",
    "horizon", "geomorph", "structure", "morph", "reflect", "amplit"
]

# Academic noise
ACADEMIC_STOPWORDS = {
    "figure", "fig", "table", "doi", "issn", "isbn", "http", "www", "org",
    "et", "al", "introduction", "conclusion", "results", "discussion",
    "study", "paper", "analysis", "method", "model", "value", "equation",
    "parameter", "example", "associated", "different", "data"
}

lemmatizer = WordNetLemmatizer()


# ==========================
#  Utility Functions
# ==========================
def setup_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)


def clean_term(term):
    """Strict cleaning + lemmatization + geological filter."""
    term = term.lower().strip()
    words = term.split()

    if not words:
        return None

    # Lemmatize last word (noun)
    words[-1] = lemmatizer.lemmatize(words[-1], pos='n')
    term = " ".join(words)

    # Reject short tokens
    if len(term) < 4:
        return None

    # Reject if too long
    if len(words) > 4:
        return None

    # Reject numbers or special symbols
    if re.search(r'[0-9µ×=+/\\<>{}()$%]', term):
        return None

    # Reject academic noise
    if any(w in ACADEMIC_STOPWORDS for w in words):
        return None

    # MUST contain a geological root
    if not any(root in term for root in GEO_ROOTS):
        return None

    return term


# ==========================
#  Load PDF Corpus
# ==========================
def load_corpus():
    print(f"Loading text from {INPUT_FOLDER}...")
    corpus = []
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]

    for filename in tqdm(files):
        try:
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                text = " ".join(
                    i['text'] for i in data
                    if i.get("text") and i.get("type") == "NarrativeText"
                )
                corpus.append(text)
        except:
            pass

    return corpus


# ==========================
#  PMI (Pointwise Mutual Information)
# ==========================
def compute_PMI(corpus):
    """Extract significant collocations using PMI."""
    print("Computing PMI collocations...")

    words = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        words.extend(tokens)

    bigrams = list(nltk.bigrams(words))
    word_freq = Counter(words)
    bigram_freq = Counter(bigrams)

    N = sum(bigram_freq.values())
    PMI_scores = {}

    for (w1, w2), freq in bigram_freq.items():
        if freq < 5:
            continue
        p_w1 = word_freq[w1] / N
        p_w2 = word_freq[w2] / N
        p_w1_w2 = freq / N
        PMI_scores[f"{w1} {w2}"] = math.log2(p_w1_w2 / (p_w1 * p_w2))

    # Keep top collocations
    return sorted(PMI_scores, key=PMI_scores.get, reverse=True)[:200]


# ==========================
#  Embedding Clustering
# ==========================
def cluster_embeddings(terms):
    """Group similar terms using Sentence-BERT."""
    print("Clustering terms using sentence embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(terms, convert_to_tensor=True)
    clusters = util.community_detection(embeddings, min_community_size=1, threshold=0.78)

    canonical_terms = []
    for cluster in clusters:
        group = [terms[i] for i in cluster]
        # pick shortest = best canonical
        canonical_terms.append(sorted(group, key=len)[0])

    return canonical_terms


# ==========================
#  TF-IDF + RAKE + PMI Pipeline
# ==========================
def run_ai_discovery(corpus):
    print("Running TF-IDF / RAKE / PMI...")

    # TF-IDF
    tfidf = TfidfVectorizer(
        stop_words=list(ACADEMIC_STOPWORDS.union(nltk.corpus.stopwords.words('english'))),
        ngram_range=(1, 2),
        max_features=3000
    )
    try:
        X = tfidf.fit_transform(corpus)
        scores = X.sum(axis=0)
        tfidf_words = [
            word for word, idx in tfidf.vocabulary_.items()
        ]
    except:
        tfidf_words = []

    # RAKE
    r = Rake()
    r.extract_keywords_from_text(" ".join(corpus)[:500000])
    rake_terms = r.get_ranked_phrases()

    # PMI
    pmi_terms = compute_PMI(corpus)

    # Combine
    merged = tfidf_words[:300] + rake_terms[:300] + pmi_terms[:200]

    # Clean + filter
    cleaned = []
    for term in merged:
        c = clean_term(term)
        if c:
            cleaned.append(c)

    # Remove duplicates
    cleaned = sorted(list(set(cleaned)))

    # Embedding clustering (group synonyms)
    clustered = cluster_embeddings(cleaned)

    return clustered[:TOP_N_AI_KEYWORDS]


# ==========================
#  Write final domain_config.py
# ==========================
def write_config_file(final_keywords):
    print(f"Writing updated config to {OUTPUT_CONFIG_FILE}...")

    content = f"""# ===============================================
#  AUTO-GENERATED DOMAIN CONFIGURATION
#  DO NOT EDIT MANUALLY — Updated by 0_auto_update_config_PRO.py
# ===============================================

DOMAIN_KEYWORDS = {json.dumps(final_keywords, indent=4)}

CRITICAL_DESCRIPTORS = [
    "facies", "shear", "surface", "scarp", "geometry", "reflection",
    "amplitude", "deposit", "channel", "basin", "fan", "flow", 
    "failure", "boundary", "layer"
]

PROTECTED_TERMS = {{
    "mass transport deposit", "mass transport complex",
    "basal shear surface", "chaotic facies", "transparent facies",
    "seismic reflection", "headwall scarp", "continental slope",
    "amazon fan", "santos basin", "foz do amazonas"
}}

RELATION_MAP = {{
    "TRIGGERED_BY": ["trigger", "cause", "initiate", "generate", "induce", "lead"],
    "COMPOSED_OF": ["contain", "comprise", "consist", "include", "show", "exhibit"],
    "LOCATED_IN": ["locate", "occur", "found", "situated", "within", "overlie"],
    "HAS_AGE": ["date", "age", "older", "younger"],
    "ASSOCIATED_WITH": ["associate", "relate", "link", "correlate", "imply"]
}}
"""
    with open(OUTPUT_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("✔ Success — domain_config.py updated.")


# ==========================
#  Main
# ==========================
def main():
    setup_nltk()
    corpus = load_corpus()
    if not corpus:
        print("No text found.")
        return

    ai_terms = run_ai_discovery(corpus)

    combined = sorted(list(set(EXPERT_KEYWORDS + ai_terms)))
    print(f"Final DOMAIN_KEYWORDS size: {len(combined)}")

    write_config_file(combined)


if __name__ == "__main__":
    main()
