import os
import json
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# --- Configuration ---
INPUT_FOLDER = "./output_json"

TOP_N_KEYWORDS = 60

# --- 1. AUTHORITATIVE GEOLOGICAL ROOTS ---
GEO_ROOTS = [
    "seism", "fault", "fract", "fail", "defom", "shear", "stress", "strain", 
    "veloc", "pressur", "stabil", "glid", "kinemat", "dynam", "mechan",
    "lith", "strat", "sediment", "deposit", "turbid", "facies", "clast", 
    "sand", "clay", "mud", "conglomerate", "accumulat", "eros",
    "slump", "slide", "mass", "transport", "debris", "avalanche", "flow", 
    "block", "lobe", "glide", "creep", "colluv",
    "slope", "basin", "margin", "chan", "levee", "fan", "canyon", 
    "toe", "headwall", "scarp", "basal", "surface", "ramp", "front", 
    "boundar", "thick", "geometr", "morph", "feature", "struc",
    "reflec", "amplitud", "chaotic", "transpar", "continu", "hummocky", 
    "acoust", "horizon", "contour", "submar", "subaq", "offshore"
]

# --- 2. ACADEMIC NOISE FILTER ---
ACADEMIC_STOPWORDS = {
    "figure", "fig", "table", "doi", "issn", "isbn", "http", "www", "org", 
    "et", "al", "introduction", "conclusion", "results", "discussion", "methodology",
    "references", "acknowledgements", "study", "paper", "data", "analysis",
    "using", "used", "based", "observed", "shown", "suggests", "within",
    "model", "value", "equation", "parameter", "method", "case", "area", "field",
    "different", "associated", "example", "following"
}

def setup_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data (Lemmatizer)...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True) # For lemmatization
        nltk.download('omw-1.4', quiet=True)

# Initialize Lemmatizer globally
lemmatizer = WordNetLemmatizer()

def clean_term(term):
    # 1. Lowercase (Fixes BLOCK vs block)
    term = term.lower().strip()
    
    # 2. Lemmatization (Fixes blocks vs block)
    # We split the phrase, lemmatize the last word (usually the noun), and join back
    words = term.split()
    if not words: return None
    
    # Lemmatize the last word (e.g., "seismic faults" -> "seismic fault")
    words[-1] = lemmatizer.lemmatize(words[-1], pos='n') 
    term = " ".join(words)

    # 3. Basic Cleaning
    if len(term) < 3: return None
    if re.search(r'[\d\W_]', term.replace(' ', '').replace('-', '')): 
        if re.search(r'[0-9µ×∈=+/\\<>\[\]\{\}\|\(\)\*\^\%\$]', term): return None

    # 4. Length Check
    if len(words) > 4: return None 
    
    # 5. Stopword Check
    if any(w in ACADEMIC_STOPWORDS for w in words): return None
    
    # 6. THE GEOLOGICAL FILTER (Must contain a root)
    if not any(root in term for root in GEO_ROOTS):
        return None
    
    return term

def load_corpus():
    print(f"Loading text from {INPUT_FOLDER}...")
    corpus = []
    if not os.path.exists(INPUT_FOLDER): return []

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    for filename in tqdm(files):
        try:
            with open(os.path.join(INPUT_FOLDER, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                doc_text = " ".join([
                    item['text'] for item in data 
                    if item.get('text') and item.get('type') == "NarrativeText"
                ])
                corpus.append(doc_text)
        except: pass
    return corpus

def extract_tfidf_keywords(corpus):
    print("Running TF-IDF analysis...")
    stopwords_list = list(ACADEMIC_STOPWORDS.union(nltk.corpus.stopwords.words('english')))
    vectorizer = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1, 2), max_features=3000)
    try:
        X = vectorizer.fit_transform(corpus)
        sum_scores = X.sum(axis=0)
        scores = [(word, sum_scores[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return [w for w, s in sorted(scores, key=lambda x: x[1], reverse=True)]
    except ValueError: return []

def extract_rake_keywords(text_block):
    print("Running RAKE analysis...")
    r = Rake()
    r.extract_keywords_from_text(text_block[:500000]) 
    return r.get_ranked_phrases()

def main():
    setup_nltk()
    corpus = load_corpus()
    if not corpus: return

    tfidf_words = extract_tfidf_keywords(corpus)
    full_text = " ".join(corpus)
    rake_phrases = extract_rake_keywords(full_text)

    candidates = tfidf_words[:300] + rake_phrases[:300]
    final_keywords = set()
    
    print("Applying LEMMATIZATION & GEOLOGICAL FILTER...")
    for term in candidates:
        clean = clean_term(term)
        if clean:
            final_keywords.add(clean)
            
    sorted_keywords = sorted(list(final_keywords))[:TOP_N_KEYWORDS]

    print("\n" + "="*60)
    print("   ULTIMATE CONFIGURATION (Copy to domain_config.py)")
    print("="*60 + "\n")
    print("DOMAIN_KEYWORDS = [")
    for kw in sorted_keywords:
        print(f"    \"{kw}\",")
    print("]")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()