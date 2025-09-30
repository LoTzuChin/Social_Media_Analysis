import os
import re
import pandas as pd
from typing import List
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import subprocess, sys



try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from gensim.utils import simple_preprocess
    from gensim.models.phrases import Phrases, Phraser
    _GENSIM_AVAILABLE = True
except Exception as _e:
    _GENSIM_AVAILABLE = False
    _GENSIM_ERROR = _e

    def simple_preprocess(text: str, deacc: bool = False, min_len: int = 2, max_len: int = 30):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = text.lower()
        toks = re.findall(r"[\w]+", text)
        return [t for t in toks if min_len <= len(t) <= max_len]

    class Phrases:
        def __init__(self, *args, **kwargs):
            pass

    class Phraser:
        def __init__(self, *args, **kwargs):
            pass
        def __getitem__(self, tokens):
            return tokens

nltk.download("stopwords", quiet=True)
EN_STOPWORDS = set(stopwords.words("english"))

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

ALLOWED_POS = {"NOUN", "ADJ", "ADV"}

def log_step(message: str) -> None:
    print("[INFO] " + str(message), flush=True)

def basic_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    tokens = simple_preprocess(text, deacc=False, min_len=3, max_len=30)
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in EN_STOPWORDS]

def build_phrases(list_of_tokens: List[List[str]]) -> Phraser:
    log_step("Training bigram phrase model")
    bigram = Phrases(list_of_tokens, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    log_step("Training trigram phrase model")
    trigram = Phrases(bigram_phraser[list_of_tokens], min_count=3, threshold=10)
    trigram_phraser = Phraser(trigram)
    log_step("Phrase models ready")
    return trigram_phraser

def apply_phrases(tokens: List[str], phraser: Phraser) -> List[str]:
    return list(phraser[tokens])

def lemmatize_and_pos_filter(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if "_" in tok:
            parts = tok.split("_")
            kept_parts = []
            for p in parts:
                doc = nlp(p)
                for t in doc:
                    if t.pos_ in ALLOWED_POS:
                        kept_parts.append(t.lemma_.lower())
            if kept_parts:
                out.append("_".join(kept_parts))
        else:
            doc = nlp(tok)
            for t in doc:
                if t.pos_ in ALLOWED_POS:
                    out.append(t.lemma_.lower())
    return out

def frequency_filter_docs(docs_tokens: List[List[str]], min_df=0.015, max_df=0.80) -> List[List[str]]:
    docs_text = [" ".join(toks) for toks in docs_tokens]
    if len(docs_text) == 0:
        log_step("No documents supplied to frequency filter; skipping.")
        return docs_tokens

    log_step("Applying document frequency filter")
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\w[\w_]+",
        min_df=min_df,
        max_df=max_df
    )
    vectorizer.fit(docs_text)
    vocab = set(vectorizer.get_feature_names_out())

    filtered = [[t for t in toks if t in vocab] for toks in docs_tokens]
    return filtered

def load_and_clean(csv_path: str, text_col: str = "Description") -> List[List[str]]:
    log_step(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")
    texts = df[text_col].fillna("").astype(str).tolist()
    log_step(f"Loaded {len(texts)} documents from column '{text_col}'")

    log_step("Tokenizing documents")
    tokens_list = [basic_tokenize(t) for t in tqdm(texts, desc="Tokenizing", unit="doc")]

    log_step("Removing stopwords")
    tokens_list = [remove_stopwords(toks) for toks in tqdm(tokens_list, desc="Stopword removal", unit="doc")]

    log_step("Building phrase models (bigrams/trigrams)")
    phraser = build_phrases(tokens_list)

    log_step("Applying learned phrases")
    tokens_list = [apply_phrases(toks, phraser) for toks in tqdm(tokens_list, desc="Applying phrases", unit="doc")]

    log_step("Lemmatizing and filtering by POS")
    tokens_list = [lemmatize_and_pos_filter(toks) for toks in tqdm(tokens_list, desc="Lemmatizing", unit="doc")]

    log_step("Filtering tokens by document frequency")
    tokens_list = frequency_filter_docs(tokens_list, min_df=0.015, max_df=0.80)

    log_step("Preprocessing complete")
    return tokens_list

if __name__ == "__main__":
    CSV_PATH = "image_descriptions_all.csv"
    log_step(f"Input CSV path: {CSV_PATH}")

    log_step("Preprocessing documents")
    cleaned_tokens = load_and_clean(CSV_PATH, text_col="Description")

    log_step("Writing cleaned documents to cleaned_texts.csv")
    cleaned_texts = [" ".join(toks) for toks in cleaned_tokens]
    out_df = pd.DataFrame({"cleaned_text": cleaned_texts})
    try:
        _src_df = pd.read_csv(CSV_PATH)
        _cands = [
            "Image Name", "Image", "ImageName", "image_name", "Filename", "filename", "File", "Name"
        ]
        _name_col = next((c for c in _cands if c in _src_df.columns), None)
        if _name_col is not None and len(_src_df) == len(out_df):
            out_df.insert(0, "image_name", _src_df[_name_col].astype(str).tolist())
    except Exception:
        pass
    out_df.to_csv("cleaned_texts.csv", index=False)
    log_step("Cleaned texts saved to cleaned_texts.csv")
