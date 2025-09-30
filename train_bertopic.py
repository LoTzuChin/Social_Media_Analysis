# 應該是環境問題，無法在地端運行，所以改用 Colab

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import re

def _auto_language_guess(sample: str) -> str:
    if not sample:
        return "multilingual"
    ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / max(1, len(sample))
    if ascii_ratio > 0.95 and re.search(r"[A-Za-z]", sample):
        return "english"
    return "multilingual"

def train(df):
    # --- 這三行是關鍵，避免空值/雜訊導致本機版卡住 ---
    descriptions = df['cleaned_text'].astype(str).str.strip()
    descriptions = descriptions[descriptions.str.len() >= 5].drop_duplicates()
    descriptions = descriptions.tolist()

    # 小樣本判斷語言（避免中文卻硬塞 english）
    sample_text = " ".join(descriptions[:50])
    lang = _auto_language_guess(sample_text)
    print(f"[INFO] Detected language: {lang} | docs: {len(descriptions)}")

    print("[INFO] Loading embedding model...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("[INFO] Fitting BERTopic...")
    topic_model = BERTopic(embedding_model=embedding_model, language=lang, calculate_probabilities=False, verbose=True)
    topics, probs = topic_model.fit_transform(descriptions)
    return topic_model, np.array(topics)

def writeCsv(topic_model, topics, filename="topic.csv"):
    topic_keywords = []
    for topic_num in topic_model.get_topics().keys():
        if topic_num == -1:
            continue
        keywords_with_weights = topic_model.get_topic(topic_num) or []
        keywords_str = "; ".join([f"{word}:{weight:.4f}" for word, weight in keywords_with_weights[:10]])
        doc_count = int((topics == topic_num).sum())
        topic_keywords.append({
            "Topic": topic_num,
            "Document_Count": doc_count,
            "Keywords_Weights": keywords_str
        })
    df_topic = pd.DataFrame(topic_keywords).sort_values("Document_Count", ascending=False)
    df_topic.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {filename}")

if __name__ == "__main__":
    print("[INFO] Reading CSV...")
    # 如有 Big5/CP950，可以換成 encoding="utf-8-sig", errors="ignore"
    df = pd.read_csv('/content/cleaned_texts.csv')
    topic_model, topics = train(df)
    writeCsv(topic_model, topics, "topic.csv")
    topic_model.save("bertopic_finetuned_model")
    print("[DONE] topic.csv 已儲存，模型已儲存於 bertopic_finetuned_model/")
