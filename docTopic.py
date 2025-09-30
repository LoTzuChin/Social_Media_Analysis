# 應該是環境問題，無法在地端運行，所以改用 Colab

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from typing import Optional, List, Tuple, Union

def load_model(model_path: str) -> BERTopic:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[INFO] Loading BERTopic model from: {model_path}")
    return BERTopic.load(model_path)

def load_texts(csv_path: str, text_col: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in CSV. Existing: {df.columns.tolist()}")
    df[text_col] = df[text_col].fillna("").astype(str)
    return df

def get_topic_ids(topic_model: BERTopic) -> List[int]:
    info = topic_model.get_topic_info()
    # 排除 -1 (outlier)，保持原順序
    return [int(t) for t in info["Topic"].tolist() if int(t) != -1]

def coerce_to_matrix(probabilities: Union[np.ndarray, List]) -> Optional[np.ndarray]:
    """
    將 BERTopic 回傳的 probabilities 轉成 2D 矩陣。
    若偵測為一維「membership strength」（每篇一個數），回傳 None 讓呼叫端走近似分佈。
    """
    if probabilities is None:
        return None
    if isinstance(probabilities, list):
        try:
            # 可能是 list of 1D arrays -> 嘗試堆疊
            arrs = [np.asarray(p).ravel() for p in probabilities]
            if len(arrs) == 0:
                return None
            if arrs[0].size == 1:
                # 每篇只有一個數（membership strength），不是主題分佈
                return None
            return np.vstack(arrs)
        except Exception:
            return None
    if isinstance(probabilities, np.ndarray):
        if probabilities.ndim == 2:
            return probabilities
        if probabilities.ndim == 1:
            # 一維：若是浮點（每篇一個概率），視為 membership strength
            if np.issubdtype(probabilities.dtype, np.floating) or np.issubdtype(probabilities.dtype, np.integer):
                return None
            # 物件陣列：嘗試堆疊
            try:
                arrs = [np.asarray(p).ravel() for p in probabilities]
                if len(arrs) and arrs[0].size > 1:
                    return np.vstack(arrs)
            except Exception:
                return None
    return None

def get_topic_distribution(topic_model: BERTopic, docs: List[str]) -> Tuple[np.ndarray, List[int], str]:
    """
    回傳 (topic_matrix[N,K], topic_ids[K], mode)
    mode: 'transform' 或 'approx'
    """
    topics, probs = topic_model.transform(docs)
    mat = coerce_to_matrix(probs)
    topic_ids = get_topic_ids(topic_model)

    if mat is not None:
        # transform 已給出 N×K
        if mat.shape[1] != len(topic_ids):
            # 欄數對不上 -> 退回連號
            print(f"[WARN] transform matrix topics({mat.shape[1]}) != topic_ids({len(topic_ids)}).")
        return mat, topic_ids, "transform"

    # 走近似：每篇文件主題分佈
    print("[INFO] Falling back to approximate_distribution(...) to get topic proportions.")
    # 依官方建議，可調 window/stride；預設這組在段落級分割上效果穩健
    res = topic_model.approximate_distribution(docs, window=8, stride=4)
    # 兼容不同版本回傳型態
    topic_distr = res[0] if isinstance(res, tuple) else res
    if not isinstance(topic_distr, np.ndarray) or topic_distr.ndim != 2:
        raise RuntimeError("approximate_distribution did not return a 2D matrix; cannot build topic proportions.")
    if topic_distr.shape[1] != len(topic_ids):
        print(f"[WARN] approx matrix topics({topic_distr.shape[1]}) != topic_ids({len(topic_ids)}). Using sequential IDs.")
        topic_ids = list(range(topic_distr.shape[1]))
    return topic_distr, topic_ids, "approx"

def build_prob_df(prob_matrix: np.ndarray, topic_ids: List[int]) -> pd.DataFrame:
    cols = [f"topic_{tid}_prob" for tid in topic_ids]
    return pd.DataFrame(prob_matrix, columns=cols)

def add_dominant(prob_df: pd.DataFrame) -> pd.DataFrame:
    probs = prob_df.values
    argmax_idx = np.argmax(probs, axis=1)
    max_probs = probs[np.arange(probs.shape[0]), argmax_idx]
    colnames = np.array(prob_df.columns)
    dom_topic_cols = colnames[argmax_idx]
    dom_topic_ids = [c.replace("topic_", "").replace("_prob", "") for c in dom_topic_cols]
    return pd.DataFrame({"dominant_topic": dom_topic_ids, "dominant_prob": max_probs})

def main(model_path: str, input_csv: str, text_col: str, output_csv: str):
    topic_model = load_model(model_path)
    df = load_texts(input_csv, text_col)

    docs = df[text_col].tolist()
    print(f"[INFO] Transforming {len(docs)} documents ...")
    topic_matrix, topic_ids, mode = get_topic_distribution(topic_model, docs)
    print(f"[INFO] Using topic proportions from: {mode}")

    prob_df = build_prob_df(topic_matrix, topic_ids)
    dom_df = add_dominant(prob_df)

    out_df = pd.concat([df.reset_index(drop=True),
                        prob_df.reset_index(drop=True),
                        dom_df.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved topic proportions to: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # ==== 依你的 Colab 路徑寫死 ====
    MODEL_PATH  = "/content/bertopic_finetuned_model"
    INPUT_CSV   = "/content/cleaned_texts.csv"
    TEXT_COLUMN = "cleaned_text"
    OUTPUT_CSV  = "topic_proportions.csv"

    main(MODEL_PATH, INPUT_CSV, TEXT_COLUMN, OUTPUT_CSV)
