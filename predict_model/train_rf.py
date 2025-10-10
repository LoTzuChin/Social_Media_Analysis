# train_rf_tfidf_chi2.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# ====== 輸入檔 ======
TEXTS_CSV   = "prepocessing_data\cleaned_texts.csv"     # 含 image_name, cleaned_text
LABELS_CSV  = "bertopic_labels_except_no_topic.csv"   # 含 image_name, new_topic_num
SPLIT_CSV   = "split.csv"             # 含 image_name, split(train/test)

# ====== 輸出資料夾 ======
RF_DIR = Path("predict_model\\rf_report_dir")
RF_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV       = RF_DIR / "rf_predictions.csv"
REPORT_TXT     = RF_DIR / "rf_report.txt"
CM_CSV         = RF_DIR / "rf_confusion_matrix.csv"
MODEL_JOBLIB   = RF_DIR / "rf_model.joblib"
VECT_WORD      = RF_DIR / "tfidf_word.joblib"
VECT_CHAR      = RF_DIR / "tfidf_char.joblib"
SEL_JOBLIB     = RF_DIR / "chi2_selector.joblib"
LABEL_MAP_JSON = RF_DIR / "label_mapping.json"
PARAMS_JSON    = RF_DIR / "rf_params.json"
META_JSON      = RF_DIR / "meta.json"

# ====== 讀文字欄位 ======
def load_texts(path: str) -> pd.DataFrame:
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp950"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            pass
    if df is None:
        raise RuntimeError(f"讀取 {path} 失敗（編碼問題？）")

    for cand in ["cleaned_text", "text", "desc", "description"]:
        if cand in df.columns:
            text_col = cand
            break
    else:
        raise ValueError(f"{path} 必須包含文字欄位 cleaned_text/text/desc/description")

    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            from pathlib import Path as _P
            df["image_name"] = df["image_path"].apply(lambda p: _P(str(p)).name)
        else:
            raise ValueError(f"{path} 必須有 image_name 欄位或 image_path 欄位")

    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def main():
    print("[INFO] 讀取與合併...")
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)[["image_name", "new_topic_num"]]
    split  = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]

    df = (
        texts.merge(labels, on="image_name", how="inner")
             .merge(split, on="image_name", how="inner")
    ).dropna(subset=["text", "new_topic_num", "split"])

    if df.empty:
        raise ValueError("合併後資料為空，請檢查 image_name 是否對齊")

    df["text"] = df["text"].astype(str)
    df["label_str"] = df["new_topic_num"].astype(str)

    print(f"[INFO] 合併後樣本數: {len(df)}；類別數: {df['label_str'].nunique()}")
    print(df["split"].value_counts())

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    if df_train.empty or df_test.empty:
        raise ValueError("train 或 test 為空，請檢查 split.csv")

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values

    # ====== TF-IDF（word + char 3–5, float32）======
    print("[INFO] 建立 TF-IDF...")
    word_vect = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.9, strip_accents="unicode",
        dtype=np.float32
    )
    char_vect = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        min_df=2, max_df=0.95,
        dtype=np.float32
    )

    Xw_tr = word_vect.fit_transform(X_train_text)
    Xc_tr = char_vect.fit_transform(X_train_text)
    X_train_sparse = hstack([Xw_tr, Xc_tr]).tocsr()

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test_sparse = hstack([Xw_te, Xc_te]).tocsr()

    n_samples, n_features = X_train_sparse.shape
    print(f"[INFO] TF-IDF 維度: {n_features}  訓練樣本: {n_samples}")

    # ====== Feature Selection: Chi2 → K 維（快速、穩定）======
    K = min(2000, n_features)  # 你可以調 1000/2000/4000
    print(f"[INFO] Chi2 特徵選擇: k={K}")
    selector = SelectKBest(score_func=chi2, k=K)
    # chi2 需要非負，因此用 TF-IDF OK；保持稀疏 -> transform 結果仍是稀疏
    X_train_sel = selector.fit_transform(X_train_sparse, df_train["label_str"].values)
    X_test_sel  = selector.transform(X_test_sparse)

    # 轉 dense（K=2000 時，~ 5k×2k ≈ 40MB）
    print("[INFO] 轉 dense 以餵 RandomForest ...")
    X_train = X_train_sel.toarray()
    X_test  = X_test_sel.toarray()

    # ====== 標籤整數化 ======
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["label_str"].values)
    y_test  = le.transform(df_test["label_str"].values)

    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "classes": le.classes_.tolist(),
            "mapping": {int(i): cls for i, cls in enumerate(le.classes_)}
        }, f, ensure_ascii=False, indent=2)

    # ====== RF 參數 ======
    params = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced_subsample",
        "verbose": 0
    }
    with open(PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("[INFO] 訓練 RandomForest ...")
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    print("[INFO] 預測與評估 ...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        labels=le.classes_
    )
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(CM_CSV, index=True)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("[Overall]\n")
        f.write(f"accuracy  : {acc:.4f}\n")
        f.write(f"precision(macro): {p_macro:.4f}\n")
        f.write(f"recall(macro)   : {r_macro:.4f}\n")
        f.write(f"f1(macro)       : {f1_macro:.4f}\n\n")
        f.write(f"precision(weighted): {p_weight:.4f}\n")
        f.write(f"recall(weighted)   : {r_weight:.4f}\n")
        f.write(f"f1(weighted)       : {f1_weight:.4f}\n\n")
        f.write("[Per-class report]\n")
        f.write(report)
        f.write("\n[Confusion Matrix saved to] " + str(CM_CSV) + "\n")

    out_pred = df_test[["image_name"]].copy()
    out_pred["true"] = le.inverse_transform(y_test)
    out_pred["pred"] = le.inverse_transform(y_pred)
    out_pred.to_csv(PRED_CSV, index=False)

    joblib.dump(clf, MODEL_JOBLIB)
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)
    joblib.dump(selector, SEL_JOBLIB)

    def safe_params(params: dict) -> dict:
        out = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)   # 把 type、function 等轉成字串
        return out

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "vectorizer_word_params": safe_params(word_vect.get_params()),
            "vectorizer_char_params": safe_params(char_vect.get_params()),
            "chi2_k": int(K),
            "train_samples": int(X_train.shape[0]),
            "train_dim": int(X_train.shape[1])
        }, f, ensure_ascii=False, indent=2)


    print(f"[OK] RF(chi2) 訓練完成 @ {RF_DIR.resolve()}")
    print(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weight:.4f}")
    print(f"- 報告: {REPORT_TXT}")
    print(f"- 混淆矩陣: {CM_CSV}")
    print(f"- 預測: {PRED_CSV}")
    print(f"- 模型: {MODEL_JOBLIB}")
    print(f"- 向量器: {VECT_WORD}, {VECT_CHAR}")
    print(f"- 特徵選擇: {SEL_JOBLIB}")
    print(f"- 標籤對應: {LABEL_MAP_JSON}")

if __name__ == "__main__":
    main()
