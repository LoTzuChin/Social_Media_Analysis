#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 批次呼叫 Google Gemini 產生圖片描述

import json
import mimetypes
import os
from pathlib import Path
from time import sleep
from typing import List, Dict
import csv
import google.generativeai as genai


# 從 secrets.json 載入 Google API 金鑰
def load_api_key_from_json(json_path: Path) -> str:
    """讀取指定路徑的 secrets.json 並回傳 GOOGLE_API_KEY。"""
    if not json_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = data.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("\"GOOGLE_API_KEY\" not found in secret.json")
    return key


# 依檔名推斷圖片的 MIME 類型
def detect_mime_type(file_path: Path) -> str:
    """推測圖片檔案的 MIME 類型，若未知則回傳通用的二進位格式。"""
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


# 列出資料夾內所有支援的圖片檔案
def list_image_files(folder: Path) -> List[Path]:
    """依檔名字母順序回傳資料夾中的圖片路徑清單。"""
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files


# 呼叫 Gemini 逐一產生圖片描述
def describe_images(model_name: str, images: List[Path], prompt: str) -> List[Dict[str, str]]:
    """使用指定模型為每張圖片產生描述文字。"""
    model = genai.GenerativeModel(model_name)
    results = []
    for idx, img_path in enumerate(images, start=1):
        mime_type = detect_mime_type(img_path)
        with img_path.open("rb") as f:
            image_part = {"mime_type": mime_type, "data": f.read()}

        try:
            # 將提示詞與圖片內容一併送入模型產生文字描述
            resp = model.generate_content([prompt, image_part])
            text = (resp.text or "").strip()
            results.append({"Image Name": img_path.name, "Description": text})
            print(f"[ok] ({idx}/{len(images)}) {img_path.name}")
        except Exception as e:
            # 失敗時僅記錄錯誤並保留空白描述
            print(f"[fail] {img_path.name}: {e}")
            results.append({"Image Name": img_path.name, "Description": ""})

    return results


# 將描述結果輸出為 CSV 檔案
def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    """把產生的描述列表寫入 CSV 方便後續分析。"""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image Name", "Description"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # === 參數設定 ===
    secrets_path = Path("secrets.json")
    input_folder = Path("D:\Social_Media_Analysis\images_remaining")
    output_csv = Path("image_descriptions_second.csv")
    model_name = "models/gemini-2.0-flash"
    prompt = (
        "Describe the image in detail, focusing on the main objects and their "
        "context. Provide a fully descriptive response without using bullet points "
        "and in English."
    )

    # 1) 初始化 API 金鑰
    api_key = load_api_key_from_json(secrets_path)
    genai.configure(api_key=api_key)

    # 2) 掃描圖片資料夾
    image_paths = list_image_files(input_folder)
    print(f"[info] found {len(image_paths)} image(s) in {input_folder}")

    # 3) 使用模型產生描述
    rows = describe_images(model_name, image_paths, prompt)

    # 4) 寫入 CSV
    write_csv(rows, output_csv)
    print(f"[done] saved {len(rows)} descriptions to {output_csv}")
