#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import mimetypes
import os
from pathlib import Path
from time import sleep
from typing import List, Dict
import csv
import google.generativeai as genai


def load_api_key_from_json(json_path: Path) -> str:
    if not json_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = data.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError('"GOOGLE_API_KEY" not found in secret.json')
    return key


def detect_mime_type(file_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files


def describe_images(model_name: str, images: List[Path], prompt: str) -> List[Dict[str, str]]:
    model = genai.GenerativeModel(model_name)
    results = []
    for idx, img_path in enumerate(images, start=1):
        mime_type = detect_mime_type(img_path)
        with img_path.open("rb") as f:
            image_part = {"mime_type": mime_type, "data": f.read()}

        try:
            resp = model.generate_content([prompt, image_part])
            text = (resp.text or "").strip()
            results.append({"Image Name": img_path.name, "Description": text})
            print(f"[ok] ({idx}/{len(images)}) {img_path.name}")
        except Exception as e:
            print(f"[fail] {img_path.name}: {e}")
            results.append({"Image Name": img_path.name, "Description": ""})

    return results


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image Name", "Description"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # === 固定參數 ===
    secrets_path = Path("secrets.json")
    input_folder = Path("D:\Social_Media_Analysis\images_remaining")
    output_csv = Path("image_descriptions_second.csv")
    model_name = "models/gemini-2.0-flash"
    prompt = (
        "Describe the image in detail, focusing on the main objects and their "
        "context. Provide a fully descriptive response without using bullet points "
        "and in English."
    )

    # 1) API key
    api_key = load_api_key_from_json(secrets_path)
    genai.configure(api_key=api_key)

    # 2) 找圖片
    image_paths = list_image_files(input_folder)
    print(f"[info] found {len(image_paths)} image(s) in {input_folder}")

    # 3) 描述
    rows = describe_images(model_name, image_paths, prompt)

    # 4) 寫入 CSV
    write_csv(rows, output_csv)
    print(f"[done] saved {len(rows)} descriptions to {output_csv}")
