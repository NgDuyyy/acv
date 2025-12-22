"""
Setup 20k Data Folder
---------------------
This script restructures the `data` directory for the 20k image experiment.

It performs the following:
1. Creates `data/train` and `data/val` directories.
2. Copies the first 20,000 images from `data/coco_2017/train2017` to `data/train`.
3. Copies all images from `data/coco_2017/val2017` to `data/val`.
4. Filters the original COCO annotation files (`captions_train2017.json` and `captions_val2017.json`)
   to include only the selected images.
5. Saves the filtered annotations as `data/train/train_data.json` and `data/val/val_data.json`.

These output files are formatted exactly as required by `scripts/prepare_data_merged.py`.

Usage:
    python setup_20k_data_folder.py
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm # Optional, for progress

# --- CONFIGURATION ---
ROOT_DATA = Path(r"E:\hoc_voi_cha_hanh\kltn\data")
SOURCE_TRAIN_IMG = ROOT_DATA / "coco_2017" / "train2017"
SOURCE_VAL_IMG   = ROOT_DATA / "coco_2017" / "val2017"
SOURCE_TRAIN_ANN = ROOT_DATA / "coco_2017" / "annotations" / "captions_train2017.json"
SOURCE_VAL_ANN   = ROOT_DATA / "coco_2017" / "annotations" / "captions_val2017.json"

OUTPUT_DIR = Path(r"E:\hoc_voi_cha_hanh\kltn\decoder_lstm\data")

# Output Roots
OUTPUT_TRAIN_ROOT = OUTPUT_DIR / "train"
OUTPUT_VAL_ROOT   = OUTPUT_DIR / "val"

# Image Directories (New Requirement)
TARGET_TRAIN_IMG_DIR = OUTPUT_TRAIN_ROOT / "train_images"
TARGET_VAL_IMG_DIR   = OUTPUT_VAL_ROOT / "val_images"

# JSON Paths (Stay in Root)
TARGET_TRAIN_JSON = OUTPUT_TRAIN_ROOT / "train_data.json"
TARGET_VAL_JSON   = OUTPUT_VAL_ROOT / "val_data.json"

TRAIN_LIMIT = 20000

def filter_coco_json(source_json_path, target_img_ids, output_path):
    print(f"Loading {source_json_path}...")
    with open(source_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter 'images' list
    print(f"Filtering images (keeping {len(target_img_ids)} IDs)...")
    filtered_images = [img for img in data['images'] if img['id'] in target_img_ids]
    
    # Filter 'annotations' list
    print("Filtering annotations...")
    target_ids_set = set(target_img_ids)
    filtered_anns = [ann for ann in data['annotations'] if ann['image_id'] in target_ids_set]
    
    out_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_anns
    }
    
    print(f"Saving filtered JSON to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False)
    print("Done.")

def setup_folder(source_dir, target_dir, limit=None):
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return []
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get List of images
    all_files = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    if limit:
        selected_files = all_files[:limit]
    else:
        selected_files = all_files
        
    print(f"Copying {len(selected_files)} images from {source_dir} to {target_dir}...")
    
    copied_ids = []
    
    for p in tqdm(selected_files):
        dest = target_dir / p.name
        # Copy file if not exists
        if not dest.exists():
            shutil.copy2(p, dest)
        
        # Extract ID from filename (standard COCO format: 000000123456.jpg -> 123456)
        try:
            img_id = int(str(p.stem).lstrip("0") or "0")
            copied_ids.append(img_id)
        except ValueError:
            print(f"[WARN] Could not parse ID from {p.name}")
            
    return copied_ids

def main():
    print("=== SETUP 20K EXPERIMENT DATA FOLDER ===")
    
    # 1. Setup TRAIN
    print("\n--- Processing TRAIN Data ---")
    train_ids = setup_folder(SOURCE_TRAIN_IMG, TARGET_TRAIN_IMG_DIR, limit=TRAIN_LIMIT)
    if train_ids:
        filter_coco_json(SOURCE_TRAIN_ANN, train_ids, TARGET_TRAIN_JSON)
        
    # 2. Setup VAL
    print("\n--- Processing VAL Data ---")
    # Take all val images (usually 5k)
    val_ids = setup_folder(SOURCE_VAL_IMG, TARGET_VAL_IMG_DIR, limit=None) 
    if val_ids:
        filter_coco_json(SOURCE_VAL_ANN, val_ids, TARGET_VAL_JSON)
        
    print("\n=== COMPLETED ===")
    print(f"Data prepared in:")
    print(f"  - {TARGET_TRAIN_IMG_DIR}")
    print(f"  - {TARGET_VAL_IMG_DIR}")
    print("\nNext Step: Run 'python scripts/prepare_data_merged.py' to create the final LSTM input.")

if __name__ == "__main__":
    main()
