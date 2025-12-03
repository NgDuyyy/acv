"""
Complete Image Captioning Pipeline
----------------------------------
Extract bottom-up features using local models/utils,
build a minimal dataset JSON, update the bottom-up feature file, and run
caption generation using local models.

Usage (PowerShell examples):
1) Extract features for all images in a folder and append to KTVIC feature file:
   python complete_image_captioning_pipeline.py ^
     extract ^
     --images_dir data/test/images ^
     --feature_out result/features/ktvicbu_att.pth ^
     --json_out result/features/custom_dataset.json

2) Run captioning on the just-created dataset:
   python complete_image_captioning_pipeline.py ^
     caption ^
     --dataset_json result/features/custom_dataset.json ^
     --feature_path result/features/ktvicbu_att.pth ^
     --model result/checkpoints/model-best.pth ^
     --infos result/checkpoints/infos_ktvic_test-best.pkl ^
     --predictions_csv result/caption_predictions.csv ^
     --metrics_csv result/evaluate_valid_data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import cv2

# Project roots
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))  # Add project root to path

# Import local modules
try:
    from utils.feature_extracting import config as cfg_frcnn
    from models.feature_extracting.faster_rcnn import resnet as faster_rcnn
    from utils.feature_extracting.blob import im_list_to_blob
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Ensure you are running from the project root and 'models'/'utils' folders are populated.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def get_image_blob(im):
    """Converts an image into a network input."""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg_frcnn.cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg_frcnn.cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg_frcnn.cfg.TEST.MAX_SIZE:
            im_scale = float(cfg_frcnn.cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def load_vg_detector(checkpoint_path: str = "models/faster_rcnn_res101_vg.pth"):
    """Load the Faster R-CNN model with ResNet101 backbone."""
    cfg_frcnn.cfg.USE_GPU_NMS = torch.cuda.is_available()
    cfg_frcnn.cfg.CUDA = torch.cuda.is_available()
    
    # Visual Genome classes (1601)
    num_classes = 1601 
    classes = ['class_' + str(i) for i in range(num_classes)]

    model = faster_rcnn.resnet(classes, 101, pretrained=False, class_agnostic=False)
    model.create_architecture()

    # Checkpoint fallback logic
    if not os.path.exists(checkpoint_path):
        # Try finding in FRCNN folder if local missing
        alt_path = ROOT / "FRCNN" / "models" / "faster_rcnn_res101_vg.pth"
        if alt_path.exists():
            checkpoint_path = str(alt_path)
        else:
            print(f"[WARN] Checkpoint not found at {checkpoint_path}. Feature extraction may fail.")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu' if not torch.cuda.is_available() else None)
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        
    return model

def extract_image_features(model, image_path: str):
    """Run the model on an image and return bottom-up features."""
    im_in = cv2.imread(image_path)
    if im_in is None:
        return None
        
    if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
    
    blobs, im_scales = get_image_blob(im_in)
    im_info_np = np.array([[blobs.shape[1], blobs.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(blobs).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        if torch.cuda.is_available():
            im_data_pt = im_data_pt.cuda()
            im_info_pt = im_info_pt.cuda()

        gt_boxes = torch.zeros((1, 1, 5)).cuda() if torch.cuda.is_available() else torch.zeros((1, 1, 5))
        num_boxes = torch.zeros(1).cuda() if torch.cuda.is_available() else torch.zeros(1)
        
        # Forward pass
        _, _, _, _, _, _, _, _, pooled_feat = model(im_data_pt, im_info_pt, gt_boxes, num_boxes)

    return {"features": pooled_feat.cpu().numpy()}


def extract_features_with_frcnn(
    image_paths: Sequence[Path],
    start_image_id: int,
    feature_out: Path,
    json_out: Path,
    split_name: str = "test",
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    
    model = load_vg_detector()

    image_paths = sorted(image_paths)
    if not image_paths:
        raise RuntimeError("No images provided for extraction")

    feature_dict: Dict[str, np.ndarray] = {}
    image_entries: List[dict] = []

    print(f"Extracting features for {len(image_paths)} images...")
    
    for idx, img_path in enumerate(image_paths):
        image_id = start_image_id + idx
        feat = extract_image_features(model, str(img_path))
        if feat is None:
            print(f"[WARN] Skip image (failed): {img_path}")
            continue

        feature_dict[str(image_id)] = feat["features"].astype(np.float32)
        image_entries.append(
            {
                "id": image_id,
                "file_path": img_path.name,
                "filename": img_path.name,
                "split": split_name,
                "sentids": [],
                "sentences": [],
            }
        )

    if not feature_dict:
        raise RuntimeError("No features extracted; aborting.")

    # Save features as individual .npz files
    feature_out.mkdir(parents=True, exist_ok=True)
    print(f"Saving features to {feature_out}...")
    
    for str_id, feat in feature_dict.items():
        # Save as compressed numpy (similar to make_bu_data.py)
        # We assume these are attention features.
        # If you need FC features, you might need to average them or extract separately.
        out_path = feature_out / f"{str_id}.npz"
        np.savez_compressed(out_path, feat=feat)

    # Write minimal dataset JSON
    json_payload = {"images": image_entries}
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Extracted {len(feature_dict)} images. Features -> {feature_out}, JSON -> {json_out}")
    return feature_dict, image_entries


# -----------------------------------------------------------------------------
# Caption generation
# -----------------------------------------------------------------------------
def run_caption_eval(
    dataset_json: Path,
    feature_path: Path,
    model_path: Path,
    infos_path: Path,
    num_images: int,
    device: str,
    split: str,
    num_workers: int,
    run_id: str,
    predictions_csv: Path,
    metrics_csv: Path,
    beam_size: int,
    batch_size: int,
    language_eval_bleu_only: int | None = None,
    bleu_only: int | None = None,
    language_eval_json: Path | None = None,
    dump_images: int | None = None,
    image_root: Path | None = None,
    dump_json: int | None = None,
) -> None:
    
    if not dataset_json.exists(): raise FileNotFoundError(dataset_json)
    if not feature_path.exists(): raise FileNotFoundError(feature_path)
    if not model_path.exists(): raise FileNotFoundError(model_path)
    if not infos_path.exists(): raise FileNotFoundError(infos_path)

    # Use eval.py in root
    eval_script = ROOT / "eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found at {eval_script}")

    cmd = [
        sys.executable,
        str(eval_script),
        "--model", str(model_path),
        "--infos_path", str(infos_path),
        "--input_json", str(dataset_json),
        "--input_att_dir", str(feature_path),
        "--language_eval", "1",
        "--num_images", str(num_images),
        "--split", split,
        "--device", device,
        "--batch_size", str(batch_size),
        "--beam_size", str(beam_size),
        "--id", run_id,
        "--force", "1",
        "--num_workers", str(num_workers),
        "--language_eval_json", str(language_eval_json or dataset_json),
        "--dump_images", str(dump_images if dump_images is not None else 0),
        "--dump_json", str(dump_json if dump_json is not None else 0),
    ]

    if image_root is not None:
        cmd.extend(["--image_root", str(image_root)])
    if language_eval_bleu_only is not None:
        cmd.extend(["--language_eval_bleu_only", str(language_eval_bleu_only)])
    if bleu_only is not None:
        cmd.extend(["--bleu_only", str(bleu_only)])

    print(f"[INFO] Running command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    # Run eval.py
    # Note: eval.py prints results to stdout/stderr
    process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    print(stdout)
    if stderr:
        print("[STDERR]", stderr)
    
    if process.returncode != 0:
        raise RuntimeError("Caption eval failed")

    # Parse results from eval_results folder (created by eval.py)
    # eval.py creates files in 'eval_results/' relative to CWD.
    # Since we ran in ROOT, check ROOT/eval_results
    eval_results_dir = ROOT / "result/eval_results"
    pred_cache = eval_results_dir / f".saved_pred_{run_id}_{split}.pth"
    result_json = eval_results_dir / f"{run_id}_{split}.json"
    
    if not pred_cache.exists() or not result_json.exists():
        fallback_dir = ROOT / "eval_results"
        fallback_pred = fallback_dir / pred_cache.name
        fallback_json = fallback_dir / result_json.name
        if fallback_pred.exists() and fallback_json.exists():
            eval_results_dir = fallback_dir
            pred_cache = fallback_pred
            result_json = fallback_json
        else:
            print("[WARN] Eval results not found. Eval might have failed or paths differ.")
            return

    # Load dataset / reference JSON in COCO format
    # prepro_reference_json.py outputs a COCO-style file:
    #   {"images": [{"id": ... , "file_name"/"filename"?: ...}, ...],
    #    "annotations": [{"image_id": ..., "caption": ...}, ...]}
    predictions, _ = torch.load(pred_cache, map_location="cpu")
    ds = json.loads(dataset_json.read_text(encoding="utf-8"))

    # Map image_id -> image info (to get filename)
    id_to_info = {img["id"]: img for img in ds.get("images", [])}

    # Map image_id -> list of GT captions from annotations (COCO style)
    id_to_captions: dict[int, List[str]] = {}
    for ann in ds.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        id_to_captions.setdefault(img_id, []).append(ann.get("caption", ""))

    # Optional supplemental mapping from a *_data.json file (Karpathy-style) to recover filenames
    supplemental_mapping: dict[int, str] = {}
    if not any("file_path" in img or "filename" in img or "file_name" in img for img in ds.get("images", [])):
        if dataset_json.name.endswith("_reference.json"):
            candidate = dataset_json.with_name(dataset_json.name.replace("_reference", "_data"))
        else:
            candidate = dataset_json.with_name(dataset_json.stem + "_data.json")
        if candidate.exists():
            extra = json.loads(candidate.read_text(encoding="utf-8")).get("images", [])
            for img in extra:
                img_id = img.get("id")
                if img_id is None:
                    continue
                supplemental_mapping[img_id] = img.get("file_path") or img.get("filename", "")

    image_root_str = str(image_root) if image_root else ""

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not predictions_csv.exists()
    with predictions_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "caption_gt", "caption_pred"])
        for entry in predictions:
            img_id = entry["image_id"]
            info = id_to_info.get(img_id, {})

            # Filename priority: COCO's file_name -> our filename -> supplemental mapping -> image_id
            filename = (
                info.get("file_name")
                or info.get("filename")
                or info.get("file_path")
                or supplemental_mapping.get(img_id, "")
            )

            if filename and image_root_str:
                filename_out = str(Path(image_root_str) / filename)
            else:
                filename_out = filename or str(img_id)

            # Ground-truth caption from COCO annotations (take the first if multiple)
            captions = id_to_captions.get(img_id) or []
            gt_caption = captions[0] if captions else ""

            writer.writerow([filename_out, gt_caption, entry.get("caption", "")])

    # Metrics
    overall = json.loads(result_json.read_text(encoding="utf-8")).get("overall", {})
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_csv.exists()
    fieldnames = ["checkpoint", "num_samples", "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge", "cider"]
    row = {
        "checkpoint": str(model_path),
        "num_samples": len(predictions),
        "bleu1": overall.get("Bleu_1"),
        "bleu2": overall.get("Bleu_2"),
        "bleu3": overall.get("Bleu_3"),
        "bleu4": overall.get("Bleu_4"),
        "meteor": overall.get("METEOR"),
        "rouge": overall.get("ROUGE_L"),
        "cider": overall.get("CIDEr"),
    }
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"[DONE] Predictions -> {predictions_csv}, metrics -> {metrics_csv}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Complete Image Captioning Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Extract
    p_ext = sub.add_parser("extract", help="Extract bottom-up features")
    p_ext.add_argument("--images_dir", type=Path, help="Folder of input images")
    p_ext.add_argument("--image_path", type=Path, help="Single image file")
    p_ext.add_argument("--start_image_id", type=int, default=2_000_000)
    p_ext.add_argument("--feature_out", type=Path, default=ROOT / "result/features/ktvicbu_att")
    p_ext.add_argument("--json_out", type=Path, default=ROOT / "result/features/custom_dataset.json")
    p_ext.add_argument("--split_name", type=str, default="test")

    # Caption
    p_cap = sub.add_parser("caption", help="Generate captions")
    p_cap.add_argument("--dataset_json", type=Path, required=True)
    p_cap.add_argument("--feature_path", type=Path, required=True)
    p_cap.add_argument("--model", type=Path, default=ROOT / "../checkpoints/model-best.pth")
    p_cap.add_argument("--infos", type=Path, default=ROOT / "../checkpoints/infos_ktvic_test-best.pkl")
    p_cap.add_argument("--num_images", type=int, default=-1)
    p_cap.add_argument("--device", type=str, default="cuda")
    p_cap.add_argument("--split", type=str, default="test")
    p_cap.add_argument("--num_workers", type=int, default=0)
    p_cap.add_argument("--run_id", type=str, default="custom_test")
    p_cap.add_argument("--predictions_csv", type=Path, default=ROOT / "result/caption_predictions.csv")
    p_cap.add_argument("--metrics_csv", type=Path, default=ROOT / "result/evaluate_valid_data.csv")
    p_cap.add_argument("--beam_size", type=int, default=1)
    p_cap.add_argument("--batch_size", type=int, default=1)
    p_cap.add_argument("--language_eval_bleu_only", type=int, default=None)
    p_cap.add_argument("--bleu_only", type=int, default=None)
    p_cap.add_argument("--language_eval_json", type=Path, default=None)
    p_cap.add_argument("--dump_images", type=int, default=0)
    p_cap.add_argument("--image_root", type=Path, default=None)
    p_cap.add_argument("--dump_json", type=int, default=0)

    # Full Pipeline
    p_full = sub.add_parser("full", help="Run full pipeline: Extract -> Caption")
    p_full.add_argument("--images_dir", type=Path, required=True, help="Folder of input images")
    p_full.add_argument("--output_root", type=Path, default=ROOT / "result/pipeline_output", help="Root for outputs")
    p_full.add_argument("--model", type=Path, default=ROOT / "../checkpoints/model-best.pth")
    p_full.add_argument("--infos", type=Path, default=ROOT / "../checkpoints/infos_ktvic_test-best.pkl")
    p_full.add_argument("--device", type=str, default="cuda")
    p_full.add_argument("--batch_size", type=int, default=1)
    p_full.add_argument("--beam_size", type=int, default=1)
    p_full.add_argument("--run_id", type=str, default="pipeline_run")
    p_full.add_argument("--split_name", type=str, default="test")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.cmd == "extract":
        img_paths = []
        if getattr(args, "image_path", None):
            img_paths = [args.image_path]
        elif getattr(args, "images_dir", None):
            img_paths = [p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
        extract_features_with_frcnn(
            image_paths=img_paths,
            start_image_id=args.start_image_id,
            feature_out=args.feature_out,
            json_out=args.json_out,
            split_name=args.split_name,
        )
    elif args.cmd == "caption":
        run_caption_eval(
            dataset_json=args.dataset_json,
            feature_path=args.feature_path,
            model_path=args.model,
            infos_path=args.infos,
            num_images=args.num_images,
            device=args.device,
            split=args.split,
            num_workers=args.num_workers,
            run_id=args.run_id,
            predictions_csv=args.predictions_csv,
            metrics_csv=args.metrics_csv,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            language_eval_bleu_only=args.language_eval_bleu_only,
            bleu_only=args.bleu_only,
            language_eval_json=args.language_eval_json,
            dump_images=args.dump_images,
            image_root=args.image_root,
            dump_json=args.dump_json,
        )
    elif args.cmd == "full":
        # Setup paths
        out_root = args.output_root
        out_root.mkdir(parents=True, exist_ok=True)
        
        feature_out = out_root / "features"
        dataset_json = out_root / "dataset.json"
        predictions_csv = out_root / "predictions.csv"
        metrics_csv = out_root / "metrics.csv"
        
        # 1. Extract
        print("=== [1/2] Extracting Features ===")
        img_paths = [p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        extract_features_with_frcnn(
            image_paths=img_paths,
            start_image_id=1, # Reset IDs for this batch
            feature_out=feature_out,
            json_out=dataset_json,
            split_name=args.split_name,
        )
        
        # 2. Caption
        print("=== [2/2] Generating Captions ===")
        run_caption_eval(
            dataset_json=dataset_json,
            feature_path=feature_out,
            model_path=args.model,
            infos_path=args.infos,
            num_images=-1,
            device=args.device,
            split=args.split_name,
            num_workers=0,
            run_id=args.run_id,
            predictions_csv=predictions_csv,
            metrics_csv=metrics_csv,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            language_eval_bleu_only=1, # Force BLEU only to avoid SPICE issues by default
            dump_images=0,
            image_root=args.images_dir,
            dump_json=0,
        )

if __name__ == "__main__":
    main()
