# VisionText AI: Advanced Image Captioning with Scene Graphs

![Python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-103369?logo=ultralytics)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

This is a comprehensive codebase for the **Image Captioning** project, combining the power of Computer Vision and
Natural Language Processing. The project implements a dual feature extraction pipeline: using **Faster R-CNN** for
regional features (Bottom-Up Attention) and **RelTR (Relation Transformer)** to construct Scene Graphs. These features
are integrated into an **LSTM** decoder model (with Attention mechanism) to generate accurate, natural, and semantically
rich image descriptions.

## Directory Structure

```text
├── configs
│   └── lstm_train.yml
├── data
│   ├── features
│   ├── LSTM
│   ├── RelTR_ckpt
│   ├── test
│   ├── train
│   └── val
├── models
│   ├── feature_extracting
│   ├── LSTM
│   └── RelTR
├── scripts
│   ├── create_reltr_hdf5.py
│   ├── extract_reltr_features.py
│   ├── generate_tsv.py
│   ├── make_bu_data.py
│   ├── prepare_data_merged.py
│   ├── prepro_labels.py
│   └── prepro_reference_json.py
├── utils
│   ├── feature_extracting
│   ├── LSTM
│   └── RelTR
├── complete_image_captioning_pipeline.py
├── eval.py
├── infer.py
└── train.py
```

## Experimental Results

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Faster R-CNN) | 71.6 | 38.8 | 34.7 | 55.7 | 121.0 | 8.7 |
| RelTR (Cross-Entropy) | 72.1 | 39.1 | 34.7 | 55.9 | 123.3 | 8.7 |
| **RelTR + SCST** | **75.9** | **41.0** | **35.3** | **57.3** | **138.7** | **8.9** |

## 1. Installation

Install necessary libraries:

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

Ensure image data is placed in `data/train`, `data/val`, and `data/test`.

### Step 1: Extract Visual Features

Mandatory for all models. Uses Faster R-CNN to extract Region of Interest (ROI) features.

* **Command:**
  ```bash
  python scripts/generate_tsv.py --cuda --cfg models/cfgs/res101.yml --net res101
  ```
  *(Note: Configure the correct image and output paths in the script or via arguments if available)*

* **Input:** Images in `data/train`, `data/val`, `data/test`.
* **Output:** `.tsv` files saved in `data/features/features_tsv/`.

### Step 2: Extract Scene Graph Features (RelTR)

Mandatory if using a model with RelTR. This step generates an `.h5` file containing object relationship information.

* **Command:**
  ```bash
  python scripts/create_reltr_hdf5.py --json_path data/LSTM/data_merged.json --data_root data --output_h5 data/features/reltr_features.h5
  ```
* **Input:** Merged JSON file (`data/LSTM/data_merged.json`) and root image directory.
* **Output:** `data/features/reltr_features.h5` containing relational features (Subject-Predicate-Object).

### Step 3: Feature Formatting

Convert TSV files to `.npy` format for faster training and organize them into correct directories.

* **Command:**
  ```bash
  python scripts/make_bu_data.py
  ```
* **Input:** `.tsv` files in `data/features/features_tsv/`.
* **Output:** Directories `features_extracted_att`, `features_extracted_fc`, `features_extracted_box` inside
  `data/features`.

### Step 4: Data Preparation

Merge and process annotation files from train/val sets into a unified JSON file for training.

* **Command:**
  ```bash
  python scripts/prepare_data_merged.py
  ```
  *(Note: Check and modify `train_path`, `val_path`, `output_path` in this file if necessary)*

* **Input:** `data/train/train_data.json`, `data/val/val_data.json`.
* **Output:** `data/LSTM/data_merged.json`.

### Step 5: Create Evaluation Reference

Create a standard reference JSON file for evaluation (required for `coco-caption`).

* **Command:**
  ```bash
  python scripts/prepro_reference_json.py --input_json data/val/val_data.json --output_json data/LSTM/val_reference.json
  ```

### Step 6: Label Preprocessing

Create vocabulary and H5 file containing encoded labels for the model.

* **Command:**
  ```bash
  python scripts/prepro_labels.py --input_json data/LSTM/data_merged.json --output_json data/LSTM/data_label.json --output_h5 data/LSTM/data
  ```
* **Input:** `data/LSTM/data_merged.json`.
* **Output:** `data/LSTM/data_label.h5` and `data/LSTM/data_label.json`.

## 3. Training

Train the Image Captioning model.

* **Command:**
  ```bash
  python train.py --cfg configs/lstm_train.yml
  ```
* **Key Parameters:**
    * `--id`: Identifier for this run.
    * `--caption_model`: Model type (e.g., `updown`, `att2in`, `transformer`).
    * `--checkpoint_path`: Directory to save checkpoints.
    * **Note for RelTR:** To enable Scene Graph features, ensure `configs/lstm_train.yml` has the line
      `input_rel_dir: data/features/reltr_features.h5`. To run Baseline (no Graph), leave this line empty (
      `input_rel_dir: ""`).
    * Parameters are configured in `configs/lstm_train.yml`.
* **Output:**
    * Model Checkpoint: `result/checkpoints/model-best.pth`.
    * Training Info: `result/checkpoints/infos_my_run-best.pkl`.
    * Training Log: `result/training_history_data.csv`.

### SCST Optimization (Self-Critical Sequence Training)

After training with Cross-Entropy (step above), we perform fine-tuning using SCST to directly optimize the **CIDEr**
metric. This significantly improves the quality and naturalness of captions.

* **Prerequisites:**
    1. Generate N-gram cache (runs once):
       ```bash
       python scripts/prepro_ngrams.py --input_json data/LSTM/data_merged.json --dict_json data/LSTM/data_label.json --output_pkl data/LSTM/data-train --split train
       ```
    2. Ensure `configs/lstm_scst.yml` points to the best Cross-Entropy checkpoint (`start_from`).

* **Command:**
  ```bash
  python train.py --cfg configs/lstm_scst.yml
  ```
* **Key Configs (`lstm_scst.yml`):**
    * `learning_rate`: 5e-5 (Lower than initial training).
    * `max_epochs`: 30 (with Early Stopping).
    * `self_critical_after`: 0 (Enable immediately).

* **Output:**
    * Optimized Model: `result/final_term/scst/log_lstm_reltr_scst/model-best.pth`.

## 4. Evaluation

Evaluate the trained model on val/test sets, generate captions, and calculate metrics (BLEU, CIDEr, SPICE...).

* **Command:**
  ```bash
  python eval.py \
    --model result/final_term/scst/log_lstm_reltr_scst/model-best.pth \
    --infos_path result/final_term/scst/log_lstm_reltr_scst/infos_reltr_scst-best.pkl \
    --input_json data/LSTM/val_reference.json \
    --language_eval_json data/LSTM/val_reference.json \
    --input_att_dir data/features/features_extracted_att \
    --split val \
    --language_eval 1 \
    --save_csv_results 1 \
    --predictions_csv result/predictions_reltr_scst_val.csv \
    --metrics_csv result/scores_reltr_scst_val.csv
  ```

* **Parameter Explanation:**
    * `--model`, `--infos_path`: Paths to model checkpoint and info file.
    * `--input_json` & `--language_eval_json`: JSON file containing image list and ground truth captions.
    * `--input_att_dir`: Directory containing corresponding attention features.
    * `--language_eval 1`: Enable language evaluation.
    * `--save_csv_results 1`: Enable saving results to CSV.

* **Output:**
    * **`result/predictions_reltr_scst_val.csv`**: Contains filenames, ground truth captions, and predicted captions.
    * **`result/scores_reltr_scst_val.csv`**: Contains detailed scores (BLEU-1..4, METEOR, ROUGE_L, CIDEr, SPICE).

## 5. Full Pipeline

Run the entire pipeline from feature extraction to caption generation for a new directory of images.

* **Command:**
  ```bash
  python complete_image_captioning_pipeline.py full --images_dir data/my_new_images --output_root result/my_output --model result/final_term/scst/log_lstm_reltr_scst/model-best.pth --infos result/final_term/scst/log_lstm_reltr_scst/infos_reltr_scst-best.pkl
  ```

## 6. Simple Inference

Run inference on a single image for a quick check.

* **Command:**
  ```bash
  python infer.py --image test_image/my_image.jpg --frcnn_model models/feature_extracting/pretrained_model/faster_rcnn_res101_vg.pth --caption_model result/final_term/scst/log_lstm_reltr_scst/model-best.pth --infos_path result/final_term/scst/log_lstm_reltr_scst/infos_reltr_scst-best.pkl --gpu
  ```

* **Parameters:**
    * `--image`: Path to the image.
    * `--frcnn_model`: Path to Faster R-CNN checkpoint (feature extractor).
    * `--caption_model`: Path to trained Captioning model checkpoint.
    * `--infos_path`: Path to `infos.pkl` file corresponding to the checkpoint.
    * `--gpu`: Add this flag to run on GPU (defaults to CPU if omitted).

* **Output:** Predicted caption printed directly to the console.

### Inference with RelTR (Scene Graph)

To use the model integrated with Scene Graph:

* **Command:**
  ```bash
  python infer.py --use_reltr --image test_image/my_image.jpg ... (other parameters as above)
  ```
* **Additional Parameters:**
    * `--use_reltr`: Enable RelTR usage.
    * `--reltr_model_path`: Path to RelTR checkpoint (default: `data/RelTR_ckpt/checkpoint0149.pth`).

## Notes

* To use CPU instead of GPU, add the flag `--force_cpu 1` (training will be very slow).
* Evaluation results and logs are centrally saved in the `result/` directory for easy management.

## 7. Web Demo

The project provides an intuitive web interface to experience the Image Captioning model.

![Web Demo Screenshot](static/Screenshot%202025-12-29%20234649.png)

### Features:

* **Modern Interface**: Dark Mode, AI/Futuristic style.
* **Easy to Use**: Supports Drag & Drop and Image Preview.
* **Smart Processing**: Auto-detects GPU/CPU, uses **Greedy Search + Block Trigrams** to generate natural,
  non-repetitive captions.

### How to Run:

Run the following command from the project root:

```bash
python app.py
```

Then access the browser at: **http://localhost:8000**
