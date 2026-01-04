![Python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c?logo=pytorch)

# Running Instructions

## 1. Overview

The project implements an image captioning model using a ResNet101 encoder and an LSTM decoder. Key components:

- `config.py`: Centralizes all paths and hyperparameters.
- `prepare_data.py`: Converts raw data (JSON + images) into pre-processed HDF5/JSON format.
- `train_and_evaluate.py`: Trains the model, saves the best checkpoint, and logs history.
- `eval.py`: Runs beam search on a dataset (train/val/test/...) and logs BLEU/METEOR/ROUGE/CIDEr scores.
- `run_single_inference.py`: Captions a single image.
- `scripts/visualize_predictions.py`: Captions up to 3 images and generates PDFs displaying Ground Truth (GT) vs.
  predictions.
- `plot_training_history.py`: Plots loss/BLEU charts from the history file.

## 2. Requirements

- Python >= 3.9 (tested with 3.10).
- CUDA GPU is recommended, but CPU can be used (slower).
- Libraries listed in `requirements.txt` (PyTorch, torchvision, NLTK, pandas, matplotlib, etc.).

## 3. Environment Preparation

```powershell
# 1) Create and activate virtual env (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install libraries
pip install --upgrade pip
pip install -r requirements.txt
```

> **PowerShell Note**: If you want to break lines in a long command, use the backtick character `` ` `` at the end of
> the line, do not use `^`.

## 4. Data Preparation

The project expects the following directory structure (example):

```
data/
  train/
    images/              # JPEG Images
    train_data.json      # COCO-like JSON containing images/annotations fields
    processed/           # Will be created automatically
  valid/
    images/
    val_data.json
    processed/
  test/
    images/
    test_data.json
    processed/
```

The JSON files must contain an `images` field (with `id`, `filename`) and an `annotations` field (with `image_id`,
`caption`). After placing the data in the correct locations, run:

```powershell
python prepare_data.py --force
```

Important options:

- `--max-len`, `--captions-per-image`, `--min-word-freq`: Must match your desired training parameters.
- Omit `--force` if you only want to recreate the test set when train/val already exist.

## 5. Training

```powershell
python train_and_evaluate.py `
    --patience 20 `
    --lr-patience 8 `
    --max-len 50 `
    --captions-per-image 5 `
    --min-word-freq 5
```

Key information:

- The best checkpoint is saved in `result/pretrained_parameters/` (based on val BLEU-4).
- Training history is written to `result/log_history/training_history.csv` every epoch.
- Change general hyperparameters (batch size, embedding dim, lr, default epochs, etc.) in `config.py`.

## 6. Model Evaluation

Use `eval.py` to calculate BLEU/METEOR/ROUGE/CIDEr:

```powershell
python eval.py `
    --split TEST `
    --beam-size 5 `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --log-csv result\log_history\eval_history_test.csv
```

Notes:

- `--split` supports `TRAIN`, `VAL`, `TEST`, and aliases `TEST_V2 ... TEST_V6` (if you are reusing the current test
  folder for those variants). To use aliases, ensure `utils/datasets.py` maps them to `PROCESSED_TEST_DIR` or update
  `config.TEST_DIR` accordingly.
- Add `--no-log` if you only want to print results without writing to CSV.
- `beam-size = 5` usually yields good quality with reasonable time; you can verify this by changing it and comparing
  logs.

## 7. Result

| Metric      | Evaluate: Encoder + Decoder(LSTM + Global mean encoding) (Val) | Inference: Encoder + Decoder(LSTM + Global mean encoding) (Test) |
|-------------|----------------------------------------------------------------|------------------------------------------------------------------|
| **BLEU-4**  | 0.3469                                                         | 0.0627                                                           |
| **METEOR**  | 0.4975                                                         | 0.2364                                                           |
| **ROUGE**   | 0.5576                                                         | 0.3115                                                           |
| **CIDEr**   | 1.2361                                                         | 0.3821                                                           |
