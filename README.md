![Python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c?logo=pytorch)

# Running Instructions

## 1. Overview

The project implements an image captioning model using a ResNet101 encoder and an LSTM decoder with beam search for
inference. Key components:

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

## 7. Inference & Visualization

### 7.1 Single Image Captioning

```powershell
python run_single_inference.py `
    --img data/test/images/000123.jpg `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --beam-size 5
```

Use the optional `--word-map` if you need to specify a different `WORDMAP_*.json` file. The result is printed to the
console.

### 7.2 Generate GT vs. Pred PDF for 3 images

```powershell
python scripts/visualize_predictions.py `
    --images data/test/images/000098.jpg data/test/images/000445.jpg data/test/images/000500.jpg `
    --gt-json data/test/test_data.json `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --output result\visualizations\demo.pdf
```

- The script will take up to the first 3 images in the list, run beam search, and save each image as a separate PDF
  file (e.g., `demo_000101.pdf`).
- Defaults can be adjusted in the `DEFAULT_IMAGE_PATHS` variable if you prefer not to pass arguments.

## 8. Plotting Training History

Once you have `training_history.csv`, run:

```powershell
python plot_training_history.py `
    --history result/log_history/training_history.csv `
    --output charts/training_history.png
```

Do not pass `--output` if you want to display the matplotlib window instead of saving to a file.

## 9. Tips & Troubleshooting

- **ModuleNotFoundError when running scripts in `scripts/`**: Ensure you run from the project root directory or have
  added `PROJECT_ROOT` to `PYTHONPATH`. `visualize_predictions.py` handles this automatically.
- Incorrect data paths: Check `config.py` to ensure `*_DIR` and `RAW_*` variables point to the correct directories you
  are using (e.g., when creating an additional test_vx, update `TEST_DIR`).
- Want to try different beam sizes: Use the `--beam-size` argument for `eval.py`, `run_single_inference.py`, and
  `scripts/visualize_predictions.py`.
- Clean up processed data: Manually delete the `data/*/processed` folders or use `python prepare_data.py --force`.
- Missing ground truth during visualization: The script will display a (`Ground truth not found`) message if it cannot
  map the `filename` in the JSON; check if `image_path.name` appears in the JSON's `images` list.

If adding new features (e.g., a different test split), simply update `config.py` and `utils/datasets.py`, then repeat
the steps above.
