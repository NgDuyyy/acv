# Image Captioning Attention Baseline

![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch)

Image Captioning pipeline for Vietnamese descriptions built on a ResNet101 encoder + additive-attention LSTM decoder.
Training uses maximum-likelihood with scheduled evaluation, while inference relies on beam-search decoding. Utilities are
included for dataset preparation, visualization, and experiment logging.

## Project Structure

```text
Image-Captioning/
├── config.py                   # Hyperparameters (batch_size, lr, paths...)
├── src/
│   ├── dataloader/             # Data ingestion and preprocessing helpers
│   ├── models/                 # Encoder/Decoder definitions
│   ├── utils/                  # Training utilities, metrics, checkpoint helpers
│   └── scripts/                # CLI helpers (prepare_data, caption, visualize, ...)
├── train.py       # Train loop + validation + logging
├── eval.py                     # Full-metric evaluation
├── inference.py                # Beam-search helper for notebooks
└── run_single_inference.py     # Lightweight CLI inference
```

## Experimental Guide

### 1. Clone & Environment

```bash
git clone -b encoder_decoder --single-branch https://github.com/NgDuyyy/acv.git
cd Image-Captioning
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Layout

Since the dataset is not included in the repository, you need to prepare the data manually.

**Download Raw Data**
`https://drive.google.com/drive/u/0/folders/1lN91FJxCL4jkXU1U1I7JcA-M88K1owpf`
Place your images (KTVIC, COCOVN) into `data/raw/`.

```
data/
  train/{images/,train_data.json,processed/}
  valid/{images/,val_data.json,processed/}
  test/{images/,test_data.json,processed/}
```

- JSON files must follow COCO-style fields: `images[id, filename]`, `annotations[image_id, caption]`.
- Images are JPEG/PNG; they will be resized to 256x256 when building the dataset.

Generate processed splits (HDF5 + tokenized captions):

```powershell
python src/scripts/prepare_data.py --max-len 50 --captions-per-image 5 --min-word-freq 5 --force
```

The command populates `data/*/processed/` with `*_IMAGES_*.hdf5`, `*_CAPTIONS_*.json`, `*_CAPLENS_*.json`, and a
`WORDMAP_*.json` stored under `data/train/processed/`.

### 3. Training Loop

```powershell
python train.py `
    --patience 20 `
    --lr-patience 8 `
    --max-len 50 `
    --captions-per-image 5 `
    --min-word-freq 5
```

- Best checkpoints land in `result/pretrained_parameters/` using validation CIDEr.
- Training history (loss, CIDEr, epoch time) streams into `result/log_history/training_history.csv`.
- Tune batch size, learning rates, epochs, etc. directly in `config.py`.

### 4. Evaluation & Beam Search

```powershell
python eval.py `
    --split TEST `
    --beam-size 5 `
    --checkpoint result\pretrained_parameters\BEST_checkpoint_custom_5_cap_per_img_5_min_word_freq.pth.tar `
    --log-csv result\log_history\eval_history_test.csv
```

- `--split` accepts `TRAIN`, `VAL`, `TEST` or custom aliases (map them inside `dataloader/dataset.py`).
- Add `--no-log` if you only want printed metrics.
- Beam size of 5 is a good balance between quality and latency; adjust freely for experiments.

### 5. Inference Utilities

| Task | Command |
|------|---------|
| Caption one image | `python src/scripts/caption.py --img path/to/img.jpg --beam-size 5` |
| Quick CLI inference | `python run_single_inference.py --img … --checkpoint …` |

All scripts auto-search for `WORDMAP_*.json`. Provide `--word-map` to override or re-run `src/scripts/prepare_data.py` if the
word map is missing.

## Results

| Metric      | Phase 1: Encoder + Decoder(LSTM + Global mean encoding) (Val) | Phase 2: Encoder + Decoder(LSTM + Attention) (Val) | Inference: Encoder + Decoder(LSTM + Attention) (Test) |
|-------------|---------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------|
| **BLEU-4**  | 0.3468                                                        | 0.3568                                             | 0.0548                                                |
| **METEOR**  | 0.4974                                                        | 0.5107                                             | 0.2184                                                |
| **ROUGE**   | 0.5575                                                        | 0.5628                                             | 0.3070                                                |
| **CIDEr**   | 1.2361                                                        | 1.2674                                             | 0.8798                                                |
| **SPICE**   | 0.4420                                                        | 0.5611                                             | 0.3198                                                |
