# Image Captioning with GAN & RL

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![License](https://img.shields.io/badge/License-MIT-green)

Vietnamese Image Captioning project using Attention-based LSTM architecture, trained with Maximum Likelihood Estimation (MLE) and optimized using Adversarial Training (GAN) & Reinforcement Learning (SCST) to maximize CIDEr score.

## Project Structure

```text
acv/
├── configs                         # Configuration files
│   └── config.yaml                 # Hyperparameters (batch_size, lr, paths...)
├── data                            # Dataset directory (Not tracked by git)
│   ├── raw                         # Raw images (KTVIC, COCOVN)
│   ├── processed                   # Preprocessed data (features, vocab)
│   │   ├── vocab.json              # Word to ID mapping
│   │   ├── train_images.npy        # ResNet extracted features
│   │   ├── train_labels.npy        # Encoded caption IDs
│   │   └── train_ref.pkl           # Reference captions for Cider Reward
│   └── test_custom                 # Evaluation data
│       ├── images                  # 500 test images (.jpg)
│       └── test_captions.json      # Ground truth captions
├── src                             # Main Source Code
│   ├── __init__.py
│   ├── data_loader                 # Data loading logic
│   │   ├── __init__.py
│   │   └── dataset.py              # Custom Dataset class
│   ├── models                      # Neural Network Architectures
│   │   ├── layers                  # Sub-layers (Attention, Core LSTM)
│   │   ├── generator.py            # Caption Generator (Agent)
│   │   └── discriminator.py        # Discriminator (Judge)
│   ├── modules                     # Helper modules
│   │   ├── loss.py                 # SequenceLoss & ReinforceLoss
│   │   └── metrics.py              # CIDEr metric implementation
│   ├── engine                      # Training engines
│   │   ├── trainer_mle.py          # Pre-training logic
│   │   ├── trainer_gan.py          # GAN/RL training logic
│   │   └── evaluator.py            # Validation logic
│   └── utils                       # Utilities
│       ├── process_text.py         # Text cleaning & Vocab building
│       └── extract_features.py     # Image feature extraction (ResNet)
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation script
├── requirements.txt                # Dependencies
└── README.md

```

## Experimental Guide

### 1. Clone the repository

```bash
git clone -b conditional-gan --single-branch https://github.com/NgDuyyy/acv.git
cd acv

```

### 2. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Data Preparation

Since the dataset is not included in the repository, you need to prepare the data manually.

**Step 1: Download Raw Data**
Place your images (KTVIC, COCOVN) into `data/raw/`.

**Step 2: Preprocessing**
Run the utility scripts to build the vocabulary and extract image features.

```bash
# 1. Process text and build vocabulary
python src/utils/process_text.py --input_json data/raw/captions.json --output_dir data/processed

# 2. Extract image features using ResNet
python src/utils/extract_features.py --image_dir data/raw/images --output_dir data/processed

```

After this step, ensure `data/processed/` contains `vocab.json`, `train_images.npy`, `val_images.npy`, `train_labels.npy`, `val_labels.npy`, and `train_ref.pkl`.

### 4. Training

The training process consists of two phases: **MLE Pre-training** and **GAN/RL Fine-tuning**.

**Run Training (Phase 1 & 2 combined):**

```bash
python train.py --checkpoint_dir checkpoints --batch_size 32 --pretrain_epochs 25 --gan_epochs 25

```

* **Phase 1 (MLE):** The model learns to generate grammatically correct sentences using Cross-Entropy Loss.
* **Phase 2 (GAN/RL):** The model is fine-tuned using SCST (Self-Critical Sequence Training) to directly optimize the CIDEr metric.

### 5. Evaluation / Inference

Evaluate the model on the custom test set (500 images).

```bash
python evaluate.py \
  --model_path checkpoints/best_model.pth \
  --image_folder data/test_custom/images \
  --output_json results/predictions.json

```

## Results

| Metric | MLE (Phase 1) | GAN/RL (Phase 2) |
| --- | --- | --- |
| **BLEU-4** | 0.xx | 0.xx |
| **METEOR** | 0.xx | 0.xx |
| **ROUGE-L** | 0.xx | 0.xx |
| **CIDEr** | 0.xx | 0.xx |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
