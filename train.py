from __future__ import annotations

import argparse
import csv
import json
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from config import (
    ATTENTION_DIM,
    BATCH_SIZE,
    DATA_NAME_BASE,
    DECODER_DIM,
    DECODER_LR,
    DEVICE,
    DROPOUT,
    EMB_DIM,
    ENCODER_LR,
    FINE_TUNE_ENCODER,
    GRAD_CLIP,
    LOG_HISTORY_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    RAW_TRAIN_IMAGES,
    RAW_TRAIN_JSON,
    RAW_VAL_IMAGES,
    RAW_VAL_JSON,
    WORD_MAP_PATH,
    WORKERS,
    CUDNN_BENCHMARK,
    EPOCHS,
)
from src.dataloader import CaptionDataset, create_input_files
from src.models import Decoder, Encoder
from src.utils import (
    adjust_learning_rate,
    save_checkpoint,
    train_one_epoch,
    validate,
)


def _format_history_rows(rows: list[dict]) -> list[dict]:
    """Return a copy of history rows with floats formatted to fixed decimals."""

    def _fmt(value: float) -> str:
        return f"{value:.6f}"

    formatted = []
    for row in rows:
        formatted.append(
            {
                'epoch': row['epoch'],
                'epoch_time_sec': _fmt(row['epoch_time_sec']),
                'train_loss': _fmt(row['train_loss']),
                'val_loss': _fmt(row['val_loss']),
                'val_cider': _fmt(row['val_cider']),
            }
        )
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the captioning model")
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument(
        '--lr-patience',
        type=int,
        default=8,
        help='epochs without improvement before decaying LR',
    )
    parser.add_argument('--max-len', type=int, default=50, help='max caption length')
    parser.add_argument('--captions-per-image', type=int, default=5)
    parser.add_argument('--min-word-freq', type=int, default=5)
    return parser.parse_args()


def ensure_processed_files(args: argparse.Namespace) -> None:
    if WORD_MAP_PATH.exists():
        return
    print('--- BẮT ĐẦU TẠO DATASET ---')
    create_input_files(
        train_json_path=RAW_TRAIN_JSON,
        val_json_path=RAW_VAL_JSON,
        train_image_folder=RAW_TRAIN_IMAGES,
        val_image_folder=RAW_VAL_IMAGES,
        captions_per_image=args.captions_per_image,
        min_word_freq=args.min_word_freq,
        max_len=args.max_len,
        train_output_dir=PROCESSED_TRAIN_DIR,
        val_output_dir=PROCESSED_VAL_DIR,
        base_filename=DATA_NAME_BASE,
        word_map_path=WORD_MAP_PATH,
    )
    print('--- TẠO DATASET THÀNH CÔNG ---')


def load_word_map() -> dict:
    with WORD_MAP_PATH.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def build_dataloaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_NAME_BASE, 'TRAIN', transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(DATA_NAME_BASE, 'VAL', transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    args = parse_args()
    ensure_processed_files(args)

    word_map = load_word_map()
    vocab_size = len(word_map)

    decoder = Decoder(
        embed_dim=EMB_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=vocab_size,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=DECODER_LR
    )

    encoder = Encoder().to(DEVICE)
    encoder.fine_tune(FINE_TUNE_ENCODER)
    encoder_optimizer = None
    if FINE_TUNE_ENCODER:
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=ENCODER_LR,
        )

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    train_loader, val_loader = build_dataloaders()

    history_rows = []
    log_file_path = LOG_HISTORY_DIR / 'training_history.csv'
    epochs_since_improvement = 0
    best_cider = 0.0

    print(f'Starting Training for {EPOCHS} epochs...')
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        if epochs_since_improvement == args.patience:
            print('Early stopping triggered.')
            break
        if (
            epochs_since_improvement > 0
            and epochs_since_improvement % args.lr_patience == 0
        ):
            adjust_learning_rate(decoder_optimizer, 0.8)
            if encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train_loss = train_one_epoch(
            train_loader,
            encoder,
            decoder,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            epoch,
            GRAD_CLIP,
        )
        recent_cider, recent_val_loss = validate(
            val_loader, encoder, decoder, criterion, word_map
        )

        epoch_duration = time.time() - epoch_start
        history_rows.append(
            {
                'epoch': epoch,
                'epoch_time_sec': epoch_duration,
                'train_loss': train_loss,
                'val_loss': recent_val_loss,
                'val_cider': recent_cider,
            }
        )
        LOG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with log_file_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=['epoch', 'epoch_time_sec', 'train_loss', 'val_loss', 'val_cider']
            )
            writer.writeheader()
            writer.writerows(_format_history_rows(history_rows))

        is_best = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)

        if not is_best:
            epochs_since_improvement += 1
            print(
                f"\nEpochs since last improvement: {epochs_since_improvement} (Best CIDEr: {best_cider:.4f})\n"
            )
        else:
            epochs_since_improvement = 0
            save_checkpoint(
                DATA_NAME_BASE,
                epoch,
                epochs_since_improvement,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                recent_cider,
                is_best,
            )
            print(f"Model saved! New Best CIDEr: {best_cider:.4f}")


if __name__ == '__main__':
    main()
