"""Utility script to instantiate and optionally serialize the model components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    DATA_NAME_BASE,
    DECODER_DIM,
    DROPOUT,
    EMB_DIM,
    FINE_TUNE_ENCODER,
    WORD_MAP_PATH,
)
from models import Decoder, Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate encoder/decoder modules")
    parser.add_argument(
        '--save-init',
        type=Path,
        default=CHECKPOINT_DIR / f'init_state_{DATA_NAME_BASE}.pth.tar',
        help='optional path to store freshly initialized weights',
    )
    parser.add_argument(
        '--skip-save', action='store_true', help='only print model stats without saving'
    )
    return parser.parse_args()


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    args = parse_args()

    if not WORD_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Word map not found at {WORD_MAP_PATH}. Run prepare_data.py first."
        )

    with WORD_MAP_PATH.open('r', encoding='utf-8') as handle:
        word_map = json.load(handle)
    vocab_size = len(word_map)

    encoder = Encoder()
    encoder.fine_tune(FINE_TUNE_ENCODER)
    decoder = Decoder(
        embed_dim=EMB_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=vocab_size,
        dropout=DROPOUT,
    )

    print(f"Encoder parameters: {_count_parameters(encoder):,}")
    print(f"Decoder parameters: {_count_parameters(decoder):,}")

    if args.skip_save:
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'vocab_size': vocab_size,
        },
        args.save_init,
    )
    print(f"Initial weights saved to {args.save_init}")


if __name__ == '__main__':
    main()
