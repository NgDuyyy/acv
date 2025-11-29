"""Dataset definitions used throughout the project."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from config import PROCESSED_TEST_DIR, PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR

_SPLIT_DIR = {
    'TRAIN': PROCESSED_TRAIN_DIR,
    'VAL': PROCESSED_VAL_DIR,
    'TEST': PROCESSED_TEST_DIR,
}

_SPLIT_FILE_PREFIX = {
    'TRAIN': 'TRAIN',
    'VAL': 'VAL',
    'TEST': 'TEST',
}


class CaptionDataset(Dataset):
    """PyTorch dataset for the captioning problem."""

    def __init__(self, data_name: str, split: str, transform=None):
        self.split = split.upper()
        if self.split not in _SPLIT_DIR:
            raise ValueError(f"Unsupported split: {split}")

        split_dir = _SPLIT_DIR[self.split]
        prefix = _SPLIT_FILE_PREFIX[self.split]
        self.data_folder = Path(split_dir)
        hdf5_path = self.data_folder / f"{prefix}_IMAGES_{data_name}.hdf5"
        captions_path = self.data_folder / f"{prefix}_CAPTIONS_{data_name}.json"
        caplens_path = self.data_folder / f"{prefix}_CAPLENS_{data_name}.json"

        self.h = h5py.File(os.fspath(hdf5_path), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_per_image']

        with captions_path.open('r', encoding='utf-8') as handle:
            self.captions = json.load(handle)
        with caplens_path.open('r', encoding='utf-8') as handle:
            self.caplens = json.load(handle)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, idx):
        # The Nth caption corresponds to the (N // cpi)-th image
        img = torch.FloatTensor(self.imgs[idx // self.cpi] / 255.0)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[idx])
        caplen = torch.LongTensor([self.caplens[idx]])

        if self.split == 'TRAIN':
            return img, caption, caplen

        start = (idx // self.cpi) * self.cpi
        end = start + self.cpi
        all_captions = torch.LongTensor(self.captions[start:end])
        return img, caption, caplen, all_captions

    def __len__(self) -> int:
        return self.dataset_size
