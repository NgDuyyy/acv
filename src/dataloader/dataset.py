"""Dataset wrappers used by dataloaders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import h5py
import torch
from torch.utils.data import Dataset

from config import PROCESSED_TEST_DIR, PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR

__all__ = ['CaptionDataset']

_SPLIT_DIR: Dict[str, Path] = {
    'TRAIN': Path(PROCESSED_TRAIN_DIR),
    'VAL': Path(PROCESSED_VAL_DIR),
    'TEST': Path(PROCESSED_TEST_DIR),
}

_SPLIT_PREFIX = {
    'TRAIN': 'TRAIN',
    'VAL': 'VAL',
    'TEST': 'TEST',
}


class CaptionDataset(Dataset):
    """HDF5-backed dataset returning caption supervision tuples."""

    def __init__(
        self,
        data_name: str,
        split: str,
        transform=None,
        data_folder_override: str | Path | None = None,
    ) -> None:
        self.split = split.upper()
        if self.split not in _SPLIT_DIR:
            raise ValueError(f"Unsupported split: {split}")

        prefix = _SPLIT_PREFIX[self.split]
        base_dir = Path(data_folder_override) if data_folder_override else _SPLIT_DIR[self.split]
        self.data_folder = Path(base_dir)

        hdf5_path = self.data_folder / f"{prefix}_IMAGES_{data_name}.hdf5"
        captions_path = self.data_folder / f"{prefix}_CAPTIONS_{data_name}.json"
        caplens_path = self.data_folder / f"{prefix}_CAPLENS_{data_name}.json"

        if not hdf5_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại {hdf5_path}. Hãy chạy src/scripts/prepare_data.py trước."
            )

        self.hdf5_path = hdf5_path
        self._hdf5: Optional[h5py.File] = None
        self._images: Optional[h5py.Dataset] = None
        self.cpi: Optional[int] = None

        with captions_path.open('r', encoding='utf-8') as handle:
            self.captions = json.load(handle)
        with caplens_path.open('r', encoding='utf-8') as handle:
            self.caplens = json.load(handle)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, idx):  # type: ignore[override]
        images = self._ensure_images()
        assert self.cpi is not None
        image = torch.FloatTensor(images[idx // self.cpi] / 255.0)
        if self.transform is not None:
            image = self.transform(image)

        caption = torch.LongTensor(self.captions[idx])
        caplen = torch.LongTensor([self.caplens[idx]])

        if self.split == 'TRAIN':
            return image, caption, caplen

        start = (idx // self.cpi) * self.cpi
        end = start + self.cpi
        all_captions = torch.LongTensor(self.captions[start:end])
        return image, caption, caplen, all_captions

    def __len__(self) -> int:  # type: ignore[override]
        return self.dataset_size

    def _ensure_images(self):
        if self._hdf5 is None:
            self._hdf5 = h5py.File(os.fspath(self.hdf5_path), 'r')
            self._images = self._hdf5['images']
            self.cpi = self._hdf5.attrs['captions_per_image']
        assert self._images is not None
        return self._images

    def close(self) -> None:
        if self._hdf5 is not None:
            self._hdf5.close()
            self._hdf5 = None
            self._images = None
            self.cpi = None

    def __del__(self) -> None:
        self.close()
