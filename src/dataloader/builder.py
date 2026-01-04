"""Utilities for transforming raw annotations into model-ready files."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from random import choice, sample, seed
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import nltk
from PIL import Image
from tqdm import tqdm


__all__ = ['process_coco_json', 'create_input_files', 'create_test_files']


def _ensure_tokenizer() -> None:
    """Download the punkt tokenizer the first time it is needed."""

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:  # pragma: no cover - executed only once per machine
        nltk.download('punkt')


def process_coco_json(
    json_path: Path, image_folder: Path, max_len: int
) -> Tuple[List[str], List[List[Sequence[str]]]]:
    """Read a simplified COCO JSON file and tokenize captions."""

    _ensure_tokenizer()
    with json_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    id_to_filename = {img['id']: img['filename'] for img in data['images']}
    id_to_captions: defaultdict[int, list[list[str]]] = defaultdict(list)

    for ann in data['annotations']:
        tokens = nltk.word_tokenize(ann['caption'].lower())
        if len(tokens) <= max_len:
            id_to_captions[ann['image_id']].append(tokens)

    img_paths: list[str] = []
    img_captions: list[list[list[str]]] = []
    for img_id, filename in id_to_filename.items():
        captions = id_to_captions.get(img_id, [])
        if not captions:
            continue
        img_paths.append(str(image_folder / filename))
        img_captions.append(captions)

    return img_paths, img_captions


def _write_split(
    impaths: Sequence[str],
    imcaps: Sequence[List[List[str]]],
    split: str,
    out_dir: Path,
    base_filename: str,
    captions_per_image: int,
    max_len: int,
    word_map: dict[str, int],
):
    h5_path = out_dir / f"{split}_IMAGES_{base_filename}.hdf5"
    if h5_path.exists():
        print(f"File {h5_path} đã tồn tại. Bỏ qua.")
        return

    enc_captions = []
    caplens = []
    with h5py.File(os.fspath(h5_path), 'a') as h5_file:
        h5_file.attrs['captions_per_image'] = captions_per_image
        images = h5_file.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

        print(f"\nĐang đọc và lưu {split} images...\n")
        for idx, path in enumerate(tqdm(impaths)):
            if len(imcaps[idx]) < captions_per_image:
                captions = imcaps[idx] + [
                    choice(imcaps[idx]) for _ in range(captions_per_image - len(imcaps[idx]))
                ]
            else:
                captions = sample(imcaps[idx], k=captions_per_image)

            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                img = np.asarray(img).transpose(2, 0, 1)
            except Exception as exc:  # pragma: no cover - defensive path
                print(f"Lỗi đọc ảnh {path}: {exc}")
                img = np.zeros((3, 256, 256), dtype='uint8')

            images[idx] = img

            for caption in captions:
                encoded_caption = [word_map['<start>']]
                encoded_caption.extend(word_map.get(word, word_map['<unk>']) for word in caption)
                encoded_caption.append(word_map['<end>'])
                padded_caption = encoded_caption + [word_map['<pad>']] * (max_len - len(caption))
                enc_captions.append(padded_caption)
                caplens.append(len(caption) + 2)

    captions_path = out_dir / f"{split}_CAPTIONS_{base_filename}.json"
    caplens_path = out_dir / f"{split}_CAPLENS_{base_filename}.json"
    with captions_path.open('w', encoding='utf-8') as handle:
        json.dump(enc_captions, handle)
    with caplens_path.open('w', encoding='utf-8') as handle:
        json.dump(caplens, handle)


def create_input_files(
    train_json_path: Path,
    val_json_path: Path,
    train_image_folder: Path,
    val_image_folder: Path,
    captions_per_image: int,
    min_word_freq: int,
    max_len: int,
    train_output_dir: Path,
    val_output_dir: Path,
    base_filename: str,
    word_map_path: Path,
) -> None:
    """Build HDF5/JSON bundles for train/val splits."""

    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    word_map_path.parent.mkdir(parents=True, exist_ok=True)

    word_freq = Counter()

    print(f"Đang xử lý tập Train từ: {train_json_path}")
    train_image_paths, train_image_captions = process_coco_json(
        train_json_path, train_image_folder, max_len
    )
    for captions in train_image_captions:
        for token_list in captions:
            word_freq.update(token_list)

    print(f"Đang xử lý tập Val từ: {val_json_path}")
    val_image_paths, val_image_captions = process_coco_json(
        val_json_path, val_image_folder, max_len
    )

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    with word_map_path.open('w', encoding='utf-8') as handle:
        json.dump(word_map, handle)

    seed(123)
    split_config = (
        (train_image_paths, train_image_captions, 'TRAIN', train_output_dir),
        (val_image_paths, val_image_captions, 'VAL', val_output_dir),
    )

    for impaths, imcaps, split, out_dir in split_config:
        _write_split(
            impaths,
            imcaps,
            split,
            out_dir,
            base_filename,
            captions_per_image,
            max_len,
            word_map,
        )


def create_test_files(
    test_json_path: Path,
    test_image_folder: Path,
    captions_per_image: int,
    max_len: int,
    test_output_dir: Path,
    base_filename: str,
    word_map_path: Path,
) -> None:
    """Build processed files for the test split using an existing word map."""

    if not word_map_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy word map tại {word_map_path}. Hãy chạy prepare_data cho train/val trước."
        )

    test_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang xử lý tập Test từ: {test_json_path}")
    test_image_paths, test_image_captions = process_coco_json(
        test_json_path, test_image_folder, max_len
    )

    with word_map_path.open('r', encoding='utf-8') as handle:
        word_map = json.load(handle)

    _write_split(
        test_image_paths,
        test_image_captions,
        'TEST',
        test_output_dir,
        base_filename,
        captions_per_image,
        max_len,
        word_map,
    )
