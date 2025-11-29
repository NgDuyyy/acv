"""Entry-point script for building the processed captioning dataset."""

from __future__ import annotations

import argparse
import shutil

from config import (
    DATA_NAME_BASE,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    RAW_TEST_IMAGES,
    RAW_TEST_JSON,
    RAW_TRAIN_IMAGES,
    RAW_TRAIN_JSON,
    RAW_VAL_IMAGES,
    RAW_VAL_JSON,
    WORD_MAP_PATH,
)
from utils import create_input_files, create_test_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare captioning dataset files")
    parser.add_argument('--max-len', type=int, default=50, help='maximum caption length')
    parser.add_argument('--captions-per-image', type=int, default=5, help='captions per image')
    parser.add_argument('--min-word-freq', type=int, default=5, help='minimum word frequency')
    parser.add_argument(
        '--force', action='store_true', help='rebuild even if processed files exist'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    directories = (PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR)
    if args.force:
        for directory in directories:
            if directory.exists():
                shutil.rmtree(directory)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rebuild_train_val = args.force or not WORD_MAP_PATH.exists()
    if not rebuild_train_val:
        print("Processed train/val đã tồn tại. Bỏ qua build lại (dùng --force nếu cần).")

    if rebuild_train_val:
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

    if RAW_TEST_JSON.exists() and RAW_TEST_IMAGES.exists():
        create_test_files(
            test_json_path=RAW_TEST_JSON,
            test_image_folder=RAW_TEST_IMAGES,
            captions_per_image=args.captions_per_image,
            max_len=args.max_len,
            test_output_dir=PROCESSED_TEST_DIR,
            base_filename=DATA_NAME_BASE,
            word_map_path=WORD_MAP_PATH,
        )
    else:
        print("Bỏ qua test set vì thiếu test_data.json hoặc thư mục ảnh.")


if __name__ == '__main__':
    main()
