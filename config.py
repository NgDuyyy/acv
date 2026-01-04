# config.py
# Tất cả tham số và đường dẫn tập trung trong file này để dễ cấu hình.

from pathlib import Path
import torch


# --- CẤU HÌNH ĐƯỜNG DẪN ---
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VALID_DIR = DATA_DIR / 'valid'
TEST_DIR = DATA_DIR / 'test'

RAW_TRAIN_JSON = TRAIN_DIR / 'train_data.json'
RAW_VAL_JSON = VALID_DIR / 'val_data.json'
RAW_TRAIN_IMAGES = TRAIN_DIR / 'images'
RAW_VAL_IMAGES = VALID_DIR / 'images'
RAW_TEST_JSON = TEST_DIR / 'test_data.json'
RAW_TEST_IMAGES = TEST_DIR / 'images'

PROCESSED_TRAIN_DIR = TRAIN_DIR / 'processed'
PROCESSED_VAL_DIR = VALID_DIR / 'processed'
PROCESSED_TEST_DIR = TEST_DIR / 'processed'

RESULT_DIR = PROJECT_ROOT / 'result'
LOG_HISTORY_DIR = RESULT_DIR / 'log_history'
CHECKPOINT_DIR = RESULT_DIR / 'pretrained_parameters'

for path in (PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR,
			 LOG_HISTORY_DIR, CHECKPOINT_DIR):
	path.mkdir(parents=True, exist_ok=True)

DATA_NAME_BASE = 'custom_5_cap_per_img_5_min_word_freq'
WORD_MAP_FILENAME = f'WORDMAP_{DATA_NAME_BASE}.json'
WORD_MAP_PATH = PROCESSED_TRAIN_DIR / WORD_MAP_FILENAME


# --- THAM SỐ MÔ HÌNH ---
EMB_DIM = 512
DECODER_DIM = 512
ATTENTION_DIM = 512
DROPOUT = 0.5
ENCODER_DIM = 2048  # Kích thước đầu ra của ResNet101

# --- THAM SỐ HUẤN LUYỆN ---
BATCH_SIZE = 32
WORKERS = 4
GRAD_CLIP = 5.
ENCODER_LR = 1e-4
DECODER_LR = 4e-4
FINE_TUNE_ENCODER = False
EPOCHS = 100


# --- CẤU HÌNH THIẾT BỊ ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDNN_BENCHMARK = True