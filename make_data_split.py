import os
import json
import random
import numpy as np
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm

INPUT_JSON = 'data/raw/captions.json'
OUTPUT_DIR = 'data/processed'
MIN_WORD_FREQ = 5
VAL_RATIO = 0.2
SEED = 42
MAX_LEN = 50



def build_vocab(captions, threshold):
    print(">> Đang thống kê tần suất từ (chỉ trên tập Train)...")
    counter = Counter()
    for cap in tqdm(captions):
        tokens = cap.lower().split()
        counter.update(tokens)

    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx = 4
    for word, cnt in counter.items():
        if cnt >= threshold:
            vocab[word] = idx
            idx += 1
    return vocab


def encode_captions(captions, vocab, max_len=50):
    captions_encoded = []
    for cap in tqdm(captions):
        tokens = cap.lower().split()
        vec = [vocab['<start>']]
        for token in tokens:
            vec.append(vocab.get(token, vocab['<unk>']))
        vec.append(vocab['<end>'])

        if len(vec) > max_len:
            vec = vec[:max_len]
        else:
            vec += [vocab['<pad>']] * (max_len - len(vec))

        captions_encoded.append(vec)
    return np.array(captions_encoded)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. ĐỌC DỮ LIỆU
    print(f">> Đang đọc file {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

        # Gom nhóm theo ID
    grouped_data = defaultdict(list)
    for item in raw_data:
        img_id = str(item['image_id'])
        caption = item['caption']
        grouped_data[img_id].append(caption)

    all_img_ids = list(grouped_data.keys())
    random.shuffle(all_img_ids)

    # Chia 80:20
    split_idx = int(len(all_img_ids) * (1 - VAL_RATIO))
    train_ids = all_img_ids[:split_idx]
    val_ids = all_img_ids[split_idx:]

    print(f">> Train set: {len(train_ids)} ảnh")
    print(f">> Val set:   {len(val_ids)} ảnh")

    # ================= XỬ LÝ TẬP TRAIN =================
    print("\n=== ĐANG XỬ LÝ TẬP TRAIN ===")

    train_captions_flat = []
    train_refs_dict = {}

    for img_id in train_ids:
        caps = grouped_data[img_id]
        train_refs_dict[img_id] = caps
        train_captions_flat.extend(caps)

    # 1. Xây Vocab (CHỈ TỪ TRAIN)
    vocab = build_vocab(train_captions_flat, MIN_WORD_FREQ)
    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    # 2. Lưu train_images.npy (ID ảnh)
    np.save(os.path.join(OUTPUT_DIR, 'train_images.npy'), train_ids)

    # 3. Mã hóa train labels
    train_labels = encode_captions(train_captions_flat, vocab, MAX_LEN)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)

    # 4. Lưu train_ref.pkl (Bắt buộc cho RL)
    with open(os.path.join(OUTPUT_DIR, 'train_ref.pkl'), 'wb') as f:
        pickle.dump(train_refs_dict, f)

    # 5. Lưu train_captions.json
    with open(os.path.join(OUTPUT_DIR, 'train_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(train_refs_dict, f, ensure_ascii=False)

    # ================= XỬ LÝ TẬP VAL =================
    print("\n=== ĐANG XỬ LÝ TẬP VAL ===")

    val_captions_flat = []
    val_refs_dict = {}

    for img_id in val_ids:
        caps = grouped_data[img_id]
        val_refs_dict[img_id] = caps
        val_captions_flat.extend(caps)

    # 1. Lưu val_images.npy (ID ảnh)
    np.save(os.path.join(OUTPUT_DIR, 'val_images.npy'), val_ids)

    # 2. Lưu val_captions.json (Thay tên captions_val_split.json cho đồng bộ)
    # File này Evaluator sẽ dùng để tính BLEU/CIDEr
    with open(os.path.join(OUTPUT_DIR, 'val_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(val_refs_dict, f, ensure_ascii=False)

    # 3. Lưu val_labels.npy (Bạn hỏi cái này)
    # Dùng để tính CrossEntropyLoss trên tập Val (nếu muốn check overfitting)
    val_labels = encode_captions(val_captions_flat, vocab, MAX_LEN)
    np.save(os.path.join(OUTPUT_DIR, 'val_labels.npy'), val_labels)


    print(f"File output tại: {OUTPUT_DIR}")
    print("- train_images.npy, train_labels.npy, train_captions.json, train_ref.pkl")
    print("- val_images.npy, val_labels.npy, val_captions.json")


if __name__ == '__main__':
    main()