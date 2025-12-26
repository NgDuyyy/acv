import os
import json
import random
import numpy as np
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm

# ================= CẤU HÌNH =================
INPUT_JSON = 'data/raw/captions.json'
OUTPUT_DIR = 'data/processed'
MIN_WORD_FREQ = 5
VAL_RATIO = 0.2
SEED = 42
MAX_LEN = 50


# ============================================

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
    train_ids_unique = all_img_ids[:split_idx]  # ID duy nhất
    val_ids_unique = all_img_ids[split_idx:]  # ID duy nhất

    print(f">> Train set (Unique Images): {len(train_ids_unique)} ảnh")
    print(f">> Val set (Unique Images):   {len(val_ids_unique)} ảnh")

    # ================= XỬ LÝ TẬP TRAIN =================
    print("\n=== ĐANG XỬ LÝ TẬP TRAIN ===")

    train_captions_flat = []
    train_img_ids_expanded = []  # <--- KEY FIX: Danh sách ID lặp lại theo caption
    train_refs_dict = {}

    # Vòng lặp quan trọng: Bung dữ liệu ra (Flatten)
    for img_id in train_ids_unique:
        caps = grouped_data[img_id]
        train_refs_dict[img_id] = caps

        for c in caps:
            train_captions_flat.append(c)
            train_img_ids_expanded.append(img_id)  # Caption có bao nhiêu, ID có bấy nhiêu

    print(f">> Số lượng mẫu train sau khi mở rộng (1 ảnh N caption): {len(train_img_ids_expanded)}")

    # 1. Xây Vocab
    vocab = build_vocab(train_captions_flat, MIN_WORD_FREQ)
    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    # 2. Lưu train_images.npy (LƯU BẢN EXPANDED)
    # Lúc này len(train_images) == len(train_labels) -> Dataset không bị lỗi nữa
    np.save(os.path.join(OUTPUT_DIR, 'train_images.npy'), train_img_ids_expanded)

    # 3. Mã hóa train labels
    train_labels = encode_captions(train_captions_flat, vocab, MAX_LEN)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)

    # 4. Lưu train_ref.pkl
    with open(os.path.join(OUTPUT_DIR, 'train_ref.pkl'), 'wb') as f:
        pickle.dump(train_refs_dict, f)

    # 5. Lưu train_captions.json
    with open(os.path.join(OUTPUT_DIR, 'train_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(train_refs_dict, f, ensure_ascii=False)

    # ================= XỬ LÝ TẬP VAL =================
    print("\n=== ĐANG XỬ LÝ TẬP VAL ===")

    val_refs_dict = {}
    for img_id in val_ids_unique:
        caps = grouped_data[img_id]
        val_refs_dict[img_id] = caps

    # 1. Lưu val_images.npy (LƯU BẢN UNIQUE)
    # Tập Val chủ yếu dùng cho Evaluator, chỉ cần duyệt qua ảnh 1 lần là đủ
    np.save(os.path.join(OUTPUT_DIR, 'val_images.npy'), val_ids_unique)

    # 2. Lưu val_captions.json
    with open(os.path.join(OUTPUT_DIR, 'val_captions.json'), 'w', encoding='utf-8') as f:
        json.dump(val_refs_dict, f, ensure_ascii=False)

    # (Không tạo val_labels.npy vì Evaluator dùng ImageDataset, ko phải CaptionDataset)

    print("\n✅ XONG TOÀN BỘ! ĐÃ FIX LỖI LỆCH DATA.")
    print(f"File output tại: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()