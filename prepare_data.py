import json
import numpy as np
import os
import collections
import random
import pickle  # <--- Bổ sung thư viện pickle để lưu file ref

# --- CẤU HÌNH ---
RAW_DIR = 'data/raw'
TEST_DIR = 'data/test_custom'
OUTPUT_PROCESSED_DIR = 'data/processed'
os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

CAPTIONS_FILE = os.path.join(RAW_DIR, 'captions.json')
TEST_CAPTIONS_FILE = os.path.join(TEST_DIR, 'test.json')

# File json gộp sẽ được lưu tại đây để kiểm tra nếu cần
MERGED_JSON_FILE = os.path.join(RAW_DIR, 'captions_merged.json')

ID_OFFSET = 1000000
MAX_LEN = 17
MIN_WORD_FREQ = 1


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def process_coco_format(coco_data):
    """
    Hàm xử lý format COCO (gồm 'images' và 'annotations')
    chuyển thành List phẳng: [{'image_id':..., 'caption':..., 'file_name':...}]
    """
    id_to_filename = {}
    for img in coco_data['images']:
        id_to_filename[img['id']] = img['filename']

    flattened_data = []
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']

        file_name = id_to_filename.get(image_id)

        if file_name:
            flattened_data.append({
                'image_id': str(image_id),
                'caption': caption,
                'file_name': file_name
            })

    return flattened_data


def main():
    print("--- 1. LOAD DỮ LIỆU ---")

    # LOAD TRAIN
    train_data = load_json(CAPTIONS_FILE)
    if isinstance(train_data, dict) and 'images' in train_data:
        print("-> Phát hiện Train data là format COCO, đang xử lý...")
        train_data = process_coco_format(train_data)

    print(f"Số lượng Train gốc (đã phẳng hóa): {len(train_data)}")

    # LOAD TEST
    test_data_raw = load_json(TEST_CAPTIONS_FILE)
    print("-> Đang xử lý format COCO cho tập Test...")
    test_data_flat = process_coco_format(test_data_raw)

    print(f"Số lượng Test gốc (đã phẳng hóa): {len(test_data_flat)}")

    # --- 2. XỬ LÝ 50% TEST ---
    split_idx = len(test_data_flat) // 2
    test_to_add = test_data_flat[:split_idx]

    print(f"-> Lấy {len(test_to_add)} ảnh từ Test gộp vào Train.")

    extraction_mapping = {}
    processed_test_data = []

    for item in test_to_add:
        old_id = item['image_id']
        file_name = item['file_name']

        # TẠO ID MỚI
        new_id_int = int(old_id) + ID_OFFSET
        new_id_str = str(new_id_int)

        extraction_mapping[file_name] = new_id_str

        new_item = {
            'image_id': new_id_str,
            'caption': item['caption'],
            'file_name': file_name
        }
        processed_test_data.append(new_item)

    save_json(extraction_mapping, 'test_extraction_list.json')
    print("-> Đã tạo file 'test_extraction_list.json' (Map filename -> new_id)")

    # --- 3. GỘP DỮ LIỆU ---
    final_data = train_data + processed_test_data
    print(f"-> Tổng mẫu train mới: {len(final_data)}")

    # <--- BỔ SUNG: LƯU FILE CAPTIONS MERGED --->
    save_json(final_data, MERGED_JSON_FILE)
    print(f"-> Đã lưu file gộp tại: {MERGED_JSON_FILE}")

    # --- 4. TẠO VOCAB ---
    print("--- 2. XÂY DỰNG VOCAB ---")
    words = []
    for item in final_data:
        cap = item['caption'].lower().replace('.', ' .').replace(',', ' ,')
        tokens = cap.split()
        words.extend(tokens)

    counter = collections.Counter(words)
    vocab = [w for w, n in counter.items() if n >= MIN_WORD_FREQ]
    vocab.sort()

    word_to_idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    for i, word in enumerate(vocab):
        word_to_idx[word] = i + 4

    print(f"-> Kích thước Vocab: {len(word_to_idx)}")
    save_json(word_to_idx, os.path.join(OUTPUT_PROCESSED_DIR, 'vocab.json'))

    # --- 5. TẠO FILE .NPY ---
    print("--- 3. TẠO FILE TRAIN NPY ---")
    image_ids = []
    labels = []

    for item in final_data:
        image_ids.append(item['image_id'])

        cap = item['caption'].lower().replace('.', ' .').replace(',', ' ,')
        tokens = cap.split()

        label = [word_to_idx['<start>']]
        for w in tokens:
            label.append(word_to_idx.get(w, word_to_idx['<unk>']))
        label.append(word_to_idx['<end>'])

        if len(label) > MAX_LEN:
            label = label[:MAX_LEN]
            label[-1] = word_to_idx['<end>']
        else:
            label += [word_to_idx['<pad>']] * (MAX_LEN - len(label))

        labels.append(label)

    np.save(os.path.join(OUTPUT_PROCESSED_DIR, 'train_images.npy'), np.array(image_ids))
    np.save(os.path.join(OUTPUT_PROCESSED_DIR, 'train_labels.npy'), np.array(labels))
    print("-> Đã lưu train_images.npy và train_labels.npy")

    # --- 6. TẠO FILE REFERENCE PKL (CHO VALIDATION/GAN) ---
    print("--- 4. TẠO FILE TRAIN_REF.PKL ---")
    refs = {}
    # Gom nhóm caption theo ID ảnh (vì một ảnh train có thể có nhiều caption)
    for item in final_data:
        img_id = item['image_id']
        caption = item['caption']

        if img_id not in refs:
            refs[img_id] = []
        refs[img_id].append(caption)

    pkl_path = os.path.join(OUTPUT_PROCESSED_DIR, 'train_ref.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(refs, f)

    print(f"-> Đã lưu train_ref.pkl (chứa {len(refs)} ảnh unique)")
    print("--- HOÀN TẤT ---")


if __name__ == '__main__':
    main()