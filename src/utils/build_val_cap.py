import json
import os
import random

# --- CẤU HÌNH ---
# Đường dẫn file caption gốc (file gộp ban đầu)
INPUT_JSON = 'data/raw/captions.json'

# Đường dẫn file output
OUTPUT_VAL_JSON = 'data/raw/captions_val_split.json'

# Số lượng ảnh dùng để Validate (chấm điểm trong lúc train)
# Bạn có thể để 10% tổng dữ liệu hoặc con số cố định (ví dụ 500)
VAL_SIZE = 500


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Lỗi: Không tìm thấy file {INPUT_JSON}")
        return

    print(f"Đang đọc dữ liệu từ {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Gom nhóm Caption theo Image ID
    # data format gốc: [{'image_id': '...', 'caption': '...'}, ...]
    img_to_caps = {}
    for item in data:
        img_id = str(item['image_id'])
        caption = item['caption']

        if img_id not in img_to_caps:
            img_to_caps[img_id] = []
        img_to_caps[img_id].append(caption)

    print(f"Tổng số ảnh trong dữ liệu: {len(img_to_caps)}")

    # 2. Tách tập Validation
    # Lấy danh sách ID ảnh và xáo trộn
    all_img_ids = list(img_to_caps.keys())
    random.seed(42)  # Cố định seed để lần sau chạy lại vẫn ra tập val như cũ
    random.shuffle(all_img_ids)

    # Lấy 500 ảnh đầu tiên làm Validation
    val_img_ids = all_img_ids[:VAL_SIZE]

    # 3. Tạo Dictionary theo chuẩn Evaluator
    val_refs = {}
    for img_id in val_img_ids:
        val_refs[img_id] = img_to_caps[img_id]

    # 4. Lưu file
    print(f"Đang lưu {len(val_refs)} ảnh vào {OUTPUT_VAL_JSON}...")
    with open(OUTPUT_VAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(val_refs, f, ensure_ascii=False, indent=2)

    print("HOÀN TẤT! File json đã sẵn sàng.")


if __name__ == '__main__':
    main()