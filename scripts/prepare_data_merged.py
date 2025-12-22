import json
import os
import string

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Bạn hãy sửa lại đường dẫn trỏ đến file chứa captions thực tế của bạn
# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Bạn hãy sửa lại đường dẫn trỏ đến file chứa captions thực tế của bạn
train_path = r'data/train/train_data.json'
val_path = r'data/val/val_data.json'
output_path = r'data/LSTM/data_merged.json'

# Hàm tách từ (giữ nguyên)
def tokenize(s):
    s = s.lower()
    for c in string.punctuation:
        s = s.replace(c, '')
    return s.split()

def process_split(file_path, split_name):
    print(f"Đang xử lý {split_name}: {file_path}")
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return []
        
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    
    # Mapping annotations nếu cần (cho định dạng COCO chuẩn)
    img_to_captions = {}
    if 'annotations' in data:
        for ann in data['annotations']:
            img_id = ann['image_id']
            cap = ann['caption']
            if img_id not in img_to_captions:
                img_to_captions[img_id] = []
            img_to_captions[img_id].append(cap)
            
    processed_images = []
    # Xử lý linh hoạt: data có thể là list ảnh hoặc dict chứa key 'images'
    images_list = data['images'] if 'images' in data else data
    
    for img in images_list:
        # --- QUAN TRỌNG: XỬ LÝ ID ---
        # Ưu tiên lấy 'id' gốc, nếu không có thì lấy 'cocoid' hoặc 'imgid'
        img_id = img.get('id', img.get('cocoid', img.get('imgid')))
        
        if img_id is None:
            print(f"⚠️ Ảnh {img.get('filename', 'unknown')} không có ID! Bỏ qua.")
            continue
            
        # Tạo object ảnh mới chuẩn format
        new_img = {
            'id': img_id,
            'file_path': img.get('filename', img.get('file_path', '')),
            'split': split_name,
            'sentences': []
        }
        
        # Lấy captions
        captions = []
        if 'caption' in img:
            captions.append(img['caption'])
        elif 'captions' in img:
            captions.extend(img['captions'])
        elif img_id in img_to_captions:
            captions.extend(img_to_captions[img_id])
            
        # Tokenize
        for cap in captions:
            new_img['sentences'].append({
                'tokens': tokenize(cap),
                'raw': cap
            })
            
        if len(new_img['sentences']) > 0:
            processed_images.append(new_img)
            
    print(f"-> Đã xử lý {len(processed_images)} ảnh cho tập {split_name}")
    return processed_images

# --- MAIN ---
all_images = []
all_images.extend(process_split(train_path, 'train'))
all_images.extend(process_split(val_path, 'val'))

# Kiểm tra trùng lặp ID
seen_ids = set()
duplicates = []
for img in all_images:
    if img['id'] in seen_ids:
        duplicates.append(img['id'])
    seen_ids.add(img['id'])

if duplicates:
    print(f"⚠️ CẢNH BÁO: Phát hiện {len(duplicates)} ID trùng lặp! (Ví dụ: {duplicates[:5]})")
    print("Việc này có thể gây lỗi khi map với features.")
else:
    print(f"✅ Kiểm tra ID: Tất cả {len(seen_ids)} ID đều là duy nhất.")

# Lưu file kết quả
final_data = {'images': all_images}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
json.dump(
    final_data,
    open(output_path, 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
)
print(f"Hoàn thành! Tổng cộng {len(all_images)} ảnh. Đã lưu tại: {output_path}")
