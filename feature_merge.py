import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

# --- CẤU HÌNH ---
TEST_IMAGE_DIR = 'data/test_custom/images/'  # Thư mục chứa ảnh test gốc
MAPPING_FILE = 'test_extraction_list.json'  # File sinh ra từ bước 1

# Output lưu chung vào thư mục features chính
OUTPUT_FC_DIR = 'data/features/fc/'
OUTPUT_ATT_DIR = 'data/features/att/'


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.resnet(images)
        # Spatial features (Attention)
        att_feat = features.permute(0, 2, 3, 1)  # (Batch, 7, 7, 2048)
        # Global features (FC)
        fc_feat = self.avgpool(features).squeeze()  # (Batch, 2048)
        return fc_feat, att_feat


def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    if not os.path.exists(OUTPUT_FC_DIR): os.makedirs(OUTPUT_FC_DIR)
    if not os.path.exists(OUTPUT_ATT_DIR): os.makedirs(OUTPUT_ATT_DIR)

    # 2. Load danh sách cần xử lý
    if not os.path.exists(MAPPING_FILE):
        print("LỖI: Không tìm thấy file 'test_extraction_list.json'. Hãy chạy prepare_data.py trước!")
        return

    with open(MAPPING_FILE, 'r') as f:
        target_files = json.load(f)  # Dạng {'abc.jpg': '1000123'}

    print(f"Số lượng ảnh test cần trích xuất và gộp: {len(target_files)}")

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # 4. Model
    model = EncoderCNN().to(device)
    model.eval()

    # 5. Loop xử lý
    # Duyệt qua danh sách mapping
    for filename, new_id in tqdm(target_files.items()):

        # Đường dẫn ảnh gốc
        img_path = os.path.join(TEST_IMAGE_DIR, filename)

        # Kiểm tra file output đã tồn tại chưa (theo ID mới)
        fc_out_path = os.path.join(OUTPUT_FC_DIR, f"{new_id}.npy")
        att_out_path = os.path.join(OUTPUT_ATT_DIR, f"{new_id}.npz")

        if os.path.exists(fc_out_path) and os.path.exists(att_out_path):
            continue

        if not os.path.exists(img_path):
            # Thử fix lỗi đuôi file nếu cần (jpg vs jpeg)
            if filename.endswith('.jpg'):
                img_path = img_path.replace('.jpg', '.jpeg')

            if not os.path.exists(img_path):
                print(f"Warning: Không tìm thấy ảnh {filename}")
                continue

        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                fc, att = model(image)

            # LƯU VỚI ID MỚI (QUAN TRỌNG)

            # 1. Save FC
            np.save(fc_out_path, fc.cpu().numpy())

            # 2. Save ATT
            att_cpu = att.cpu().squeeze(0)  # (7, 7, 2048)
            att_reshaped = att_cpu.reshape(-1, 2048).numpy()  # (49, 2048)
            np.savez_compressed(att_out_path, feat=att_reshaped)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Extraction & Merge completed!")


if __name__ == '__main__':
    main()