import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models

from src.models.generator import Generator
from src.engine.evaluator import Evaluator  # Dùng để tính điểm trên tập Test set


# --- CẤU HÌNH RESNET ĐỂ TRÍCH XUẤT ĐẶC TRƯNG ---
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet101(pretrained=True)
        # Bỏ 2 lớp cuối (AvgPool và FC) để lấy feature map 7x7
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)
        self.eval()  # Freeze

    def forward(self, images):
        with torch.no_grad():
            # Output: [Batch, 2048, 7, 7]
            features = self.resnet(images)

            # 1. Att Feats: [Batch, 49, 2048]
            att_feats = features.permute(0, 2, 3, 1)  # [Batch, 7, 7, 2048]
            att_feats = att_feats.view(att_feats.size(0), -1, att_feats.size(3))  # Flatten không gian

            # 2. FC Feats: [Batch, 2048] (Average Pooling)
            fc_feats = features.mean(3).mean(2)

            return fc_feats, att_feats


def parse_args():
    parser = argparse.ArgumentParser()

    # Mode: 'metric' (chấm điểm tập test) hoặc 'inference' (chạy ảnh custom)
    parser.add_argument('--mode', type=str, default='inference', choices=['metric', 'inference'])

    # Paths
    parser.add_argument('--image_folder', type=str, default='data/test_custom/images', help='Folder ảnh cần caption')
    parser.add_argument('--model_path', type=str, default='checkpoints/generator_best_gan.pth',
                        help='Path model đã train')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocab.json')
    parser.add_argument('--output_json', type=str, default='predictions.json')

    # Model Params (Phải khớp với lúc train)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)  # Inference từng ảnh

    args = parser.parse_args()
    return args


# --- HÀM GIẢI MÃ ID -> TEXT ---
def decode_sequence(vocab, seq):
    id_to_word = {v: k for k, v in vocab.items()}
    words = []
    for idx in seq:
        idx = idx.item()
        if idx == 0: break  # End token
        word = id_to_word.get(str(idx), id_to_word.get(idx, '<UNK>'))
        words.append(word)
    return ' '.join(words)


def run_inference(args, device):
    print(f">> Đang chạy chế độ INFERENCE trên folder: {args.image_folder}")

    # 1. Load Vocab
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    # 2. Load Generator
    generator = Generator(args).to(device)
    if os.path.exists(args.model_path):
        generator.load_state_dict(torch.load(args.model_path, map_location=device))
        print(">> Model loaded successfully!")
    else:
        print("Lỗi: Không tìm thấy file model checkpoint.")
        return

    generator.eval()

    # 3. Load Feature Extractor (ResNet101)
    extractor = FeatureExtractor().to(device)

    # Transform ảnh chuẩn input ResNet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Duyệt qua folder ảnh
    results = {}
    image_files = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print("------------------------------------------------")
    for img_name in image_files:
        img_path = os.path.join(args.image_folder, img_name)

        # Load ảnh
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 256, 256]

        # Extract features
        fc_feats, att_feats = extractor(img_tensor)
        att_masks = None  # Không cần mask vì batch=1 ko có padding

        # Generate Caption
        with torch.no_grad():
            # Mode sample (Greedy)
            seq, _ = generator(fc_feats, att_feats, att_masks, mode='sample')

        caption = decode_sequence(vocab, seq[0])

        print(f"{img_name}: {caption}")
        results[img_name] = caption

    # 5. Lưu kết quả
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n>> Đã lưu kết quả vào {args.output_json}")


def run_metric_eval(args, device):
    # Hàm này dùng class Evaluator cũ để tính điểm trên tập Test (đã extract feature)
    print(">> Đang chạy chế độ METRIC EVALUATION (BLEU/CIDEr)...")

    generator = Generator(args).to(device)
    generator.load_state_dict(torch.load(args.model_path))

    # Override lại batch_size để chạy nhanh hơn
    args.batch_size = 32
    evaluator = Evaluator(generator, args, device)
    metrics = evaluator.evaluate()

    print("\nKẾT QUẢ ĐÁNH GIÁ:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'inference':
        run_inference(args, device)
    else:
        run_metric_eval(args, device)


if __name__ == '__main__':
    main()