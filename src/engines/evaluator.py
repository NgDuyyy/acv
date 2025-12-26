import sys
import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from src.data_loader.dataset import ImageDataset
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


# from pycocoevalcap.spice.spice import Spice

class Evaluator:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

        # Load vocab
        vocab_path = os.path.join('data/processed/vocab.json')
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.id_to_word = {str(v): k for k, v in self.vocab.items()}

        # 1. Load Dataset Val (dùng data/processed/val_images.npy)
        try:
            print(">> Evaluator đang load tập 'val'...")
            self.dataset = ImageDataset('val', args)
        except Exception as e:
            print(f">> Lỗi load tập Val: {e}. Đang chuyển sang train (debug)...")
            full_train_set = ImageDataset('train', args)
            self.dataset = Subset(full_train_set, list(range(500)))

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


        gt_path = 'data/processed/val_captions.json'

        if os.path.exists(gt_path):
            print(f">> Đã tìm thấy file đáp án: {gt_path}")
            self.references = json.load(open(gt_path))
        else:
            print(f">> CẢNH BÁO TUYỆT ĐỐI: Không tìm thấy {gt_path}. Metrics sẽ bằng 0!")
            # Fallback thử tìm file cũ nếu lỡ tay xóa
            gt_path_old = 'data/raw/captions_val_split.json'
            if os.path.exists(gt_path_old):
                self.references = json.load(open(gt_path_old))
            else:
                self.references = {}

    def evaluate(self):
        self.model.eval()
        predictions = {}
        target_img_ids = set(self.references.keys())

        print(f">> Bắt đầu đánh giá trên {len(self.dataset)} ảnh...")
        printed_debug = False

        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                fc_feats = data['fc_feats'].to(self.device)
                att_feats = data['att_feats'].to(self.device)
                att_masks = data['att_masks'].to(self.device)
                img_ids = data['image_ids']

                seqs, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
                sents = self.decode_sequence(seqs)

                if not printed_debug:
                    print("\n[PREVIEW] === MODEL ĐANG NÓI GÌ? ===")
                    for k in range(min(5, len(sents))):
                        print(f"Sample {k}: {sents[k]}")
                    print("======================================\n")
                    printed_debug = True

                for img_id, sent in zip(img_ids, sents):
                    if isinstance(img_id, torch.Tensor):
                        str_id = str(img_id.item())
                    else:
                        str_id = str(img_id)

                    # Chỉ chấm điểm những ảnh có trong file đáp án
                    if str_id in target_img_ids:
                        predictions[str_id] = [sent]

        if not self.references or not predictions:
            print(">> LỖI: Không có dữ liệu khớp ID để chấm điểm.")
            return {'CIDEr': 0.0, 'BLEU-4': 0.0}

        metrics = self.compute_metrics(self.references, predictions)
        return metrics

    def decode_sequence(self, seqs):
        sents = []
        for row in seqs:
            words = []
            for token_id in row:
                if isinstance(token_id, torch.Tensor):
                    if token_id.numel() > 1:
                        token_id = torch.argmax(token_id).item()
                    else:
                        token_id = token_id.item()

                word = self.id_to_word.get(str(token_id), '<unk>')
                if word == '<end>': break
                if word == '<start>' or word == '<pad>': continue
                words.append(word)
            sents.append(' '.join(words))
        return sents

    def compute_metrics(self, ref, res):
        inter_keys = set(ref.keys()) & set(res.keys())
        if not inter_keys:
            return {'CIDEr': 0.0}

        ref_filtered = {k: ref[k] for k in inter_keys}
        res_filtered = {k: res[k] for k in inter_keys}

        scorers = [
            (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE-L"),
            (Cider(), "CIDEr")
        ]

        eval_result = {}
        for scorer, method in scorers:
            try:
                score, _ = scorer.compute_score(ref_filtered, res_filtered)
                if isinstance(method, list):
                    for m, s in zip(method, score): eval_result[m] = s
                else:
                    eval_result[method] = score
            except:
                if isinstance(method, list):
                    for m in method: eval_result[m] = 0.0
                else:
                    eval_result[method] = 0.0

        return eval_result