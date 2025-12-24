import sys
import os
import json
import torch
from torch.utils.data import DataLoader
from src.data_loader.dataset import ImageDataset
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Evaluator:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

        # Load vocab để decode (ID -> Word)
        # Giả sử file vocab nằm ở data/processed/vocab.json
        vocab_path = os.path.join('data/processed/vocab.json')
        if not os.path.exists(vocab_path):
            vocab_path = 'data/vocab.json'  # Fallback path

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        self.id_to_word = {v: k for k, v in self.vocab.items()}

        self.dataset = ImageDataset('train', args)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        gt_path = 'data/raw/captions_val_split.json'
        if os.path.exists(gt_path):
            self.references = json.load(open(gt_path))
        else:
            print(f"CẢNH BÁO: Không tìm thấy {gt_path}. Cần tạo file này để chấm điểm!")
            self.references = {}

    def evaluate(self):
        self.model.eval()
        predictions = {}
        target_img_ids = set(self.references.keys())

        print(">> Đang chạy đánh giá trên tập Val (Sinh caption)...")

        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                fc_feats = data['fc_feats'].to(self.device)
                att_feats = data['att_feats'].to(self.device)
                att_masks = data['att_masks'].to(self.device)
                img_ids = data['image_ids']  # List các image id

                seqs, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
                sents = self.decode_sequence(seqs)

                for img_id, sent in zip(img_ids, sents):
                    str_id = str(img_id.item() if isinstance(img_id, torch.Tensor) else img_id)

                    if str_id in target_img_ids:
                        predictions[str(img_id)] = [sent]

        if not self.references:
            return {'CIDEr': 0.0, 'BLEU-4': 0.0}

        metrics = self.compute_metrics(self.references, predictions)
        return metrics

    def decode_sequence(self, seqs):
        """
        Chuyển Tensor ID thành câu văn.
        seqs: [Batch, Max_Len]
        """
        sents = []
        for row in seqs:
            words = []
            for token_id in row:
                token_id = token_id.item()
                if token_id == 0:  # Giả sử 0 là <end> hoặc padding
                    break

                # Bỏ qua token <start> nếu có (thường là 1 hoặc index cuối)
                # Tùy thuộc vào vocab của bạn

                word = self.id_to_word.get(str(token_id), '')  # json key thường là string
                if not word:
                    word = self.id_to_word.get(token_id, 'UNK')

                words.append(word)

            sents.append(' '.join(words))
        return sents

    def compute_metrics(self, ref, res):
        inter_keys = set(ref.keys()) & set(res.keys())
        ref_filtered = {k: ref[k] for k in inter_keys}
        res_filtered = {k: res[k] for k in inter_keys}

        scorers = [
            (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
            (Meteor(), "METEOR"), # Cần Java
            (Rouge(), "ROUGE-L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")   # Cần Java
        ]

        eval_result = {}
        for scorer, method in scorers:
            try:
                score, _ = scorer.compute_score(ref_filtered, res_filtered)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        eval_result[m] = s
                else:
                    eval_result[method] = score
            except Exception as e:
                print(f"Lỗi khi tính {method}: {e}")
                if isinstance(method, list):
                    for m in method: eval_result[m] = 0.0
                else:
                    eval_result[method] = 0.0

        return eval_result