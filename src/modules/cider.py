import os
import json
import pickle
import torch
from pyciderevalcap.ciderD.ciderD import CiderD
import sys
from collections import defaultdict


class Cider:
    def __init__(self, args):
        self.args = args

        # 1. Load Vocab
        vocab_path = os.path.join('data/processed/vocab.json')
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.id_to_word = {str(v): k for k, v in self.vocab.items()}

        # 2. Khởi tạo CiderD Scorer Wrapper
        self.cider_scorer = CiderD(df='corpus')

        # 3. Load Reference Data
        ref_path = os.path.join('data/processed/train_ref.pkl')
        print(f">> Cider Module: Loading references from {ref_path}...")

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy {ref_path}")

        with open(ref_path, 'rb') as f:
            self.refs = pickle.load(f)

        # 4. TÍNH TOÁN DOCUMENT FREQUENCY (FIX MẠNH TAY)
        print(">> Cider Module: Computing document frequency (IDF)...")

        # Truy cập object Scorer thật sự bên trong
        if hasattr(self.cider_scorer, 'cider_scorer'):
            internal_scorer = self.cider_scorer.cider_scorer
        else:
            internal_scorer = self.cider_scorer

        # --- FIX LỖI Ở ĐÂY: KHỞI TẠO BIẾN THIẾU ---
        # Nếu thư viện quên khởi tạo document_frequency, ta tự làm thay nó
        if not hasattr(internal_scorer, 'document_frequency'):
            # print(">> DEBUG: Tự khởi tạo document_frequency...")
            internal_scorer.document_frequency = defaultdict(float)

        # Đảm bảo ref_len cũng được khởi tạo (phòng hờ)
        if not hasattr(internal_scorer, 'ref_len'):
            internal_scorer.ref_len = None

        # Nạp dữ liệu vào Scorer thủ công
        for img_id, refs in self.refs.items():
            internal_scorer.cook_append(refs[0], refs)

        # Gọi hàm tính toán
        if hasattr(internal_scorer, 'compute_doc_freq'):
            internal_scorer.compute_doc_freq()
            print(f">> Cider Module: Document frequency computed! (Ref len: {len(self.refs)})")
        else:
            print(">> CẢNH BÁO: Không tìm thấy hàm compute_doc_freq.")

    def get_scores(self, seqs, img_ids):
        """
        Input: seqs [Batch, Max_Len], img_ids [Batch]
        Output: scores [Batch]
        """
        # 1. Giải mã Tensor ID -> Câu chữ
        res = []
        for row, img_id in zip(seqs, img_ids):
            sent = self.decode_sequence(row)
            if isinstance(img_id, torch.Tensor):
                img_id = str(img_id.item())
            else:
                img_id = str(img_id)
            res.append({'image_id': img_id, 'caption': [sent]})

        # 2. Lấy Ground Truth
        gts = {}
        for img_id in img_ids:
            if isinstance(img_id, torch.Tensor):
                img_id = str(img_id.item())
            else:
                img_id = str(img_id)

            if img_id in self.refs:
                gts[img_id] = self.refs[img_id]
            else:
                gts[img_id] = [""]

        # 3. Tính điểm
        try:
            _, scores = self.cider_scorer.compute_score(gts, res)
        except Exception as e:
            # print(f"Lỗi tính điểm CIDEr: {e}")
            return torch.zeros(len(seqs)).numpy()

        return scores

    def decode_sequence(self, row):
        words = []
        for token_id in row:
            if isinstance(token_id, torch.Tensor):
                val = token_id.item()
            else:
                val = token_id

            word = self.id_to_word.get(str(val), '<unk>')

            if word == '<end>': break
            if word == '<start>' or word == '<pad>': continue

            words.append(word)
        return ' '.join(words)