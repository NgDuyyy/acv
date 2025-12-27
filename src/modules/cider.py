# import os
# import json
# import pickle
# import torch
# from pyciderevalcap.ciderD.ciderD import CiderD
# import sys
# from collections import defaultdict
#
#
# class Cider:
#     def __init__(self, args):
#         self.args = args
#
#         # 1. Load Vocab
#         vocab_path = os.path.join('data/processed/vocab.json')
#         with open(vocab_path, 'r') as f:
#             self.vocab = json.load(f)
#         self.id_to_word = {str(v): k for k, v in self.vocab.items()}
#
#         # 2. Khởi tạo CiderD Scorer Wrapper
#         self.cider_scorer = CiderD(df='corpus')
#
#         # 3. Load Reference Data
#         ref_path = os.path.join('data/processed/train_ref.pkl')
#         print(f">> Cider Module: Loading references from {ref_path}...")
#
#         if not os.path.exists(ref_path):
#             raise FileNotFoundError(f"LỖI: Không tìm thấy {ref_path}")
#
#         with open(ref_path, 'rb') as f:
#             self.refs = pickle.load(f)
#
#         # 4. TÍNH TOÁN DOCUMENT FREQUENCY (FIX MẠNH TAY)
#         print(">> Cider Module: Computing document frequency (IDF)...")
#
#         # Truy cập object Scorer thật sự bên trong
#         if hasattr(self.cider_scorer, 'cider_scorer'):
#             internal_scorer = self.cider_scorer.cider_scorer
#         else:
#             internal_scorer = self.cider_scorer
#
#         # --- FIX LỖI Ở ĐÂY: KHỞI TẠO BIẾN THIẾU ---
#         # Nếu thư viện quên khởi tạo document_frequency, ta tự làm thay nó
#         if not hasattr(internal_scorer, 'document_frequency'):
#             # print(">> DEBUG: Tự khởi tạo document_frequency...")
#             internal_scorer.document_frequency = defaultdict(float)
#
#         # Đảm bảo ref_len cũng được khởi tạo (phòng hờ)
#         if not hasattr(internal_scorer, 'ref_len'):
#             internal_scorer.ref_len = None
#
#         # Nạp dữ liệu vào Scorer thủ công
#         for img_id, refs in self.refs.items():
#             internal_scorer.cook_append(refs[0], refs)
#
#         # Gọi hàm tính toán
#         if hasattr(internal_scorer, 'compute_doc_freq'):
#             internal_scorer.compute_doc_freq()
#             print(f">> Cider Module: Document frequency computed! (Ref len: {len(self.refs)})")
#         else:
#             print(">> CẢNH BÁO: Không tìm thấy hàm compute_doc_freq.")
#
#     def get_scores(self, seqs, img_ids):
#         """
#         Input: seqs [Batch, Max_Len], img_ids [Batch]
#         Output: scores [Batch]
#         """
#         # 1. Giải mã Tensor ID -> Câu chữ
#         res = []
#         for row, img_id in zip(seqs, img_ids):
#             sent = self.decode_sequence(row)
#             if isinstance(img_id, torch.Tensor):
#                 img_id = str(img_id.item())
#             else:
#                 img_id = str(img_id)
#             res.append({'image_id': img_id, 'caption': [sent]})
#
#         # 2. Lấy Ground Truth
#         gts = {}
#         for img_id in img_ids:
#             if isinstance(img_id, torch.Tensor):
#                 img_id = str(img_id.item())
#             else:
#                 img_id = str(img_id)
#
#             if img_id in self.refs:
#                 gts[img_id] = self.refs[img_id]
#             else:
#                 gts[img_id] = [""]
#
#         # 3. Tính điểm
#         try:
#             _, scores = self.cider_scorer.compute_score(gts, res)
#         except Exception as e:
#             # print(f"Lỗi tính điểm CIDEr: {e}")
#             return torch.zeros(len(seqs)).numpy()
#
#         return scores
#
#     def decode_sequence(self, row):
#         words = []
#         for token_id in row:
#             if isinstance(token_id, torch.Tensor):
#                 val = token_id.item()
#             else:
#                 val = token_id
#
#             word = self.id_to_word.get(str(val), '<unk>')
#
#             if word == '<end>': break
#             if word == '<start>' or word == '<pad>': continue
#
#             words.append(word)
#         return ' '.join(words)
#
import os
import json
import pickle
import torch
import numpy as np
import math
from collections import defaultdict


# =============================================================================
# CLASS 1: CIDER SCORER (THUẬT TOÁN GỐC ĐƯỢC VIẾT LẠI TRỰC TIẾP TẠI ĐÂY)
# =============================================================================
class NativeCiderScorer:
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.ref_len = None

    def cook_append(self, test, refs):
        """
        Lưu trữ thống kê n-gram cho Candidate (test) và References (refs)
        """
        # Nếu test (candidate) tồn tại, tính n-grams cho nó
        if test is not None:
            self.ctest.append(self.counts2ngrams(test))

        # Tính n-grams cho các câu reference
        self.crefs.append([self.counts2ngrams(ref) for ref in refs])

    def counts2ngrams(self, sentence):
        """Tách câu thành dict đếm n-gram"""
        counts = defaultdict(int)
        words = sentence.split()
        n = self.n
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
        return counts

    def compute_doc_freq(self):
        """Tính IDF cho toàn bộ tập dữ liệu"""
        for refs in self.crefs:
            # refs là list các dict n-gram của 1 ảnh
            # Gom tất cả n-gram xuất hiện trong bất kỳ ref nào của ảnh đó
            unique_ngrams = set()
            for ref_ngram_dict in refs:
                unique_ngrams.update(ref_ngram_dict.keys())

            for ngram in unique_ngrams:
                self.document_frequency[ngram] += 1

        # ref_len: log của tổng số ảnh (để dùng trong công thức IDF)
        self.ref_len = np.log(float(len(self.crefs)))

    def compute_cider(self):
        def counts2vec(cnts):
            """Chuyển dict n-gram thành vector TF-IDF"""
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]

            for (ngram, term_freq) in cnts.items():
                # Lấy số lần n-gram xuất hiện trong toàn bộ dataset
                df = self.document_frequency[ngram]
                n = len(ngram)

                # Công thức TF-IDF của CIDEr
                # tf = term_freq (clip artifacts? no, standard tf)
                # idf = log(N / df) -> ở đây ref_len chính là log(N)
                # idf = max(0, ref_len - log(df))

                # Tránh chia 0 hoặc log 0
                if df > 0:
                    idf = np.maximum(0.0, self.ref_len - np.log(df))
                else:
                    idf = 0.0

                weight = term_freq * idf
                vec[n - 1][ngram] = weight

                # Tính bình phương để lát tính norm (độ dài vector)
                norm[n - 1] += weight * weight

            # Căn bậc 2 để ra độ dài vector chuẩn
            norm = [np.sqrt(n) for n in norm]
            return vec, norm

        def sim(vec_hyp, norm_hyp, vec_ref, norm_ref):
            """Tính cosine similarity giữa 2 vector"""
            delta = float(norm_hyp * norm_ref)
            if delta == 0:
                return 0.0

            val = 0.0
            # Dot product
            for ngram, weight in vec_hyp.items():
                if ngram in vec_ref:
                    val += weight * vec_ref[ngram]
            return val / delta

        scores = []

        # Duyệt qua từng cặp (Candidate, List References)
        for test, refs in zip(self.ctest, self.crefs):
            # 1. Vector hóa Candidate
            vec_test, norm_test = counts2vec(test)

            # 2. Vector hóa từng Reference và so sánh
            score = []
            for ref in refs:
                vec_ref, norm_ref = counts2vec(ref)

                # Tính điểm cho từng n-gram (1->4) và cộng dồn
                score_avg = 0.0
                for n in range(self.n):
                    score_avg += sim(vec_test[n], norm_test[n], vec_ref[n], norm_ref[n])

                # Nhân 10 để scale điểm (theo chuẩn CIDEr) và chia trung bình n
                score_avg = score_avg * 10.0 / self.n
                score.append(score_avg)

            # Điểm cuối cùng là trung bình cộng điểm so với các ref (hoặc max)
            # Chuẩn CIDEr dùng Average
            scores.append(np.mean(score))

        return np.mean(scores), np.array(scores)


# =============================================================================
# CLASS 2: CIDER WRAPPER (INTERFACE CHO CODE TRAIN)
# =============================================================================
class Cider:
    def __init__(self, args):
        self.args = args

        # 1. Load Vocab
        vocab_path = os.path.join('data/processed/vocab.json')
        if not os.path.exists(vocab_path):
            vocab_path = 'data/vocab.json'

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.id_to_word = {str(v): k for k, v in self.vocab.items()}

        # 2. Load References
        ref_path = os.path.join('data/processed/train_ref.pkl')
        print(f">> Cider Module (Native): Loading references from {ref_path}...")
        with open(ref_path, 'rb') as f:
            raw_refs = pickle.load(f)

        # 3. Setup Native Scorer
        self.scorer = NativeCiderScorer(n=4, sigma=6.0)

        count_refs = 0
        self.refs = {}

        print(">> Cider Module (Native): Cooking refs for IDF...")
        for k, v_list in raw_refs.items():
            clean_captions = []
            for cap in v_list:
                if isinstance(cap, list) and len(cap) > 0 and isinstance(cap[0], int):
                    clean_captions.append(self.decode_from_ids(cap))
                elif isinstance(cap, str):
                    clean_captions.append(cap)
                else:
                    clean_captions.append("")

            # Nấu dữ liệu (chỉ cần refs để học IDF)
            self.scorer.cook_append(None, clean_captions)
            count_refs += 1

            # Lưu Mapping
            self.refs[str(k)] = clean_captions
            try:
                k_int = int(k)
                self.refs[k_int] = clean_captions
                self.refs[str(k_int)] = clean_captions
            except:
                pass

        # 4. Tính toán IDF Matrix
        print(f">> Cider Module (Native): Computing IDF matrix ({count_refs} samples)...")
        self.scorer.compute_doc_freq()

        # Clear bộ nhớ tạm để chuẩn bị cho Validation/Testing
        self.scorer.crefs = []
        self.scorer.ctest = []

        print(f">> Cider Module (Native): Ready. (IDF Vocab Size: {len(self.scorer.document_frequency)})")

    def decode_from_ids(self, ids_list):
        words = []
        for val in ids_list:
            word = self.id_to_word.get(str(val), '<unk>')
            if word == '<end>': break
            if word == '<start>' or word == '<pad>': continue
            words.append(word)
        return ' '.join(words)

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

    def compute_score(self, seqs, data_dict):
        img_ids = data_dict['image_ids']

        # Reset batch memory
        self.scorer.crefs = []
        self.scorer.ctest = []

        # Prepare Batch
        for row, img_id in zip(seqs, img_ids):
            cand_sent = self.decode_sequence(row)

            if isinstance(img_id, torch.Tensor):
                raw_id = img_id.item()
            else:
                raw_id = img_id

            clean_id = str(raw_id)
            if clean_id not in self.refs:
                try:
                    clean_id = int(raw_id)
                except:
                    pass

            refs_sents = self.refs.get(clean_id, [])
            if not refs_sents and len(self.refs) > 0:
                refs_sents = [""]

                # Đẩy vào Scorer
            self.scorer.cook_append(cand_sent, refs_sents)

        # Tính toán
        try:
            _, scores = self.scorer.compute_cider()
        except Exception as e:
            print(f">> CIDER NATIVE ERROR: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(len(seqs)).numpy()

        # Reset lại ngay sau khi tính
        self.scorer.crefs = []
        self.scorer.ctest = []

        if scores is None:
            return torch.zeros(len(seqs)).numpy()

        return scores