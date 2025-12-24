import sys
import os
import pickle

from pyciderevalcap.ciderD.ciderD import CiderD


class Cider:
    def __init__(self, args=None):
        self.cider = CiderD(df='corpus')

        ref_path = 'data/processed/train_ref.pkl'

        if not os.path.exists(ref_path):
            ref_path = 'data/train_ref.pkl'

        print(f"Loading Cider references from {ref_path}...")
        with open(ref_path, 'rb') as fid:
            self.references = pickle.load(fid)

    def get_scores(self, seqs, image_ids):
        """
        seqs: Tensor [Batch, Len] (Token IDs)
        image_ids: List hoặc Tensor [Batch] (ID ảnh để tìm caption gốc)
        """
        captions = self._get_captions(seqs)

        # format dữ liệu đầu vào cho CiderD
        res = [{'image_id': i, 'caption': [caption]}
               for i, caption in enumerate(captions)]

        gts = {i: self.references[str(img_id)] for i, img_id in enumerate(image_ids)}
        _, scores = self.cider.compute_score(gts, res)

        return scores

    def _get_captions(self, seqs):
        return [self._get_caption(seq) for seq in seqs]

    def _get_caption(self, seq):
        words = []
        for word_idx in seq:
            if word_idx == 0: break
            words.append(str(word_idx.item()))

        return ' '.join(words)