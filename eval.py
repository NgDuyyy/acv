"""Evaluate captioning metrics (BLEU, METEOR, ROUGE, CIDEr) trên toàn bộ tập."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_NAME_BASE, DEVICE, WORD_MAP_PATH, LOG_HISTORY_DIR
from models import Decoder, Encoder
from utils import CaptionDataset


def evaluate_metrics(
    *,
    split: str = 'VAL',
    beam_size: int = 5,
    checkpoint_path: Path | None = None,
    word_map_path: Path | None = None,
    log_csv_path: Path | None = None,
    loss_value: float | None = None,
) -> Dict[str, float]:
    """Chạy inference (Beam Search) và tính/cập nhật các số đo chất lượng caption."""
    
    # 1. Load Model và Word Map
    checkpoint_path = Path(
        checkpoint_path
        if checkpoint_path is not None
        else CHECKPOINT_DIR / f'BEST_checkpoint_{DATA_NAME_BASE}.pth.tar'
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {checkpoint_path}")

    print(f"Đang load model tốt nhất: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    decoder = checkpoint['decoder']
    decoder = decoder.to(DEVICE)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(DEVICE)
    encoder.eval()

    word_map_path = Path(word_map_path or WORD_MAP_PATH)
    if not word_map_path.exists():
        raise FileNotFoundError(f"Không tìm thấy word map: {word_map_path}")
    with word_map_path.open('r', encoding='utf-8') as handle:
        word_map = json.load(handle)
    rev_word_map = {idx: word for word, idx in word_map.items()}
    vocab_size = len(word_map)

    # 2. DataLoader (Batch size = 1 cho Beam Search)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    loader = DataLoader(
        CaptionDataset(DATA_NAME_BASE, split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    references = []  # dạng chỉ số (cho BLEU)
    hypotheses = []
    references_tokens: List[List[List[str]]] = []
    hypotheses_tokens: List[List[str]] = []

    print(f"Bắt đầu đánh giá tập {split} với Beam Size = {beam_size}...")
    start_time = time.perf_counter()
    
    # 3. Inference Loop
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm.tqdm(loader)):

        image = image.to(DEVICE)

        # Encode
        encoder_out = encoder(image) 
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim) 
        num_pixels = encoder_out.size(1)

        # Setup Beam Search
        k = beam_size
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(DEVICE)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(DEVICE)

        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # Decoding
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            context = encoder_out.mean(dim=1)
            inputs = torch.cat([embeddings, context], dim=1)
            h, c = decoder.decode_step(inputs, (h, c))
            scores = F.log_softmax(decoder.fc(h), dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = (top_k_words / vocab_size).long()
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
                
            # Cập nhật state cho beam search
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1

        # Chọn sequence có điểm số cao nhất
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        elif len(seqs) > 0:
            seq = seqs[0].tolist()
        else:
            continue

        # --- Chuẩn bị References ---
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        clean_refs = [
            [rev_word_map[w] for w in ref if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            for ref in img_captions
        ]
        clean_hyp = [
            rev_word_map[w]
            for w in seq
            if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}
        ]
        references.append(img_captions)
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        references_tokens.append(clean_refs)
        hypotheses_tokens.append(clean_hyp)

    # 4. Tính toán BLEU-4
    duration = time.perf_counter() - start_time

    smoothing = SmoothingFunction().method1
    metrics = {
        'num_samples': float(len(hypotheses_tokens)),
        'bleu1': corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'bleu2': corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'bleu3': corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing),
        'bleu4': corpus_bleu(references, hypotheses, smoothing_function=smoothing),
    }
    metrics['meteor'] = _compute_meteor(references_tokens, hypotheses_tokens)
    metrics['rouge'] = _compute_rouge_l(references_tokens, hypotheses_tokens)
    metrics['cider'] = _compute_cider(references_tokens, hypotheses_tokens)

    if log_csv_path is not None:
        _log_eval_row(
            log_csv_path,
            checkpoint_path=str(checkpoint_path),
            metrics=metrics,
            loss_value=loss_value,
        )

    print(f"Thời gian đánh giá: {duration:.2f}s")
    return metrics


def _compute_meteor(
    references_tokens: Sequence[Sequence[Sequence[str]]],
    hypotheses_tokens: Sequence[Sequence[str]],
) -> float:
    scores = []
    for refs, hyp in zip(references_tokens, hypotheses_tokens):
        best = 0.0
        for ref in refs:
            score = single_meteor_score(ref, hyp)
            if score > best:
                best = score
        scores.append(best)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token in a:
        prev = 0
        for j in range(len(b)):
            temp = dp[j + 1]
            if token == b[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp
    return dp[-1]


def _compute_rouge_l(
    references_tokens: Sequence[Sequence[Sequence[str]]],
    hypotheses_tokens: Sequence[Sequence[str]],
) -> float:
    scores = []
    for refs, hyp in zip(references_tokens, hypotheses_tokens):
        if not hyp:
            scores.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            if not ref:
                continue
            lcs = _lcs_length(hyp, ref)
            if lcs == 0:
                continue
            precision = lcs / len(hyp)
            recall = lcs / len(ref)
            if precision + recall == 0:
                continue
            score = (2 * precision * recall) / (precision + recall)
            if score > best:
                best = score
        scores.append(best)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _compute_cider(
    references_tokens: Sequence[Sequence[Sequence[str]]],
    hypotheses_tokens: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    df, num_docs = _build_document_frequency(references_tokens, max_n)
    scores = []
    for refs, hyp in zip(references_tokens, hypotheses_tokens):
        scores.append(_cider_score_for_image(refs, hyp, df, num_docs, max_n))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _build_document_frequency(
    references_tokens: Sequence[Sequence[Sequence[str]]],
    max_n: int,
) -> Tuple[Dict[Tuple[Tuple[str, ...], int], int], int]:
    df: Dict[Tuple[Tuple[str, ...], int], int] = defaultdict(int)
    num_docs = 0
    for ref_list in references_tokens:
        for ref in ref_list:
            num_docs += 1
            unique_ngrams = set()
            for n in range(1, max_n + 1):
                for ngram in _iter_ngrams(ref, n):
                    unique_ngrams.add((ngram, n))
            for ngram in unique_ngrams:
                df[ngram] += 1
    return df, max(num_docs, 1)


def _iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _tfidf_vector(
    tokens: Sequence[str],
    df: Dict[Tuple[Tuple[str, ...], int], int],
    num_docs: int,
    n: int,
) -> Tuple[Dict[Tuple[str, ...], float], float]:
    counts = Counter(_iter_ngrams(tokens, n))
    if not counts:
        return {}, 0.0
    norm_factor = float(sum(counts.values())) or 1.0
    vec: Dict[Tuple[str, ...], float] = {}
    norm_sq = 0.0
    for ngram, tf in counts.items():
        key = (ngram, n)
        df_val = df.get(key, 0)
        idf = math.log((num_docs + 1.0) / (df_val + 1.0))
        value = (tf / norm_factor) * idf
        vec[ngram] = value
        norm_sq += value * value
    return vec, math.sqrt(norm_sq)


def _cider_score_for_image(
    refs: Sequence[Sequence[str]],
    hyp: Sequence[str],
    df: Dict[Tuple[Tuple[str, ...], int], int],
    num_docs: int,
    max_n: int,
) -> float:
    if not hyp:
        return 0.0
    hyp_vecs = [_tfidf_vector(hyp, df, num_docs, n) for n in range(1, max_n + 1)]
    ref_vecs = [[_tfidf_vector(ref, df, num_docs, n) for n in range(1, max_n + 1)] for ref in refs]
    scores = []
    for ref_vec in ref_vecs:
        score_n = 0.0
        for n_idx in range(max_n):
            hyp_vec, hyp_norm = hyp_vecs[n_idx]
            ref_comp, ref_norm = ref_vec[n_idx]
            if hyp_norm == 0 or ref_norm == 0:
                continue
            dot = 0.0
            for ngram, value in hyp_vec.items():
                dot += value * ref_comp.get(ngram, 0.0)
            if hyp_norm and ref_norm:
                score_n += dot / (hyp_norm * ref_norm)
        scores.append(score_n / max_n)
    return 10.0 * (sum(scores) / len(scores)) if scores else 0.0


def _log_eval_row(
    path: Path,
    *,
    checkpoint_path: str,
    metrics: Dict[str, float],
    loss_value: float | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'checkpoint',
        'num_samples',
        'loss',
        'bleu1',
        'bleu2',
        'bleu3',
        'bleu4',
        'meteor',
        'rouge',
        'cider',
    ]
    file_exists = path.exists()
    row = {
        'checkpoint': checkpoint_path,
        'num_samples': f"{metrics['num_samples']:.0f}",
        'loss': '' if loss_value is None else f"{loss_value:.6f}",
        'bleu1': f"{metrics['bleu1']:.6f}",
        'bleu2': f"{metrics['bleu2']:.6f}",
        'bleu3': f"{metrics['bleu3']:.6f}",
        'bleu4': f"{metrics['bleu4']:.6f}",
        'meteor': f"{metrics['meteor']:.6f}",
        'rouge': f"{metrics['rouge']:.6f}",
        'cider': f"{metrics['cider']:.6f}",
    }
    with path.open('a', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate captioning metrics on a chosen split.")
    parser.add_argument(
        '--split',
        default='VAL',
        choices=[
            'TRAIN',
            'VAL',
            'TEST',
            'TEST_V2',
            'TEST_V3',
            'TEST_V4',
            'TEST_V5',
                        'TEST_V6',
              'train', 
              'val', 
              'test', 
              'test_v2', 
              'test_v3', 
              'test_v4', 
              'test_v5', 
              'test_v6',
        ],
        help='Tập dữ liệu để đánh giá (mặc định: VAL).',
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam size sử dụng cho giải mã (mặc định: 5).',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Đường dẫn checkpoint cụ thể (mặc định lấy BEST_checkpoint_* trong result/pretrained_parameters).',
    )
    parser.add_argument(
        '--word-map-path',
        type=Path,
        default=None,
        help='Đường dẫn word_map JSON (mặc định lấy từ config).',
    )
    parser.add_argument(
        '--log-csv',
        type=Path,
        default=LOG_HISTORY_DIR / 'eval_history.csv',
        help='Đường dẫn file CSV để ghi kết quả (bỏ trống để bỏ qua).',
    )
    parser.add_argument(
        '--loss',
        type=float,
        default=None,
        help='Giá trị loss (nếu có) để ghi vào CSV.',
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Không ghi CSV (kể cả khi --log-csv được chỉ định).',
    )
    parser.add_argument('--epoch', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--train-loss', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--val-loss', type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    split = args.split.upper()
    log_path = None if args.no_log else args.log_csv
    metrics = evaluate_metrics(
        split=split,
        beam_size=args.beam_size,
        checkpoint_path=args.checkpoint,
        word_map_path=args.word_map_path,
        log_csv_path=log_path,
        loss_value=args.loss,
    )
    print("-" * 50)
    print(f"Kết quả ({split}, beam={args.beam_size}):")
    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge', 'cider']:
        print(f"  {key.upper():<6}: {metrics[key]:.4f}")
    print("-" * 50)