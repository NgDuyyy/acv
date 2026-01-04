"""Shared metric helpers used during training and evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence, Tuple


def compute_cider(
    references_tokens: Sequence[Sequence[Sequence[str]]],
    hypotheses_tokens: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """Compute the CIDEr score for corpus-level predictions."""

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
