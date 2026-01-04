"""Infer captions for an entire split and export GT/PRED pairs to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, DATA_NAME_BASE, DEVICE, TEST_DIR, WORD_MAP_PATH
from src.models import Decoder, Encoder

DEFAULT_JSON = TEST_DIR / 'test_data.json'
DEFAULT_IMAGES_DIR = TEST_DIR / 'images'
DEFAULT_OUTPUT = PROJECT_ROOT / 'result' / 'log_history' / 'test_predictions.csv'


def _ensure_pickled_classes() -> None:
    main_module = sys.modules.get('__main__')
    if main_module is None:
        return
    for name, cls in (('Encoder', Encoder), ('Decoder', Decoder)):
        if getattr(main_module, name, None) is None:
            setattr(main_module, name, cls)


def _load_models(checkpoint_path: Path) -> tuple[Encoder, Decoder, Dict[str, int] | None]:
    _ensure_pickled_classes()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder: Encoder = checkpoint['encoder'].to(DEVICE).eval()
    decoder: Decoder = checkpoint['decoder'].to(DEVICE).eval()
    word_map: Dict[str, int] | None = checkpoint.get('word_map')
    return encoder, decoder, word_map


def _load_word_map(path: Path | None, fallback: Dict[str, int] | None) -> Dict[str, int]:
    if fallback is not None:
        return fallback

    def _candidates() -> Iterable[Path]:
        seen: set[Path] = set()
        for candidate in filter(None, [path, WORD_MAP_PATH]):
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
        for candidate in sorted((PROJECT_ROOT / 'data').rglob('WORDMAP_*.json')):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    for candidate in _candidates():
        if candidate.exists():
            with candidate.open('r', encoding='utf-8') as handle:
                return json.load(handle)
    raise FileNotFoundError('Không tìm thấy word map. Hãy truyền --word-map hoặc chạy src/scripts/prepare_data.py.')


def _preprocess(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(DEVICE)


def _beam_search(
    encoder: Encoder,
    decoder: Decoder,
    image_tensor: torch.Tensor,
    word_map: Dict[str, int],
    beam_size: int,
    max_steps: int,
) -> List[int]:
    import torch.nn.functional as F

    k = beam_size
    vocab_size = len(word_map)

    encoder_out = encoder(image_tensor)
    encoder_dim = encoder_out.size(3)
    encoder_out = encoder_out.view(1, -1, encoder_dim).expand(k, -1, encoder_dim)

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(DEVICE)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(DEVICE)

    complete_seqs: List[List[int]] = []
    complete_scores: List[torch.Tensor] = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        attention_weighted_encoding, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        inputs = torch.cat([embeddings, attention_weighted_encoding], dim=1)
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

        incomplete_inds = [idx for idx, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > max_steps:
            break
        step += 1

    if complete_scores:
        best_idx = complete_scores.index(max(complete_scores))
        return complete_seqs[best_idx]
    if len(seqs) > 0:
        return seqs[0].tolist()
    return []


def _tokens_to_sentence(tokens: Sequence[int], rev_word_map: Dict[int, str], special_ids: set[int]) -> str:
    return ' '.join(
        rev_word_map.get(tok, '<unk>')
        for tok in tokens
        if tok not in special_ids
    )


@dataclass
class Sample:
    filename: str
    image_path: Path
    captions: List[str]


def _load_samples(json_path: Path, images_dir: Path) -> List[Sample]:
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {json_path}")
    with json_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    id_to_filename = {img['id']: img['filename'] for img in data.get('images', [])}
    filename_to_caps: Dict[str, List[str]] = {fn: [] for fn in id_to_filename.values()}
    for ann in data.get('annotations', []):
        filename = id_to_filename.get(ann['image_id'])
        if filename is None:
            continue
        filename_to_caps.setdefault(filename, []).append(ann.get('caption', ''))

    samples: List[Sample] = []
    for img in data.get('images', []):
        filename = img['filename']
        path = images_dir / filename
        samples.append(
            Sample(
                filename=filename,
                image_path=path,
                captions=filename_to_caps.get(filename, []),
            )
        )
    return samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Infer toàn bộ tập và xuất CSV GT/PRED.')
    parser.add_argument('--json', type=Path, default=DEFAULT_JSON, help='Đường dẫn test_data.json.')
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_IMAGES_DIR, help='Thư mục chứa ảnh tương ứng.')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DIR / f'BEST_checkpoint_{DATA_NAME_BASE}.pth.tar', help='Checkpoint dùng để inference.')
    parser.add_argument('--word-map', type=Path, default=None, help='Đường dẫn WORDMAP_*.json (tự tìm nếu bỏ trống).')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size cho decoder.')
    parser.add_argument('--max-steps', type=int, default=50, help='Số bước decode tối đa.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='CSV đầu ra (2 cột GT, PRED).')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples = _load_samples(args.json, args.images_dir)
    if not samples:
        raise RuntimeError('Không tìm thấy ảnh nào trong JSON.')

    encoder, decoder, word_map_from_ckpt = _load_models(args.checkpoint)
    word_map = _load_word_map(args.word_map, word_map_from_ckpt)
    rev_word_map = {v: k for k, v in word_map.items()}
    special_ids = {word_map.get('<start>'), word_map.get('<end>'), word_map.get('<pad>')}

    rows: List[tuple[str, str]] = []
    for sample in tqdm(samples, desc='Inferencing', unit='img'):
        if not sample.image_path.exists():
            print(f"[CẢNH BÁO] Thiếu file ảnh: {sample.image_path}")
            continue
        image_tensor = _preprocess(sample.image_path)
        token_seq = _beam_search(
            encoder=encoder,
            decoder=decoder,
            image_tensor=image_tensor,
            word_map=word_map,
            beam_size=args.beam_size,
            max_steps=args.max_steps,
        )
        prediction = _tokens_to_sentence(token_seq, rev_word_map, special_ids)
        gt = ' || '.join(sample.captions) if sample.captions else '(Không có GT)'
        rows.append((gt, prediction))

    if not rows:
        raise RuntimeError('Không tạo được dòng kết quả nào. Kiểm tra lại dữ liệu đầu vào.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['GT', 'PRED'])
        writer.writerows(rows)
    print(f"Đã ghi {len(rows)} dòng vào {args.output}")


if __name__ == '__main__':
    main()
