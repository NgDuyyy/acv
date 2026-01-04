from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

# Đảm bảo có thể import config/models dù file nằm trong src/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHECKPOINT_DIR,
    DATA_NAME_BASE,
    DEVICE,
    TEST_DIR,
    WORD_MAP_PATH,
)
from src.models import Decoder, Encoder

# Paste your custom image paths here if you prefer editing a constant instead of CLI args.
DEFAULT_IMAGE_PATHS: List[Path] = [
    TEST_DIR / 'images' / '000098.jpg',
    TEST_DIR / 'images' / '000445.jpg',
    TEST_DIR / 'images' / '000500.jpg',
]


def _ensure_pickled_classes() -> None:
    """Ensure Encoder/Decoder classes are discoverable when loading old checkpoints."""
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

    def _candidate_paths() -> Iterable[Path]:
        seen: set[Path] = set()
        for candidate in [path, WORD_MAP_PATH]:
            if candidate is not None:
                candidate = candidate.resolve()
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate
        data_dir = Path('data')
        if data_dir.exists():
            for candidate in sorted(data_dir.rglob('WORDMAP_*.json')):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved

    for candidate in _candidate_paths():
        if candidate.exists():
            with candidate.open('r', encoding='utf-8') as handle:
                return json.load(handle)
    raise FileNotFoundError('Không tìm thấy word map. Vui lòng cung cấp --word-map hoặc chạy src/scripts/prepare_data.py.')


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

        if len(complete_inds) > 0:
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

    if len(complete_scores) > 0:
        best_idx = complete_scores.index(max(complete_scores))
        return complete_seqs[best_idx]
    if len(seqs) > 0:
        return seqs[0].tolist()
    return []


def _tokens_to_sentence(tokens: Sequence[int], rev_word_map: Dict[int, str], special_ids: set[int]) -> str:
    words = [rev_word_map.get(tok, '<unk>') for tok in tokens if tok not in special_ids]
    return ' '.join(words)


def _load_ground_truth_captions(json_path: Path) -> Dict[str, List[str]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ground-truth: {json_path}")
    with json_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)

    id_to_filename = {img['id']: img['filename'] for img in data.get('images', [])}
    filename_to_captions: Dict[str, List[str]] = {}
    for ann in data.get('annotations', []):
        filename = id_to_filename.get(ann['image_id'])
        if filename is None:
            continue
        filename_to_captions.setdefault(filename, []).append(ann['caption'])
    return filename_to_captions


def _resolve_image_paths(cli_paths: Sequence[Path] | None) -> List[Path]:
    if cli_paths:
        return [p if p.is_absolute() else p.resolve() for p in cli_paths]
    existing_defaults = [path for path in DEFAULT_IMAGE_PATHS if path.exists()]
    if existing_defaults:
        return existing_defaults
    raise ValueError('Hãy truyền --images hoặc chỉnh DEFAULT_IMAGE_PATHS với đường dẫn hợp lệ.')


def _plot_each_sample(samples: List[dict], output_path: Path) -> None:
    if not samples:
        raise ValueError('Không có ảnh nào để vẽ.')

    output_path = output_path.resolve()
    if output_path.suffix.lower() == '.pdf':
        base_dir = output_path.parent
        prefix = output_path.stem
    else:
        base_dir = output_path
        prefix = 'sample'
    base_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        fig, ax = plt.subplots(figsize=(6, 7))
        image = Image.open(sample['image_path']).convert('RGB')
        ax.imshow(image)
        ax.axis('off')
        caption_text = f"GT: {sample['ground_truth']}\nPred: {sample['prediction']}"
        ax.text(
            0.01,
            -0.12,
            caption_text,
            transform=ax.transAxes,
            fontsize=11,
            ha='left',
            va='top',
            wrap=True,
        )

        filename = f"{prefix}_{sample['image_path'].stem}.pdf"
        fig.savefig(base_dir / filename, format='pdf', bbox_inches='tight')
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Infer captions for selected images and export a PDF summary.')
    parser.add_argument('--images', type=Path, nargs='*', default=None, help='Danh sách đường dẫn ảnh (hoặc chỉnh DEFAULT_IMAGE_PATHS).')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_DIR / f'BEST_checkpoint_{DATA_NAME_BASE}.pth.tar', help='Checkpoint dùng để inference.')
    parser.add_argument('--word-map', type=Path, default=None, help='Đường dẫn word map nếu muốn override.')
    parser.add_argument('--gt-json', type=Path, default=TEST_DIR / 'test_data.json', help='JSON chứa ground-truth captions.')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size cho decoder.')
    parser.add_argument('--max-steps', type=int, default=50, help='Số bước decode tối đa.')
    parser.add_argument('--output', type=Path, default=Path('result') / 'sample_predictions.pdf', help='File PDF đầu ra.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_paths = _resolve_image_paths(args.images)

    encoder, decoder, word_map_from_ckpt = _load_models(args.checkpoint)
    word_map = _load_word_map(args.word_map, word_map_from_ckpt)
    rev_word_map = {v: k for k, v in word_map.items()}
    special_ids = {word_map.get('<start>'), word_map.get('<end>'), word_map.get('<pad>')}

    gt_map = _load_ground_truth_captions(args.gt_json)

    samples = []
    for image_path in image_paths[:3]:
        image_tensor = _preprocess(image_path)
        token_seq = _beam_search(encoder, decoder, image_tensor, word_map, args.beam_size, args.max_steps)
        prediction = _tokens_to_sentence(token_seq, rev_word_map, special_ids)
        gt_caption_list = gt_map.get(image_path.name, [])
        ground_truth = gt_caption_list[0] if gt_caption_list else '(Không tìm thấy ground truth)'
        samples.append({'image_path': image_path, 'ground_truth': ground_truth, 'prediction': prediction})

    _plot_each_sample(samples, args.output)
    print(f"Đã lưu PDF tại: {args.output}")


if __name__ == '__main__':
    main()
