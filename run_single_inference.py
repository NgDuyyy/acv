from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torchvision.transforms as transforms
from PIL import Image

from config import CHECKPOINT_DIR, DATA_NAME_BASE, DEVICE, WORD_MAP_PATH
from models import Decoder, Encoder


def _ensure_pickled_classes() -> None:
    """Một số checkpoint cũ lưu Encoder/Decoder dưới module __main__."""
    main_module = sys.modules.get('__main__')
    if main_module is None:
        return
    for name, cls in (('Encoder', Encoder), ('Decoder', Decoder)):
        if getattr(main_module, name, None) is None:
            setattr(main_module, name, cls)


def _load_models(checkpoint_path: Path) -> tuple[Encoder, Decoder, Dict[str, int]]:
    _ensure_pickled_classes()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder: Encoder = checkpoint['encoder'].to(DEVICE).eval()
    decoder: Decoder = checkpoint['decoder'].to(DEVICE).eval()
    word_map: Dict[str, int] = checkpoint.get('word_map')  # có thể None
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

    raise FileNotFoundError(
        "Không tìm thấy word map. Vui lòng truyền --word-map hoặc chạy prepare_data.py để sinh WORDMAP_*.json"
    )


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
    max_steps: int = 50,
) -> List[int]:
    import torch.nn.functional as F  # tránh import toàn cục không cần thiết

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


def _tokens_to_sentence(tokens: List[int], rev_word_map: Dict[int, str], special_ids: set[int]) -> str:
    words = [rev_word_map.get(tok, '<unk>') for tok in tokens if tok not in special_ids]
    return ' '.join(words)


def main() -> None:
    parser = argparse.ArgumentParser(description='Caption một ảnh duy nhất bằng checkpoint hiện có.')
    parser.add_argument('--img', '-i', type=Path, required=True, help='Đường dẫn ảnh cần caption')
    parser.add_argument(
        '--checkpoint',
        '-c',
        type=Path,
        default=CHECKPOINT_DIR / f'BEST_checkpoint_{DATA_NAME_BASE}.pth.tar',
        help='Đường dẫn checkpoint (.pth.tar)',
    )
    parser.add_argument(
        '--word-map',
        '-wm',
        type=Path,
        default=None,
        help='Đường dẫn word_map JSON (tự tìm nếu bỏ trống)',
    )
    parser.add_argument('--beam-size', '-b', type=int, default=5, help='Beam size khi decode')
    parser.add_argument('--max-steps', type=int, default=50, help='Số bước decode tối đa')

    args = parser.parse_args()

    encoder, decoder, word_map_from_ckpt = _load_models(args.checkpoint)
    word_map = _load_word_map(args.word_map, word_map_from_ckpt)
    rev_word_map = {v: k for k, v in word_map.items()}
    special_ids = {word_map.get('<start>'), word_map.get('<end>'), word_map.get('<pad>')}

    image_tensor = _preprocess(args.img)
    token_seq = _beam_search(encoder, decoder, image_tensor, word_map, args.beam_size, args.max_steps)
    caption = _tokens_to_sentence(token_seq, rev_word_map, special_ids)

    print(f"Ảnh: {args.img}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Caption dự đoán: {caption}")


if __name__ == '__main__':
    main()
