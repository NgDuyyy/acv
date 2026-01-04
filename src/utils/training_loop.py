"""Training and validation loops shared by the orchestration scripts."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

from config import DEVICE
from .metrics import compute_cider
from .training_utils import AverageMeter, clip_gradient


def train_one_epoch(
    train_loader,
    encoder,
    decoder,
    criterion: nn.Module,
    encoder_optimizer,
    decoder_optimizer,
    epoch: int,
    grad_clip: float,
) -> float:
    decoder.train()
    encoder.train()
    losses = AverageMeter()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        caps = caps.to(DEVICE)
        caplens = caplens.to(DEVICE)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, _ = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        losses.update(loss.item(), sum(decode_lengths))

        if i % 100 == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(val_loader, encoder, decoder, criterion, word_map) -> Tuple[float, float]:
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()
    references = []
    hypotheses = []
    references_tokens: List[List[List[str]]] = []
    hypotheses_tokens: List[List[str]] = []
    rev_word_map = {idx: word for word, idx in word_map.items()}

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            caps = caps.to(DEVICE)
            caplens = caplens.to(DEVICE)
            allcaps = allcaps.to(DEVICE)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]

            scores_packed = pack_padded_sequence(
                scores, decode_lengths, batch_first=True
            ).data
            targets_packed = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = criterion(scores_packed, targets_packed)
            losses.update(loss.item(), sum(decode_lengths))

            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].cpu().tolist()
                img_captions = list(
                    map(
                        lambda c: [
                            w
                            for w in c
                            if w
                            not in {
                                word_map['<start>'],
                                word_map['<pad>'],
                                word_map['<end>'],
                            }
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)
                references_tokens.append(
                    [
                        [rev_word_map[w] for w in caption if w in rev_word_map]
                        for caption in img_captions
                    ]
                )

            _, preds = torch.max(scores, dim=2)
            preds = preds.cpu().tolist()
            temp_preds = []
            temp_tokens = []
            for j, _ in enumerate(preds):
                pred_clean = preds[j][: decode_lengths[j]]
                pred_clean = [
                    w
                    for w in pred_clean
                    if w
                    not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}
                ]
                temp_preds.append(pred_clean)
                temp_tokens.append([rev_word_map[w] for w in pred_clean if w in rev_word_map])
            hypotheses.extend(temp_preds)
            hypotheses_tokens.extend(temp_tokens)

            if i % 100 == 0:
                print(
                    f"Validation: [{i}/{len(val_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})"
                )

    bleu4 = corpus_bleu(references, hypotheses)
    cider = compute_cider(references_tokens, hypotheses_tokens)
    print(f"\n * VAL LOSS - {losses.avg:.3f}, BLEU-4 - {bleu4:.4f}, CIDEr - {cider:.4f}\n")
    return cider, losses.avg
