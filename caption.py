"""Run beam-search inference on a single image and visualise the result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from config import CHECKPOINT_DIR, DATA_NAME_BASE, DEVICE, WORD_MAP_PATH
from models import Decoder, Encoder

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=5):
    """Sinh caption sử dụng Beam Search."""
    k = beam_size
    vocab_size = len(word_map)

    # 1. Đọc và xử lý ảnh
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.transpose(2, 0, 1) / 255.
    img = torch.FloatTensor(img).to(DEVICE)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(img)

    # 2. Encode
    image = image.unsqueeze(0)
    encoder_out = encoder(image)
    encoder_dim = encoder_out.size(3)
    encoder_out = encoder_out.view(1, -1, encoder_dim).expand(k, -1, encoder_dim)

    # 3. Decoding Setup
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(DEVICE)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(DEVICE)
    
    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # 4. Decoding Loop
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
            
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    elif len(seqs) > 0:
        seq = seqs[0].tolist()
    else:
        return []

    return seq


def main():
    parser = argparse.ArgumentParser(description='Image Captioning Inference')
    parser.add_argument('--img', '-i', help='path to image', required=True)
    parser.add_argument(
        '--model',
        '-m',
        type=Path,
        default=CHECKPOINT_DIR / f'BEST_checkpoint_{DATA_NAME_BASE}.pth.tar',
        help='path to model checkpoint',
    )
    parser.add_argument(
        '--word_map',
        '-wm',
        type=Path,
        default=WORD_MAP_PATH,
        help='path to word map JSON',
    )
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=DEVICE, weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(DEVICE)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(DEVICE)
    encoder.eval()

    # Load word map
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Encode, decode with beam search
    seq = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    words = [rev_word_map.get(ind, '') for ind in seq if ind not in {word_map['<start>'], word_map['<pad>']}]
    caption = ' '.join(words)
    
    print('Predicted caption: {}'.format(caption))
    
    # Visualize
    plt.figure(figsize=(8, 8))
    img = Image.open(args.img)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Caption: {caption}")
    plt.show()

if __name__ == '__main__':
    main()