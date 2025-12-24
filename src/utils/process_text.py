import json
import os
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm


INPUT_JSON = 'data/raw/captions.json'
OUTPUT_DIR = 'data/processed/'
MAX_LEN = 20
MIN_WORD_FREQ = 5


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading data from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)


    print("Building vocabulary...")
    counts = Counter()
    for item in tqdm(data):
        words = item['caption'].lower().split()
        counts.update(words)


    # 0: padding, 1: start, 2: end, 3: unk
    vocab = {w: i + 4 for i, (w, c) in enumerate(counts.items()) if c >= MIN_WORD_FREQ}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3

    with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Saved vocab.json (Size: {len(vocab)})")


    encoded_captions = []
    image_ids = []
    refs = {}  # {image_id: [cap1, cap2, ...]} dùng cho CIDEr score

    print("Encoding captions...")
    for item in tqdm(data):
        img_id = str(item['image_id'])
        caption = item['caption'].lower().split()

        if img_id not in refs:
            refs[img_id] = []
        refs[img_id].append(' '.join(caption))

        vec = [vocab['<start>']]
        for w in caption:
            vec.append(vocab.get(w, vocab['<unk>']))
        vec.append(vocab['<end>'])

        if len(vec) > MAX_LEN:
            vec = vec[:MAX_LEN]
        else:
            vec += [vocab['<pad>']] * (MAX_LEN - len(vec))

        encoded_captions.append(vec)
        image_ids.append(img_id)

    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), np.array(encoded_captions))
    np.save(os.path.join(OUTPUT_DIR, 'train_images.npy'), np.array(image_ids))

    with open(os.path.join(OUTPUT_DIR, 'train_ref.pkl'), 'wb') as f:
        pickle.dump(refs, f)

    print(f"\nHOÀN TẤT!")
    print(f"- Labels shape: {np.array(encoded_captions).shape}")
    print(f"- Unique images: {len(refs)}")
    print(f"- Saved files to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()