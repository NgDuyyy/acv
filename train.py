import argparse
import os
import torch
import numpy as np
import random
import json

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.engines.trainer_mle import TrainerMLE
from src.engines.trainer_gan import TrainerGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning Training Script')

    # --- Data Paths ---
    parser.add_argument('--input_fc_dir', type=str, default='data/processed/train_fc_feats.npy',
                        help='Path to FC features')
    parser.add_argument('--input_att_dir', type=str, default='data/processed/train_att_feats.npy',
                        help='Path to Att features')
    parser.add_argument('--input_label_h5', type=str, default='data/processed/train_labels.npy', help='Path to labels')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocab.json', help='Path to vocab json')

    # --- Model Hyperparams ---
    parser.add_argument('--rnn_size', type=int, default=512, help='Size of LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in LSTM')
    parser.add_argument('--input_encoding_size', type=int, default=512, help='Word embedding size')
    parser.add_argument('--fc_feat_size', type=int, default=2048, help='ResNet FC feature size')
    parser.add_argument('--att_feat_size', type=int, default=2048, help='ResNet Att feature size')
    parser.add_argument('--att_hid_size', type=int, default=512, help='Attention hidden size')

    # --- Discriminator Hyperparams ---
    parser.add_argument('--d_rnn_size', type=int, default=512, help='Hidden size for Discriminator LSTM')
    parser.add_argument('--disc_input_size', type=int, default=2048, help='Input size for Discriminator')

    # --- Training Configuration ---
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--lambda_val', type=float, default=0.5, help='Balance between Reward GAN and Cider')

    # --- Epochs Control ---
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Epochs for Phase 1 (MLE)')
    parser.add_argument('--pretrain_discriminator_epochs', type=int, default=5, help='Epochs for Phase 2 (Disc)')
    parser.add_argument('--gan_epochs', type=int, default=40, help='Epochs for Phase 3 (Adversarial)')

    # --- Flags ---
    parser.add_argument('--skip_mle', action='store_true',
                        help='Skip Phase 1 if you already have a pretrained generator')
    parser.add_argument('--resume_gan', action='store_true', help='Resume GAN training from checkpoint')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset & Model will run on: {device}")

    # Create Checkpoint Dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # --- SỬA ĐOẠN NÀY: TÍNH VOCAB SIZE TỪ DỮ LIỆU THỰC TẾ ---
    print(f">> Checking max ID in label data: {args.input_label_h5}...")

    # 1. Load file labels lên để xem index lớn nhất là bao nhiêu
    try:
        train_labels = np.load(args.input_label_h5)
        max_id = int(np.max(train_labels))
        print(f"   Max Label ID found: {max_id}")

        # 2. Set vocab_size an toàn = max_id + 1
        args.vocab_size = max_id + 1

    except Exception as e:
        print(f"❌ Error loading labels to determine vocab size: {e}")
        # Fallback nếu không load được file npy (ít khi xảy ra)
        print("   Fallback to vocab.json len...")
        with open(args.vocab_path, 'r') as f:
            vocab = json.load(f)
        args.vocab_size = len(vocab) + 1

    print(f">> Final Vocab Size set to: {args.vocab_size}")
    # --------------------------------------------------------

    # --- Init Models ---
    print(">> Initializing Models...")
    generator = Generator(args).to(device)
    discriminator = Discriminator(args).to(device)

    # --- PHASE 1: MLE TRAINING (Supervised Learning) ---
    if not args.skip_mle:
        print("\n========================================")
        print(" PHASE 1: Maximum Likelihood Estimation (MLE)")
        print("========================================")
        trainer_mle = TrainerMLE(generator, args, device)
        trainer_mle.train()
    else:
        print(">> Skipped Phase 1 (MLE). Loading best MLE model...")
        # Load model đã train để chuẩn bị cho phase sau
        mle_path = os.path.join(args.checkpoint_dir, 'generator_best_mle.pth')
        if os.path.exists(mle_path):
            generator.load_state_dict(torch.load(mle_path))
            print("loaded generator_best_mle.pth")
        else:
            print("Warning: Cannot find MLE checkpoint to load!")

    # --- PHASE 2 & 3: GAN TRAINING (Adversarial Learning) ---
    print("\n========================================")
    print(" PHASE 2 & 3: Adversarial Training (GAN)")
    print("========================================")

    trainer_gan = TrainerGAN(generator, discriminator, args, device)
    trainer_gan.train()

    print("\n ALL TRAINING FINISHED!")

    # --- THÊM ĐOẠN NÀY ĐỂ VẼ BIỂU ĐỒ TỰ ĐỘNG ---
    try:
        print(">> Generating training charts...")
        import visualize
        # Hack một chút để truyền arguments vào nếu cần, hoặc gọi hàm main của visualize
        # Cách đơn giản nhất là chạy nó như một script con
        os.system(f"python visualize.py --checkpoint_dir {args.checkpoint_dir}")
    except Exception as e:
        print(f"Error plotting charts: {e}")


if __name__ == '__main__':
    main()