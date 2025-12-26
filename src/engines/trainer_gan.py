import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import csv
import os
import numpy as np
from src.data_loader.dataset import CaptionDataset
from src.modules.loss import ReinforceLoss
from src.modules.cider import Cider
from .evaluator import Evaluator


class TrainerGAN:
    def __init__(self, generator, discriminator, args, device):
        self.generator = generator
        self.discriminator = discriminator
        self.args = args
        self.device = device

        # Load Generator đã train xong Phase 1
        mle_path = os.path.join(args.checkpoint_dir, 'generator_best_mle.pth')
        if os.path.exists(mle_path):
            # Load weight vào CPU trước để tiết kiệm GPU
            state_dict = torch.load(mle_path, map_location='cpu')
            self.generator.load_state_dict(state_dict)
            self.generator.to(device)
            print(f"Loaded Pre-trained Generator from {mle_path}")
        else:
            print("CẢNH BÁO: Không tìm thấy model MLE.")

        self.dataset = CaptionDataset('train', args)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Learning Rate nhỏ
        self.g_optim = optim.Adam(self.generator.parameters(), lr=5e-5)
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=5e-5)

        self.rl_criterion = ReinforceLoss()
        self.d_criterion = torch.nn.BCELoss()
        self.cider_scorer = Cider(args)
        self.evaluator = Evaluator(generator, args, device)

        self.log_file = os.path.join(args.checkpoint_dir, 'log_gan.csv')
        self.metrics_header = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Epoch', 'Time', 'G_Loss', 'D_Loss', 'Avg_Reward'] + self.metrics_header
                writer.writerow(header)

    def train(self):
        print("=== BẮT ĐẦU PHA 3: ADVERSARIAL TRAINING ===")
        self.best_cider = 0.0

        for epoch in range(self.args.gan_epochs):
            start = time.time()
            self.generator.train()
            self.discriminator.train()

            total_g_loss = 0
            total_d_loss = 0
            total_reward = 0

            for i, data in enumerate(self.dataloader):
                fc_feats = data['fc_feats'].to(self.device)
                att_feats = data['att_feats'].to(self.device)
                att_masks = data['att_masks'].to(self.device)
                real_captions = data['labels'].to(self.device)
                img_ids = data['image_ids']

                # ===============================================================
                # 1. UPDATE DISCRIMINATOR
                # ===============================================================
                # Bước quan trọng: Lấy đúng Indices từ Generator
                with torch.no_grad():
                    gen_out = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                    # Logic tự động phát hiện đâu là indices (2D), đâu là probs (3D)
                    if isinstance(gen_out, tuple):
                        val1, val2 = gen_out
                        if val1.dim() == 2:  # [Batch, Len] -> Đây là Indices
                            fake_captions = val1
                        elif val2.dim() == 2:
                            fake_captions = val2
                        else:
                            # Nếu cả 2 đều 3D (hiếm gặp), lấy argmax của cái đầu
                            fake_captions = torch.argmax(val1, dim=-1)
                    else:
                        fake_captions = gen_out

                # Đảm bảo fake_captions là LongTensor và cắt gradient
                fake_captions = fake_captions.long().detach()

                # Train D với ảnh thật
                d_out_real = self.discriminator(fc_feats, real_captions)
                loss_d_real = self.d_criterion(d_out_real, torch.ones_like(d_out_real))

                # Train D với ảnh giả
                d_out_fake = self.discriminator(fc_feats, fake_captions)
                loss_d_fake = self.d_criterion(d_out_fake, torch.zeros_like(d_out_fake))

                loss_d = loss_d_real + loss_d_fake

                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()
                total_d_loss += loss_d.item()

                # ===============================================================
                # 2. UPDATE GENERATOR (POLICY GRADIENT)
                # ===============================================================
                # Lấy cả seqs (để tính reward) và log_probs (để tính gradient)
                out_sample = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                # Tự động gán đúng biến dựa trên chiều dữ liệu
                if out_sample[0].dim() == 2:
                    sample_seqs, sample_log_probs = out_sample
                else:
                    sample_log_probs, sample_seqs = out_sample

                # Baseline (Greedy)
                with torch.no_grad():
                    out_greedy = self.generator(fc_feats, att_feats, att_masks, mode='sample')
                    # Xử lý tương tự cho greedy
                    if isinstance(out_greedy, tuple):
                        greedy_seqs = out_greedy[0] if out_greedy[0].dim() == 2 else out_greedy[1]
                    else:
                        greedy_seqs = out_greedy

                # Tính Reward (CIDEr)
                r_cider_sample = self.cider_scorer.get_scores(sample_seqs, img_ids)
                r_cider_greedy = self.cider_scorer.get_scores(greedy_seqs, img_ids)

                # Tính Reward (Discriminator)
                with torch.no_grad():
                    r_gan = self.discriminator(fc_feats, sample_seqs.long())

                r_cider_sample = torch.from_numpy(r_cider_sample).float().to(self.device)
                r_cider_greedy = torch.from_numpy(r_cider_greedy).float().to(self.device)

                # Kết hợp Reward
                reward = self.args.lambda_val * r_gan.view(-1) + (1 - self.args.lambda_val) * r_cider_sample
                baseline = (1 - self.args.lambda_val) * r_cider_greedy

                loss_g = self.rl_criterion(reward, baseline, sample_log_probs, sample_seqs)

                self.g_optim.zero_grad()
                loss_g.backward()
                self.g_optim.step()

                total_g_loss += loss_g.item()
                total_reward += reward.mean().item()

                if i % 50 == 0:
                    print(
                        f"\rEpoch {epoch + 1} | Batch {i} | G_Loss: {loss_g.item():.4f} | D_Loss: {loss_d.item():.4f} | Reward: {reward.mean().item():.4f}",
                        end="")

            # Đánh giá
            print("\n>> Đang chạy đánh giá GAN...")
            metrics = self.evaluator.evaluate()

            avg_g_loss = total_g_loss / len(self.dataloader)
            avg_d_loss = total_d_loss / len(self.dataloader)
            avg_reward = total_reward / len(self.dataloader)

            print(f"Epoch {epoch + 1} Done | Reward: {avg_reward:.4f} | CIDEr: {metrics.get('CIDEr', 0):.4f}")

            # Lưu Log
            log_row = [epoch + 1, int(time.time() - start), f"{avg_g_loss:.4f}", f"{avg_d_loss:.4f}",
                       f"{avg_reward:.4f}"]
            for key in self.metrics_header:
                log_row.append(f"{metrics.get(key, 0.0):.4f}")
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow(log_row)

            if metrics.get('CIDEr', 0) > self.best_cider:
                self.best_cider = metrics['CIDEr']
                torch.save(self.generator.state_dict(),
                           os.path.join(self.args.checkpoint_dir, 'generator_best_gan.pth'))
                print(">> Saved Best GAN Model!")