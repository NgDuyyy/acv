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
            self.generator.load_state_dict(torch.load(mle_path, map_location=device))
            print(f"Loaded Pre-trained Generator from {mle_path}")
        else:
            print("CẢNH BÁO: Không tìm thấy model MLE. GAN sẽ train từ đầu (rất khó hội tụ)!")

        self.dataset = CaptionDataset('train', args)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Learning Rate cho GAN thường nhỏ hơn MLE để tránh vỡ model
        self.g_optim = optim.Adam(self.generator.parameters(), lr=args.learning_rate)
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=args.learning_rate)

        self.rl_criterion = ReinforceLoss()
        self.d_criterion = torch.nn.BCELoss()
        self.cider_scorer = Cider(args)
        self.evaluator = Evaluator(generator, args, device)

        self.log_file = os.path.join(args.checkpoint_dir, 'log_gan.csv')
        self.metrics_header = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']

        # Tạo file log nếu chưa có
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Epoch', 'Time', 'G_Loss', 'D_Loss', 'Avg_Reward'] + self.metrics_header
                writer.writerow(header)

    def pretrain_discriminator(self, epochs=1):  # Giảm xuống 1 epoch demo, thực tế nên là 5
        print("\n=== PRE-TRAINING DISCRIMINATOR ===")
        # (Giữ nguyên logic pretrain cũ, nhưng nhớ ép kiểu .long() tương tự hàm train bên dưới nếu bị lỗi)
        # Tạm thời bỏ qua ở đây vì bạn đã chạy được hàm này trước đó rồi.
        pass

    def train(self):
        # self.pretrain_discriminator() # Có thể bật lại nếu muốn train kỹ D trước

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
                with torch.no_grad():
                    # Sinh caption giả
                    fake_captions, _ = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                # Train với ảnh thật
                d_out_real = self.discriminator(fc_feats, real_captions)
                loss_d_real = self.d_criterion(d_out_real, torch.ones_like(d_out_real))

                # Train với ảnh giả
                # --- FIX LỖI TẠI ĐÂY: fake_captions.long() ---
                d_out_fake = self.discriminator(fc_feats, fake_captions.long().detach())
                loss_d_fake = self.d_criterion(d_out_fake, torch.zeros_like(d_out_fake))

                loss_d = loss_d_real + loss_d_fake

                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()
                total_d_loss += loss_d.item()

                # ===============================================================
                # 2. UPDATE GENERATOR (POLICY GRADIENT)
                # ===============================================================
                # Sample (để tính Reward)
                sample_seqs, sample_log_probs = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                # Greedy (để làm Baseline - Self Critical)
                with torch.no_grad():
                    greedy_seqs, _ = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                # Tính điểm CIDEr (Reward)
                r_cider_sample = self.cider_scorer.get_scores(sample_seqs, img_ids)
                r_cider_greedy = self.cider_scorer.get_scores(greedy_seqs, img_ids)

                # Tính điểm từ Discriminator (Reward bổ sung)
                with torch.no_grad():
                    # --- FIX LỖI TẠI ĐÂY: sample_seqs.long() ---
                    r_gan = self.discriminator(fc_feats, sample_seqs.long())

                # Tổng hợp Reward
                r_cider_sample = torch.from_numpy(r_cider_sample).float().to(self.device)
                r_cider_greedy = torch.from_numpy(r_cider_greedy).float().to(self.device)

                # Reward = lambda * GAN_Score + (1-lambda) * CIDEr_Score
                reward = self.args.lambda_val * r_gan.view(-1) + (1 - self.args.lambda_val) * r_cider_sample
                baseline = (1 - self.args.lambda_val) * r_cider_greedy

                # Policy Gradient Loss
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

            # Đánh giá cuối Epoch
            print("\n>> Đang chạy đánh giá (Evaluator)...")
            metrics = self.evaluator.evaluate()

            epoch_time = time.time() - start
            avg_g_loss = total_g_loss / len(self.dataloader)
            avg_d_loss = total_d_loss / len(self.dataloader)
            avg_reward = total_reward / len(self.dataloader)

            print(
                f"\nEpoch {epoch + 1} Done | G_Loss: {avg_g_loss:.3f} | D_Loss: {avg_d_loss:.3f} | Avg Reward: {avg_reward:.3f}")
            print("-" * 30)

            log_row = [epoch + 1, int(epoch_time), f"{avg_g_loss:.4f}", f"{avg_d_loss:.4f}", f"{avg_reward:.4f}"]
            for key in self.metrics_header:
                val = metrics.get(key, 0.0)
                print(f"{key}: {val:.4f}")
                log_row.append(f"{val:.4f}")
            print("-" * 30 + "\n")

            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow(log_row)

            # Lưu model tốt nhất theo CIDEr
            val_cider = metrics.get('CIDEr', 0.0)
            if val_cider > self.best_cider:
                self.best_cider = val_cider
                torch.save(self.generator.state_dict(),
                           os.path.join(self.args.checkpoint_dir, 'generator_best_gan.pth'))
                print(">> Saved Best GAN Model!")