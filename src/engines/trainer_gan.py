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

        mle_path = os.path.join(args.checkpoint_dir, 'generator_best_mle.pth')
        if os.path.exists(mle_path):
            self.generator.load_state_dict(torch.load(mle_path, map_location=device))
            print(f"Loaded Pre-trained Generator from {mle_path}")

        self.dataset = CaptionDataset('train', args)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        self.g_optim = optim.Adam(self.generator.parameters(), lr=args.learning_rate)
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=args.learning_rate)

        self.rl_criterion = ReinforceLoss()
        self.d_criterion = torch.nn.BCELoss()
        self.cider_scorer = Cider(args)
        self.evaluator = Evaluator(generator, args, device)

        self.log_file = os.path.join(args.checkpoint_dir, 'log_gan.csv')

        self.metrics_header = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Thêm cột G_Loss, D_Loss, Reward
            header = ['Epoch', 'Time', 'G_Loss', 'D_Loss', 'Avg_Reward'] + self.metrics_header
            writer.writerow(header)

    def train(self):
        self.pretrain_discriminator()

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

                # Update D
                with torch.no_grad():
                    fake_captions, _ = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                d_out_real = self.discriminator(fc_feats, real_captions)
                loss_d_real = self.d_criterion(d_out_real, torch.ones_like(d_out_real))
                d_out_fake = self.discriminator(fc_feats, fake_captions.detach())
                loss_d_fake = self.d_criterion(d_out_fake, torch.zeros_like(d_out_fake))
                loss_d = loss_d_real + loss_d_fake

                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()
                total_d_loss += loss_d.item()

                # Update G
                sample_seqs, sample_log_probs = self.generator(fc_feats, att_feats, att_masks, mode='sample')
                with torch.no_grad():
                    greedy_seqs, _ = self.generator(fc_feats, att_feats, att_masks, mode='sample')

                r_cider_sample = self.cider_scorer.get_scores(sample_seqs, img_ids)
                r_cider_greedy = self.cider_scorer.get_scores(greedy_seqs, img_ids)
                with torch.no_grad():
                    r_gan = self.discriminator(fc_feats, sample_seqs)

                r_cider_sample = torch.from_numpy(r_cider_sample).float().to(self.device)
                r_cider_greedy = torch.from_numpy(r_cider_greedy).float().to(self.device)
                reward = self.args.lambda_val * r_gan.view(-1) + (1 - self.args.lambda_val) * r_cider_sample
                baseline = (1 - self.args.lambda_val) * r_cider_greedy

                loss_g = self.rl_criterion(reward, baseline, sample_log_probs, sample_seqs)
                self.g_optim.zero_grad()
                loss_g.backward()
                self.g_optim.step()
                total_g_loss += loss_g.item()
                total_reward += reward.mean().item()

            metrics = self.evaluator.evaluate()

            epoch_time = time.time() - start
            avg_g_loss = total_g_loss / len(self.dataloader)
            avg_d_loss = total_d_loss / len(self.dataloader)
            avg_reward = total_reward / len(self.dataloader)

            print(
                f"\nEpoch {epoch + 1} | G_Loss: {avg_g_loss:.3f} | D_Loss: {avg_d_loss:.3f} | Reward: {avg_reward:.3f}")
            print("-" * 30)

            log_row = [epoch + 1, int(epoch_time), f"{avg_g_loss:.4f}", f"{avg_d_loss:.4f}", f"{avg_reward:.4f}"]

            for key in self.metrics_header:
                val = metrics.get(key, 0.0)
                print(f"{key}: {val:.4f}")
                log_row.append(f"{val:.4f}")
            print("-" * 30 + "\n")

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            val_cider = metrics.get('CIDEr', 0.0)
            if val_cider > self.best_cider:
                self.best_cider = val_cider
                torch.save(self.generator.state_dict(),
                           os.path.join(self.args.checkpoint_dir, 'generator_best_gan.pth'))
                print(">> Saved Best GAN Model!")

    def pretrain_discriminator(self):
        # (Giữ nguyên code pretrain_discriminator tôi đã gửi ở tin nhắn trước)
        # Code này chạy độc lập, log ra file log_pretrain_d.csv nên không ảnh hưởng
        pass