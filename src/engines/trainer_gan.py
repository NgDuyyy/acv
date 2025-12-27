import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
import os
from src.modules.loss import ReinforceLoss
from src.data_loader.dataset import CaptionDataset
from src.engines.evaluator import Evaluator


class TrainerGAN:
    def __init__(self, model, discriminator, config, device):
        """
        Đã sửa __init__ để khớp với train.py:
        nhận vào (generator, discriminator, args, device)
        """
        self.model = model
        self.discriminator = discriminator  # Giữ lại dù thuật toán SCST cơ bản có thể chưa dùng tới
        self.config = config
        self.device = device

        # --- 1. TỰ KHỞI TẠO DATALOADER (Giống TrainerMLE) ---
        print(">> TrainerGAN: Initializing DataLoaders...")
        self.train_dataset = CaptionDataset('train', config)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        # --- 2. TỰ KHỞI TẠO EVALUATOR (Giống TrainerMLE) ---
        # Evaluator sẽ tự load tập Val
        self.evaluator = Evaluator(model, config, device)

        # Lấy val_loader từ evaluator để dùng cho vòng lặp evaluate sau này
        self.val_loader = self.evaluator.dataloader

        # --- 3. SETUP OPTIMIZER & LOSS ---
        # Learning rate cho GAN thường thấp hơn MLE (5e-5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

        # Loss Reinforce
        self.rl_criterion = ReinforceLoss()

        self.epochs = config.gan_epochs

        # Đường dẫn lưu model
        self.save_path = os.path.join(config.checkpoint_dir, 'model_best_gan.pth')

    def train_step(self, images, captions, mask):
        """
        SCST Training Step
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Greedy Search (Baseline)
        self.model.eval()
        with torch.no_grad():
            greedy_seq, _ = self.model.sample(images, max_len=captions.size(1))
        self.model.train()

        # 2. Sample Search (Exploration)
        sample_seq, sample_log_probs = self.model.sample(images, max_len=captions.size(1), sample=True)

        # 3. Tính Reward (CIDEr Score)
        reward, greedy_score = self.evaluator.cider_scorer.compute_reward(
            sample_seq,
            greedy_seq,
            images.size(0)
        )

        reward = torch.from_numpy(reward).float().to(self.device).view(-1, 1)
        baseline = torch.from_numpy(greedy_score).float().to(self.device).view(-1, 1)

        # 4. Tính Loss
        loss = self.rl_criterion(reward, baseline, sample_log_probs, sample_seq)

        # 5. Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()

        return loss.item(), torch.mean(reward).item()

    def train(self):
        print("=== BẮT ĐẦU PHA 3: ADVERSARIAL TRAINING ===")

        # Load best CIDEr từ trước (nếu có) hoặc bắt đầu từ 0
        best_cider = 0.0

        for epoch in range(self.epochs):
            start = time.time()
            total_loss = 0
            total_reward = 0

            # Training Loop
            for i, (images, captions, mask) in enumerate(self.train_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                mask = mask.to(self.device)

                loss, reward = self.train_step(images, captions, mask)
                total_loss += loss
                total_reward += reward

                # Log mỗi 50 batch
                if i % 50 == 0 and i > 0:
                    print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss:.4f} | Reward: {reward:.4f}")

            # --- VALIDATION LOOP (FULL METRICS) ---
            print(f">> Đang chạy đánh giá GAN Epoch {epoch + 1}...")

            # Evaluate trả về full metrics dict
            scores = self.evaluator.evaluate(self.model, self.val_loader)

            cider = scores.get('CIDEr', 0.0)

            # In bảng kết quả
            print("-" * 30)
            print(f"Epoch {epoch + 1} Summary | Time: {time.time() - start:.0f}s")
            print(f"Avg Train Reward: {total_reward / len(self.train_loader):.4f}")
            print("-" * 30)

            # In tất cả các chỉ số (BLEU, METEOR, ROUGE, CIDEr)
            for metric, score in scores.items():
                print(f"{metric}: {score:.4f}")
            print("-" * 30)

            # Save Best Model dựa trên CIDEr
            if cider > best_cider:
                best_cider = cider
                torch.save(self.model.state_dict(), self.save_path)
                print(f">> ⭐️ Saved Best GAN Model (CIDEr: {best_cider:.4f})!")

            print("\n")