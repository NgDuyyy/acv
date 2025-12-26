import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from src.modules.loss import ReinforceLoss


class TrainerGAN:
    def __init__(self, model, train_loader, val_loader, evaluator, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.config = config
        self.device = device

        # Optimizer riêng cho GAN (thường learning rate rất nhỏ)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

        # Loss function: ReinforceLoss
        self.rl_criterion = ReinforceLoss()

        self.epochs = config.get('gan_epochs', 10)
        self.save_path = 'checkpoints/model_best_gan.pth'

    def train_step(self, images, captions, mask):
        """
        SCST Training Step
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Greedy Search (Baseline)
        # Model tự sinh câu mà không có ngẫu nhiên (dùng max probability)
        self.model.eval()
        with torch.no_grad():
            greedy_seq, _ = self.model.sample(images, max_len=captions.size(1))
        self.model.train()

        # 2. Sample Search (Exploration)
        # Model sinh câu có tính ngẫu nhiên (dùng multinomial sampling)
        sample_seq, sample_log_probs = self.model.sample(images, max_len=captions.size(1), sample=True)

        # 3. Tính Reward (Điểm thưởng)
        # Dùng Cider Scorer để chấm điểm cho cả câu Greedy và câu Sample
        # Reward ở đây chính là điểm CIDEr
        reward, greedy_score = self.evaluator.cider_scorer.compute_reward(
            sample_seq,
            greedy_seq,
            images.size(0)  # Batch size
        )

        # Chuyển reward sang tensor
        reward = torch.from_numpy(reward).float().to(self.device).view(-1, 1)  # [Batch, 1]
        baseline = torch.from_numpy(greedy_score).float().to(self.device).view(-1, 1)  # [Batch, 1]

        # 4. Tính Loss (REINFORCE)
        # Chúng ta muốn: Tăng xác suất của các câu có Reward cao hơn Baseline
        # rl_criterion đã được fix lỗi dimension ở các bước trước
        loss = self.rl_criterion(reward, baseline, sample_log_probs, sample_seq)

        # 5. Backprop
        loss.backward()

        # Gradient Clipping để tránh nổ gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        self.optimizer.step()

        return loss.item(), torch.mean(reward).item()

    def train(self):
        print("=== BẮT ĐẦU PHA 3: ADVERSARIAL TRAINING ===")
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

            # --- VALIDATION LOOP (ĐÃ SỬA ĐỂ IN ĐẦY ĐỦ METRICS) ---
            print(f">> Đang chạy đánh giá GAN Epoch {epoch + 1}...")

            # Hàm evaluate trả về dict chứa {BLEU-1, ..., CIDEr}
            scores = self.evaluator.evaluate(self.model, self.val_loader)

            cider = scores.get('CIDEr', 0.0)

            # In bảng kết quả đẹp như MLE
            print("-" * 30)
            print(f"Epoch {epoch + 1} Summary | Time: {time.time() - start:.0f}s")
            print(f"Avg Train Reward: {total_reward / len(self.train_loader):.4f}")
            print("-" * 30)
            # Loop qua tất cả metrics để in
            for metric, score in scores.items():
                print(f"{metric}: {score:.4f}")
            print("-" * 30)

            # Save Best Model dựa trên CIDEr
            if cider > best_cider:
                best_cider = cider
                torch.save(self.model.state_dict(), self.save_path)
                print(f">> Saved Best GAN Model (CIDEr: {best_cider:.4f})!")

            print("\n")