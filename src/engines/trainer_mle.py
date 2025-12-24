import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import csv
import os
from src.data_loader.dataset import CaptionDataset
from src.modules.loss import SequenceLoss
from .evaluator import Evaluator


class TrainerMLE:
    def __init__(self, generator, args, device):
        self.generator = generator
        self.args = args
        self.device = device

        self.dataset = CaptionDataset('train', args)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.learning_rate)
        self.criterion = SequenceLoss()

        self.log_file = os.path.join(args.checkpoint_dir, 'log_mle.csv')
        self.best_cider = 0.0
        self.evaluator = Evaluator(generator, args, device)

        # --- SỬA 1: Header CSV đầy đủ ---
        self.metrics_header = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Epoch, Time, Loss + Các metrics
            header = ['Epoch', 'Time', 'Train_Loss'] + self.metrics_header
            writer.writerow(header)

    def train(self):
        print("=== BẮT ĐẦU PHA 1: PRE-TRAIN GENERATOR (MLE) ===")

        for epoch in range(self.args.pretrain_epochs):
            start = time.time()
            self.generator.train()
            total_loss = 0

            for i, data in enumerate(self.dataloader):
                fc_feats = data['fc_feats'].to(self.device)
                att_feats = data['att_feats'].to(self.device)
                att_masks = data['att_masks'].to(self.device)
                labels = data['labels'].to(self.device)

                # --- [FIX SIÊU CẤP] ÉP DỮ LIỆU THEO MODEL ---

                # 1. Lấy kích thước thật sự của Embedding trong Generator
                # Đây là "Sự thật tuyệt đối", không phụ thuộc vào args hay json nào cả
                real_vocab_limit = self.generator.embedding.num_embeddings

                # 2. Debug in ra màn hình ở batch đầu tiên để kiểm tra
                if i == 0 and epoch == 0:
                    max_label_in_batch = labels.max().item()
                    print(f"\n[DEBUG] Epoch {epoch + 1} Check:")
                    print(f" - Model Embedding Size: {real_vocab_limit}")
                    print(f" - Max Label ID in Data: {max_label_in_batch}")
                    if max_label_in_batch >= real_vocab_limit:
                        print(f" => PHÁT HIỆN LỖI: Dữ liệu ({max_label_in_batch}) lớn hơn Model ({real_vocab_limit})!")

                # 3. Xử lý triệt để: CLAMP dữ liệu về mức an toàn
                # Bất kỳ số nào >= real_vocab_limit sẽ bị ép về (real_vocab_limit - 1)
                labels[labels >= real_vocab_limit] = real_vocab_limit - 1

                # 4. Xử lý số âm (nếu có) -> về 0 (Padding)
                labels[labels < 0] = 0

                # ---------------------------------------------

                self.optimizer.zero_grad()
                probs = self.generator(fc_feats, att_feats, att_masks, seqs=labels, mode='forward')
                targets = labels[:, 1:]
                loss = self.criterion(probs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            # --- ĐÁNH GIÁ ---
            # Hàm này trả về dict: {'BLEU-1': 0.7, 'CIDEr': 1.2 ...}
            metrics = self.evaluator.evaluate()

            # --- SỬA 2: In toàn bộ Metrics ra màn hình ---
            epoch_time = time.time() - start
            print(f"\nEpoch {epoch + 1} | Time: {epoch_time:.0f}s | Loss: {avg_loss:.4f}")
            print("-" * 30)
            log_row = [epoch + 1, int(epoch_time), f"{avg_loss:.4f}"]

            # Duyệt qua danh sách header để lấy đúng giá trị từ dict metrics
            for key in self.metrics_header:
                val = metrics.get(key, 0.0)  # Nếu thiếu (ví dụ ko có SPICE) thì để 0
                print(f"{key}: {val:.4f}")
                log_row.append(f"{val:.4f}")
            print("-" * 30 + "\n")

            # --- SỬA 3: Ghi vào CSV ---
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            # Checkpoint dựa trên CIDEr
            val_cider = metrics.get('CIDEr', 0.0)
            if val_cider > self.best_cider:
                self.best_cider = val_cider
                torch.save(self.generator.state_dict(),
                           os.path.join(self.args.checkpoint_dir, 'generator_best_mle.pth'))
                print(">> Đã lưu model MLE tốt nhất!")