import torch
import torch.nn as nn


class ReinforceLoss(nn.Module):
    """
    Dùng cho pha GAN.
    Chiến thuật: Expand -> Flatten -> Cast to Long.
    Khắc phục triệt để lỗi Dimension và Data Type.
    """

    def __init__(self):
        super(ReinforceLoss, self).__init__()

    def forward(self, reward, baseline, log_probs, seq):
        """
        - reward: [Batch, 1]
        - log_probs: [Batch, Seq, Vocab]
        - seq: [Batch, Seq] (Index của từ)
        """
        # 1. Tính Advantage [Batch, 1]
        if baseline is None:
            advantage = reward
        else:
            advantage = reward - baseline

        # Đảm bảo advantage là [Batch, 1]
        if advantage.dim() == 1:
            advantage = advantage.unsqueeze(1)

        # --- BƯỚC QUAN TRỌNG: EXPAND TRƯỚC KHI FLATTEN ---
        batch_size = seq.size(0)
        seq_len = seq.size(1)

        # Expand advantage: [Batch, 1] -> [Batch, Seq_Len]
        advantage_expanded = advantage.expand(batch_size, seq_len)

        # 2. FLATTEN TOÀN BỘ VỀ 1 CỘT DỌC [Batch * Seq_Len, 1]

        # A. Flatten Advantage
        adv_flat = advantage_expanded.reshape(-1, 1)

        seq_flat = seq.reshape(-1, 1).long()

        # C. Flatten Mask (Mask thì cần Float để nhân)
        mask_flat = (seq_flat > 0).float()

        # D. Flatten Log Probs
        # [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
        if log_probs.dim() > 2:
            log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        else:
            log_probs_flat = log_probs

        # 3. GATHER (Lấy xác suất của từ đã chọn)
        # log_probs_flat: [N, Vocab] (Float)
        # seq_flat: [N, 1] (Long - Đã fix)
        log_probs_selected = log_probs_flat.gather(1, seq_flat)

        # 4. TÍNH LOSS
        # Loss = - log_prob * advantage * mask
        loss = - log_probs_selected * adv_flat * mask_flat

        # Tính trung bình
        sum_mask = torch.sum(mask_flat)
        if sum_mask > 0:
            loss = torch.sum(loss) / sum_mask
        else:
            loss = torch.sum(loss)

        return loss


class SequenceLoss(nn.Module):
    """
    Dùng cho pha Pre-train (MLE).
    """

    def __init__(self):
        super(SequenceLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, mask=None):
        if mask is None:
            mask = (target > 0).float()

        min_len = min(input.size(1), target.size(1))
        input = input[:, :min_len, :]
        target = target[:, :min_len]
        mask = mask[:, :min_len]

        input_flat = input.reshape(-1, input.size(2))
        target_flat = target.reshape(-1).long()
        mask_flat = mask.reshape(-1)

        loss = self.loss_fn(input_flat, target_flat)
        loss = torch.sum(loss * mask_flat) / torch.sum(mask_flat)
        return loss