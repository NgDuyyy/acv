import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()

    def forward(self, input, target):
        """
        input: [Batch, Seq_Len, Vocab] (Logits từ Generator)
        target: [Batch, Seq_Len] (Indices - Labels)
        """
        # 1. Cắt ngắn target nếu input bị ngắn hơn (do logic Teacher Forcing lệch 1 bước)
        # Input: 19 bước, Target: 20 bước -> Cắt Target còn 19
        if target.size(1) > input.size(1):
            target = target[:, :input.size(1)]

        # 2. Tạo Mask (Bỏ qua padding = 0)
        # QUAN TRỌNG: .bool() để sửa lỗi "expected BoolTensor"
        mask = (target > 0).bool()

        # 3. Tính Log Softmax (Biến đổi Logits thành Log Probability)
        # Giúp tính toán ổn định hơn so với việc log(softmax)
        log_probs = F.log_softmax(input, dim=2)

        # 4. Gom (Gather) log_prob của đúng nhãn target
        # target unsqueeze: [Batch, Seq_Len, 1]
        target_unsqueezed = target.unsqueeze(2)

        # Lấy giá trị log_prob tại đúng index của target
        # gathered: [Batch, Seq_Len, 1] -> squeeze -> [Batch, Seq_Len]
        gathered_log_probs = log_probs.gather(2, target_unsqueezed).squeeze(2)

        # 5. Chọn lọc (Masked Select)
        # Chỉ lấy loss ở những chỗ không phải padding (mask = True)
        masked_loss = gathered_log_probs.masked_select(mask)

        # 6. Tính Trung bình (Negative Log Likelihood)
        loss = -masked_loss.mean()

        return loss


# Giữ nguyên class ReinforceLoss bên dưới (nếu có)
class ReinforceLoss(nn.Module):
    def __init__(self):
        super(ReinforceLoss, self).__init__()

    def forward(self, reward, baseline, log_probs, seqs):
        # reward: [Batch] or [Batch, Seq_Len]
        # log_probs: [Batch, Seq_Len]

        # Tính advantage
        reward = reward.view(-1, 1)  # [Batch, 1]
        baseline = baseline.view(-1, 1)
        advantage = reward - baseline  # [Batch, 1]

        # Mask padding (seqs > 0)
        mask = (seqs > 0).float()

        # Policy Gradient Loss: - (Reward - Baseline) * Log_Prob
        # Expand advantage to [Batch, Seq_Len]
        advantage = advantage.expand_as(log_probs)

        loss = - advantage * log_probs * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss