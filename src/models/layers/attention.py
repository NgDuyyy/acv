import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Tính toán Context Vector dựa trên Hidden State của LSTM và Features ảnh.
    """
    def __init__(self, args):
        super(Attention, self).__init__()
        self.rnn_size = args.rnn_size
        self.att_hid_size = args.att_hid_size

        # Chuyển đổi feature ảnh và hidden state về cùng không gian chiều
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats1, att_feats2, att_masks=None):
        # h: hidden state của Attention LSTM [Batch, rnn_size]
        # att_feats1: Feature ảnh gốc [Batch, num_pixels, att_feat_size]
        # att_feats2: Feature ảnh đã qua layer Linear (để đỡ tính lại nhiều lần)

        # 1. Chiếu hidden state h
        att_h = self.h2att(h)  # [Batch, att_hid_size]
        att_h = att_h.unsqueeze(1).expand_as(att_feats2)  # [Batch, num_pixels, att_hid_size]

        # 2. Cộng Feature ảnh + Hidden state -> Tanh -> Linear -> Scalar
        dot = self.alpha_net(F.tanh(att_feats2 + att_h)).squeeze(2)  # [Batch, num_pixels]

        # 3. Tính Softmax để ra trọng số (probability)
        weight = F.softmax(dot, dim=1)

        # 4. Masking (nếu có padding)
        if att_masks is not None:
            weight = weight * att_masks.float()
            weight = weight / weight.sum(1, keepdim=True)  # Chuẩn hóa lại

        # 5. Tính tổng có trọng số (Weighted Sum) -> Context Vector
        # [Batch, 1, num_pixels] * [Batch, num_pixels, feat_size] -> [Batch, 1, feat_size]
        att = torch.bmm(weight.unsqueeze(1), att_feats1).squeeze(1)

        return att