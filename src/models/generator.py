# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # Import TopDown từ folder layers
# from .layers.top_down import TopDown
#
#
# class Generator(nn.Module):
#     def __init__(self, args):
#         super(Generator, self).__init__()
#         self.args = args
#
#         # Load vocab để biết kích thước
#         vocab_path = 'data/processed/vocab.json'
#         with open(vocab_path, encoding='utf-8') as fid:
#             vocab = json.load(fid)
#
#         # self.vocab_size = len(vocab) + 1  # +1 cho trường hợp index lớn nhất
#         self.vocab_size = args.vocab_size
#         self.max_length = 20  # Độ dài câu tối đa
#
#         # Các tham số mạng
#         self.num_layers = 2  # TopDown luôn có 2 tầng
#         self.rnn_size = args.rnn_size
#         self.fc_feat_size = args.fc_feat_size
#         self.att_feat_size = args.att_feat_size
#         self.att_hid_size = args.att_hid_size
#
#         self.embedding = nn.Embedding(self.vocab_size, args.input_encoding_size)
#         self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
#                                       nn.ReLU(),
#                                       nn.Dropout(0.5))
#
#         self.att_embed1 = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
#                                         nn.ReLU(),
#                                         nn.Dropout(0.5))
#         # Layer này dùng cho Attention tính toán nhanh hơn
#         self.att_embed2 = nn.Linear(self.att_feat_size, self.att_hid_size)
#
#         # 3. Core Decoder
#         self.decoder = TopDown(args)
#
#         # 4. Output Layer (Vector -> Word Probability)
#         self.output_layer = nn.Linear(self.rnn_size, self.vocab_size)
#
#         # Khởi tạo weight
#         self.init_weights()
#
#     def init_weights(self):
#         """Khởi tạo trọng số đều để train ổn định hơn"""
#         tf = 0.1
#         self.embedding.weight.data.uniform_(-tf, tf)
#         self.output_layer.bias.data.fill_(0)
#         self.output_layer.weight.data.uniform_(-tf, tf)
#
#     def _prepare_feature(self, fc_feats, att_feats):
#         """Xử lý feature ảnh trước khi đưa vào vòng lặp LSTM"""
#         # Embed FC feature
#         fc_feats = self.fc_embed(fc_feats)
#
#         # Embed Att feature (cho 2 mục đích khác nhau trong TopDown)
#         att_feats1 = self.att_embed1(att_feats)
#         att_feats2 = self.att_embed2(att_feats)
#
#         return fc_feats, att_feats1, att_feats2
#
#     def forward(self, fc_feats, att_feats, att_masks=None, seqs=None, mode='sample'):
#         """
#         fc_feats:  [Batch, 2048]
#         att_feats: [Batch, 196, 2048]
#         seqs:      [Batch, max_len] (Caption thật - dùng cho training)
#         mode:      'sample' (RL/Inference) hoặc 'forward' (MLE training)
#         """
#         # Chuẩn bị features
#         fc_feats, att_feats1, att_feats2 = self._prepare_feature(fc_feats, att_feats)
#
#         batch_size = fc_feats.size(0)
#         device = fc_feats.device
#
#         # Khởi tạo hidden state và cell state bằng 0
#         state = (
#             torch.zeros(self.num_layers, batch_size, self.rnn_size).to(device),
#             torch.zeros(self.num_layers, batch_size, self.rnn_size).to(device)
#         )
#
#         outputs = []
#
#         # Nếu train MLE (Teacher Forcing), ta lặp theo chiều dài caption thật
#         if seqs is not None and mode == 'forward':
#             max_step = seqs.size(1) - 1
#             # Input đầu tiên là <start> token (thường là 1) hoặc padding
#             # Ở đây ta giả định seqs đã có <start> ở đầu
#             embedded = self.embedding(seqs)
#         else:
#             # Nếu sampling, độ dài cố định
#             max_step = self.max_length
#             # Bắt đầu bằng <start> token (giả sử ID=1)
#             it = torch.ones(batch_size, dtype=torch.long).to(device)
#
#         probs_log = []
#
#         for t in range(max_step):
#             if seqs is not None and mode == 'forward':
#                 # Teacher Forcing: Input là từ thật tại bước t
#                 xt = embedded[:, t]
#             else:
#                 if t == 0:
#                     xt = self.embedding(it)
#                 else:
#                     # Sampling: Input là từ vừa sinh ra ở bước t-1
#                     # (Logic chọn từ sẽ nằm ở cuối vòng lặp)
#                     xt = self.embedding(it)
#
#             # --- GỌI DECODER ---
#             output, state = self.decoder(xt, fc_feats, att_feats1, att_feats2, att_masks, state)
#
#             # Tính xác suất từ tiếp theo
#             logit = self.output_layer(output)  # [Batch, Vocab]
#             prob = F.softmax(logit, dim=1)
#
#             # Lưu lại kết quả
#             outputs.append(logit)
#
#             if mode == 'sample':
#                 # Lấy mẫu (Sampling) cho bước tiếp theo
#                 # Dùng Multinomial distribution để sample cho RL
#                 w_t = torch.multinomial(prob, 1).view(-1)
#                 probs_log.append(prob.gather(1, w_t.unsqueeze(1)))  # Lưu xác suất để tính RL loss
#                 it = w_t  # Gán input cho bước sau
#
#         # Format output tùy mode
#         if mode == 'forward':
#             return torch.stack(outputs, dim=1)  # [Batch, Len, Vocab]
#         else:
#             # Trả về cả prob log để tính Policy Gradient
#             return torch.stack(outputs, dim=1), torch.stack(probs_log, dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args

        # Load vocab
        vocab_path = 'data/processed/vocab.json'
        if not os.path.exists(vocab_path):
            vocab_path = 'data/vocab.json'

        with open(vocab_path, encoding='utf-8') as fid:
            vocab = json.load(fid)

        self.vocab_size = len(vocab) + 1
        self.max_length = 20

        # Model params
        self.rnn_size = args.rnn_size
        self.fc_feat_size = args.fc_feat_size
        self.att_feat_size = args.att_feat_size
        self.input_encoding_size = args.input_encoding_size

        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.input_encoding_size)

        self.fc_embed = nn.Linear(self.fc_feat_size, self.rnn_size)
        self.att_embed = nn.Linear(self.att_feat_size, self.rnn_size)

        self.lstm_input_size = self.input_encoding_size + self.rnn_size * 2
        self.lstm = nn.LSTMCell(self.lstm_input_size, self.rnn_size)

        self.ctx2att = nn.Linear(self.rnn_size, self.rnn_size)
        self.h2att = nn.Linear(self.rnn_size, self.rnn_size)
        self.att2alpha = nn.Linear(self.rnn_size, 1)

        self.output_layer = nn.Linear(self.rnn_size, self.vocab_size)
        self.dropout = nn.Dropout(0.5)

    def attention(self, h, att_feats, att_masks):
        att_h = self.h2att(h).unsqueeze(1)
        att_ctx = self.ctx2att(att_feats)
        att = torch.tanh(att_h + att_ctx)
        alpha = self.att2alpha(att).squeeze(2)

        if att_masks is not None:
            alpha = alpha.masked_fill(att_masks == 0, -1e9)

        alpha = F.softmax(alpha, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)
        return context, alpha

    def forward(self, fc_feats, att_feats, att_masks, seqs=None, mode='forward'):
        """
        Logic quan trọng để sửa lỗi 6572 elements:
        - mode='forward': Trả về Logits [Batch, Len, Vocab] (Cho MLE)
        - mode='sample': Trả về Indices [Batch, Len] (Cho GAN Train)
        - mode='sample_greedy': Trả về Indices [Batch, Len] (Cho GAN Baseline)
        """
        batch_size = fc_feats.size(0)

        fc_feats_proj = self.fc_embed(fc_feats)
        att_feats_proj = self.att_embed(att_feats)

        h_state = torch.zeros(batch_size, self.rnn_size).to(fc_feats.device)
        c_state = torch.zeros(batch_size, self.rnn_size).to(fc_feats.device)

        outputs = []
        seq_indices = []
        seq_log_probs = []

        # Start token (giả sử ID=1)
        it = torch.LongTensor([1] * batch_size).to(fc_feats.device)

        max_step = self.max_length
        if mode == 'forward' and seqs is not None:
            max_step = seqs.size(1) - 1

        for t in range(max_step):
            if mode == 'forward' and seqs is not None:
                it = seqs[:, t]

            xt = self.embedding(it)
            context, alpha = self.attention(h_state, att_feats_proj, att_masks)

            lstm_input = torch.cat([xt, fc_feats_proj, context], dim=1)
            h_state, c_state = self.lstm(lstm_input, (h_state, c_state))

            output = self.dropout(h_state)
            logit = self.output_layer(output)

            if mode == 'forward':
                outputs.append(logit)
            elif mode == 'sample' or mode == 'sample_greedy':
                # --- ĐÂY LÀ ĐOẠN SỬA LỖI ---
                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)

                if mode == 'sample':
                    # Lấy mẫu ngẫu nhiên -> Ra INDEX (số nguyên)
                    w_t = torch.multinomial(prob, 1).detach().view(-1)
                else:
                    # Lấy max -> Ra INDEX (số nguyên)
                    w_t = torch.max(prob, 1)[1].detach().view(-1)

                # Lưu index (để Cider dùng)
                seq_indices.append(w_t)

                # Lưu log_prob (để Loss function dùng)
                log_prob_selected = log_prob.gather(1, w_t.unsqueeze(1))
                seq_log_probs.append(log_prob_selected)

                it = w_t  # Input cho bước sau là từ vừa sinh ra

        if mode == 'forward':
            return torch.stack(outputs, dim=1)
        else:
            # Trả về Indices (cho Cider) và LogProbs (cho Loss)
            return torch.stack(seq_indices, dim=1), torch.stack(seq_log_probs, dim=1).squeeze(2)