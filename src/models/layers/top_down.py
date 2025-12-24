import torch
import torch.nn as nn
import torch.nn.functional as F
# Import file attention.py cùng thư mục
from .attention import Attention


class TopDown(nn.Module):
    def __init__(self, args):
        super(TopDown, self).__init__()
        self.args = args
        self.drop_prob = 0.5

        # --- Layer 1: Attention LSTM ---
        # Input: [Previous Hidden State, Mean-Pooled Image Feature, Word Embedding]
        att_lstm_input_size = args.input_encoding_size + args.rnn_size * 2
        self.att_lstm = nn.LSTMCell(att_lstm_input_size, args.rnn_size)

        # --- Layer 2: Language LSTM ---
        # Input: [Attention Context, Hidden State của Attention LSTM]
        lang_lstm_input_size = args.rnn_size * 2
        self.lang_lstm = nn.LSTMCell(lang_lstm_input_size, args.rnn_size)

        # Module Attention
        self.attention = Attention(args)

    def forward(self, embed, fc_feats, att_feats1, att_feats2, att_masks, state):
        """
        State gồm:
        - state[0]: [h_att, h_lang] (hidden states)
        - state[1]: [c_att, c_lang] (cell states)
        """
        # Lấy hidden state của Language LSTM bước trước (h_lang_prev)
        prev_h = state[0][1]

        # --- BƯỚC 1: ATTENTION LSTM ---
        # Input gộp: h_lang_prev + fc_feats (ảnh tổng quát) + embed (từ hiện tại)
        att_lstm_input = torch.cat([prev_h, fc_feats, embed], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # --- BƯỚC 2: ATTENTION MECHANISM ---
        # Dùng h_att vừa tính để "soi" att_feats (ảnh chi tiết)
        att = self.attention(h_att, att_feats1, att_feats2, att_masks)

        # --- BƯỚC 3: LANGUAGE LSTM ---
        # Input gộp: Context vector (att) + h_att
        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # Dropout và cập nhật state
        output = F.dropout(h_lang, self.drop_prob, self.training)
        new_state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, new_state