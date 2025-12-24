import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import json
import os


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        vocab_path = 'data/processed/vocab.json'
        if not os.path.exists(vocab_path):
            vocab_path = 'data/vocab.json'

        with open(vocab_path, encoding='utf-8') as fid:
            vocab = json.load(fid)
        self.vocab_size = len(vocab) + 1

        # Configs
        self.hidden_size = args.d_rnn_size  # Kích thước hidden state của Discriminator
        self.input_size = args.disc_input_size  # Kích thước input word vector
        self.fc_feat_size = args.fc_feat_size

        self.vembedding = nn.Linear(self.fc_feat_size, self.input_size)
        self.wembedding = nn.Embedding(self.vocab_size, self.input_size)

        # LSTM (Standard RNN-based)
        self.dis_lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, fc_feats, seqs):
        """
        Args:
            fc_feats: [Batch, 2048] - Feature ảnh toàn cục (Global FC)
            seqs:     [Batch, Max_Len] - Caption (Token IDs)
        Return:
            score:    [Batch, 1] - Xác suất (0-1) caption là thật
        """
        device = fc_feats.device
        batch_size = fc_feats.size(0)

        # fc_feats (Batch, 2048) -> (Batch, Input_Size)
        img_emb = self.vembedding(fc_feats)
        img_emb = img_emb.unsqueeze(1)  # (Batch, 1, Input_Size)

        word_emb = self.wembedding(seqs)

        lstm_input = torch.cat([img_emb, word_emb], dim=1)
        lengths = (seqs > 0).sum(dim=1) + 1
        lengths_cpu = lengths.cpu()

        packed_input = rnn_utils.pack_padded_sequence(
            lstm_input,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False  # Tự động sắp xếp lại batch theo độ dài
        )

        # h_n shape: (num_layers, batch, hidden_size)
        _, (h_n, c_n) = self.dis_lstm(packed_input)
        final_h = h_n[-1]  # (Batch, Hidden_Size)
        res = self.output_layer(final_h)  # (Batch, 1)

        score = torch.sigmoid(res)

        return score