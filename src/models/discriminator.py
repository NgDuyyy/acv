import torch
import torch.nn as nn
import json
import os


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        # 1. Load Vocab
        vocab_path = 'data/processed/vocab.json'
        if not os.path.exists(vocab_path):
            vocab_path = 'data/vocab.json'

        if os.path.exists(vocab_path):
            with open(vocab_path, encoding='utf-8') as fid:
                vocab = json.load(fid)
            self.vocab_size = len(vocab) + 1
        else:
            self.vocab_size = 3517

        # 2. Configs
        self.hidden_size = getattr(args, 'd_rnn_size', getattr(args, 'hidden_size', 512))
        self.input_size = getattr(args, 'disc_input_size', getattr(args, 'embed_size', 512))
        self.fc_feat_size = getattr(args, 'fc_feat_size', 2048)
        self.num_layers = 1

        # 3. Layers
        self.vembedding = nn.Linear(self.fc_feat_size, self.input_size)
        self.wembedding = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fc_feats, seqs):
        """
        Input Debug:
        - fc_feats mong đợi: [Batch, 2048]
        - seqs mong đợi:     [Batch, Seq_Len]
        """
        # Ép fc_feats về 2D [Batch, -1]
        if fc_feats.dim() > 2:
            fc_feats = fc_feats.view(fc_feats.size(0), -1)

        # Kiểm tra nếu seqs đang là 3D [Batch, Seq_Len, 1] -> Ép về 2D [Batch, Seq_Len]
        if seqs.dim() > 2:
            # print(f"DEBUG: Seqs đang bị 3D {seqs.shape} -> Đang ép về 2D...")
            seqs = seqs.view(seqs.size(0), -1)


        # 1. Embed Image
        # [Batch, 2048] -> [Batch, Input_Size]
        img_emb = self.vembedding(fc_feats)
        # Thêm chiều time -> [Batch, 1, Input_Size] (Dạng 3D)
        img_emb = img_emb.unsqueeze(1)

        # 2. Embed Caption
        # [Batch, Seq_Len] -> [Batch, Seq_Len, Input_Size] (Dạng 3D)
        word_emb = self.wembedding(seqs)

        if img_emb.dim() != word_emb.dim():
            print(f"!!! CRASH ALERT !!!")
            print(f"Img Shape: {img_emb.shape} (Dim: {img_emb.dim()})")
            print(f"Word Shape: {word_emb.shape} (Dim: {word_emb.dim()})")
            # Cố gắng cứu vãn lần cuối:
            if word_emb.dim() == 4:
                word_emb = word_emb.squeeze(2)

        # 3. Concatenate
        # [Batch, Seq_Len + 1, Input_Size]
        lstm_input = torch.cat([img_emb, word_emb], dim=1)

        # 4. LSTM & Output
        out, _ = self.lstm(lstm_input)
        logits = self.classifier(out)
        probs = self.sigmoid(logits)

        return probs.mean(dim=1)