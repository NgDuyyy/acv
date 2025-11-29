"""Model components for the image captioning pipeline.

This module houses the convolutional encoder built on top of a pre-trained
ResNet-101 backbone together with a lightweight LSTM decoder that consumes
global context vectors. The implementation mirrors the original behaviour but
is organized so it can be imported from the new ``models`` package.
"""

from __future__ import annotations

import torch
from torch import nn
import torchvision

from config import DEVICE, ENCODER_DIM

__all__ = ["Encoder", "Decoder"]


class Encoder(nn.Module):
    """ResNet-101 visual encoder with a configurable spatial resolution."""

    def __init__(self, encoded_image_size: int = 14) -> None:
        super().__init__()
        self.enc_image_size = encoded_image_size

        backbone = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        )
        self.resnet = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )
        self.fine_tune(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        return features.permute(0, 2, 3, 1)

    def fine_tune(self, fine_tune: bool = True) -> None:
        """Enable or freeze the deeper convolutional blocks."""

        for param in self.resnet.parameters():
            param.requires_grad = False
        for block in list(self.resnet.children())[5:]:
            for param in block.parameters():
                param.requires_grad = fine_tune


class Decoder(nn.Module):
    """LSTM decoder that conditions every time-step on global image context."""

    def __init__(
        self,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = ENCODER_DIM,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(
        self, encoder_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(
            batch_size, max(decode_lengths), self.vocab_size, device=DEVICE
        )
        global_encoding = encoder_out.mean(dim=1)

        for step in range(max(decode_lengths)):
            batch_size_t = sum(length > step for length in decode_lengths)
            context = global_encoding[:batch_size_t]
            step_input = torch.cat(
                [embeddings[:batch_size_t, step, :], context], dim=1
            )
            h, c = self.decode_step(step_input, (h[:batch_size_t], c[:batch_size_t]))
            predictions[:batch_size_t, step, :] = self.fc(self.dropout_layer(h))

        return predictions, encoded_captions, decode_lengths, sort_ind
