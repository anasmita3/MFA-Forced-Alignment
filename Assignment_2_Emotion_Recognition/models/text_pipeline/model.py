import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TextBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        hidden_size=32,
        num_classes=7,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)

        representation = output.mean(dim=1)
        representation = self.dropout(representation)

        return self.classifier(representation)


class TransformerTextModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=7,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        pad_idx=0
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention_pool = nn.Linear(d_model, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, tokens):
        padding_mask = tokens == self.pad_idx

        x = self.embedding(tokens)
        x = self.position(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        scores = self.attention_pool(x).squeeze(-1)
        scores = scores.masked_fill(padding_mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)

        representation = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return self.classifier(representation)
class UpdatedTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3,
        max_len=6
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )

        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention_pool = nn.Linear(d_model, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, tokens):
        mask = tokens != 0

        x = self.embedding(tokens)
        x = self.position(x)

        x = self.transformer(
            x,
            src_key_padding_mask=~mask
        )

        scores = self.attention_pool(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)

        return self.classifier(pooled)
