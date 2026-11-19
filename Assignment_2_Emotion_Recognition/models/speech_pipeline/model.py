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


class OriginalCNNTransformerSpeech(nn.Module):
    def __init__(
        self,
        num_classes=7,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.projection = nn.Linear(64 * 16, d_model)
        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
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

    def forward(self, x):
        # Input shape: (batch, 300, 64)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.cnn(x)

        batch, channels, frequency, time = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * frequency)

        x = self.projection(x)
        x = self.position(x)
        x = self.transformer(x)

        weights = torch.softmax(
            self.attention_pool(x),
            dim=1
        )

        representation = torch.sum(weights * x, dim=1)

        return self.classifier(representation)
