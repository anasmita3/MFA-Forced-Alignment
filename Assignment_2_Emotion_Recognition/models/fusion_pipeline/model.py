import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        num_classes=7,
        dropout=0.3
    ):
        super().__init__()

        # Audio attends to text
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Text attends to audio
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization
        self.audio_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)

        # Attention pooling
        self.audio_pool = nn.Linear(d_model, 1)
        self.text_pool = nn.Linear(d_model, 1)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        audio_features,
        text_features,
        text_padding_mask
    ):

        # Audio queries text
        audio_context, _ = self.audio_to_text(
            query=audio_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_padding_mask,
            need_weights=False
        )

        # Text queries audio
        text_context, _ = self.text_to_audio(
            query=text_features,
            key=audio_features,
            value=audio_features,
            need_weights=False
        )

        # Residual connections and normalization
        audio_enhanced = self.audio_norm(
            audio_features + audio_context
        )

        text_enhanced = self.text_norm(
            text_features + text_context
        )

        # Audio attention pooling
        audio_scores = self.audio_pool(
            audio_enhanced
        ).squeeze(-1)

        audio_weights = torch.softmax(
            audio_scores,
            dim=1
        )

        audio_vector = torch.sum(
            audio_enhanced * audio_weights.unsqueeze(-1),
            dim=1
        )

        # Text attention pooling
        text_scores = self.text_pool(
            text_enhanced
        ).squeeze(-1)

        text_scores = text_scores.masked_fill(
            text_padding_mask,
            float("-inf")
        )

        text_weights = torch.softmax(
            text_scores,
            dim=1
        )

        text_vector = torch.sum(
            text_enhanced * text_weights.unsqueeze(-1),
            dim=1
        )

        # Concatenate four representations
        fused = torch.cat(
            [
                audio_vector,
                text_vector,
                audio_context.mean(dim=1),
                text_context.masked_fill(
                    text_padding_mask.unsqueeze(-1),
                    0.0
                ).sum(dim=1) /
                (~text_padding_mask).sum(
                    dim=1, keepdim=True
                ).clamp(min=1)
            ],
            dim=1
        )

        return self.classifier(fused)
