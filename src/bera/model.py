import torch
import torch.nn as nn
from .config import BERAConfig


class BERAEncoderLayer(nn.Module):
    def __init__(self, config: BERAConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.activation = nn.GELU()

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        x = self.norm1(x + attn_out)

        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = self.norm2(x + ffn_out)
        return x


class BERAEncoder(nn.Module):
    def __init__(self, config: BERAConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            BERAEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class BERA(nn.Module):
    def __init__(
        self,
        config: BERAConfig,
        num_sentiment_labels: int,
        num_emotion_labels: int,
        max_position_embeddings: int = None
    ):
        super().__init__()
        self.config = config

        if max_position_embeddings is None:
            max_position_embeddings = config.max_position_embeddings

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = BERAEncoder(config)

        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)
        self.emotion_classifier = nn.Linear(config.hidden_size, num_emotion_labels)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(self.dropout(x))

        hidden_states = self.encoder(x, attention_mask)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden_states.mean(1)

        return {
            "sentiment_logits": self.sentiment_classifier(pooled),
            "emotion_logits": self.emotion_classifier(pooled),
        }
