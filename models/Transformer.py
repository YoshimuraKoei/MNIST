import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28   # 1行(28画素)を1トークンの特徴量として使う

        self.d_model = 128
        self.num_heads = 8
        self.num_layers = 2
        self.ff_dim = 256
        self.dropout = 0.1

        self.input_proj = nn.Linear(self.feature_size, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, 10)

    def forward(self, x, device):
        batch_size = x.shape[0]

        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.view(batch_size, self.seq_len, self.feature_size)

        # (B, T, 28) -> (B, T, d_model)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : self.seq_len, :]

        x = self.transformer(x)   # (B, T, d_model)
        x = self.norm(x)

        # Sequence-to-label classification: mean pooling over time
        x = x.mean(dim=1)         # (B, d_model)
        x = self.fc(x)            # (B, 10)
        return x
