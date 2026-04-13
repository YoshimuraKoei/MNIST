import torch
import torch.nn as nn


class SequenceAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        cell_type: str,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pooling: str = "last",
        transpose_input: bool = False,
    ):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pooling = pooling
        self.transpose_input = transpose_input

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        module_by_type = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        recurrent_cls = module_by_type[cell_type]
        self.recurrent = recurrent_cls(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=recurrent_dropout,
            batch_first=True,
        )

        output_size = self.hidden_size * self.num_directions
        if self.pooling == "attention":
            self.pool = SequenceAttentionPooling(output_size)
        else:
            self.pool = None

        self.fc = nn.Linear(output_size, 10)

    def forward(self, x: torch.Tensor, device=None) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_size)
        if self.transpose_input:
            x = x.transpose(1, 2)

        outputs, hidden = self.recurrent(x)

        if self.pooling == "last":
            hidden_state = hidden[0] if isinstance(hidden, tuple) else hidden
            hidden_state = hidden_state.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            last_layer_state = hidden_state[-1]
            if self.bidirectional:
                x = torch.cat([last_layer_state[0], last_layer_state[1]], dim=1)
            else:
                x = last_layer_state[0]
        elif self.pooling == "mean":
            x = outputs.mean(dim=1)
        elif self.pooling == "attention":
            x = self.pool(outputs)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.fc(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        transpose_input: bool = False,
    ):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.d_model = d_model
        self.pooling = pooling
        self.transpose_input = transpose_input

        token_count = self.seq_len + (1 if pooling == "cls" else 0)
        self.input_proj = nn.Linear(self.feature_size, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_count, self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if pooling == "cls" else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, 10)

    def forward(self, x: torch.Tensor, device=None) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_size)
        if self.transpose_input:
            x = x.transpose(1, 2)

        x = self.input_proj(x)
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)

        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.fc(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.skip(x))


class TCNClassifier(nn.Module):
    def __init__(self, channels=(64, 128, 128), dropout: float = 0.1, transpose_input: bool = False):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.transpose_input = transpose_input

        blocks = []
        in_channels = self.feature_size
        for index, out_channels in enumerate(channels):
            blocks.append(
                ResidualConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 10)

    def forward(self, x: torch.Tensor, device=None) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.feature_size)
        if self.transpose_input:
            x = x.contiguous()
        else:
            x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
