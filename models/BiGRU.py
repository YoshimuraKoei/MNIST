import torch
import torch.nn as nn


class BiGRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.hidden_size = 128
        self.gru_layers = 1
        self.num_directions = 2

        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.gru_layers,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_size * self.num_directions, 10)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(
            self.gru_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return hidden

    def forward(self, x, device):
        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size, device)

        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.view(batch_size, self.seq_len, self.feature_size)
        # nn.GRU default expects (T, B, C)
        x = x.permute(1, 0, 2)

        gru_out, h_n = self.gru(x, self.hidden)
        # h_n shape: (num_layers * 2, B, H)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.fc(x)
        return x
