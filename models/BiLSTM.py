import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.hidden_size = 128
        self.lstm_layers = 1
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_size * self.num_directions, 10)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(
            self.lstm_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        cell = torch.zeros(
            self.lstm_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (hidden, cell)

    def forward(self, x, device):
        batch_size = x.shape[0]
        hidden, cell = self.init_hidden(batch_size, device)
        self.hidden = (hidden, cell)

        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.view(batch_size, self.seq_len, self.feature_size)
        # nn.LSTM default expects (T, B, C)
        x = x.permute(1, 0, 2)

        lstm_out, (h_n, c_n) = self.lstm(x, self.hidden)
        # h_n shape: (num_layers * 2, B, H)
        # 最終レイヤーの forward/backward を連結して (B, 2H) にする
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.fc(x)

        return x
