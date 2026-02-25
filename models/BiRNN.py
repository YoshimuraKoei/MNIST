import torch
import torch.nn as nn

class BiRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.hidden_size = 128
        self.rnn_layers = 1
        self.num_directions = 2

        self.rnn = nn.RNN(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            nonlinearity="tanh",
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_size * self.num_directions, 10)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(
            self.rnn_layers * self.num_directions,
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
        # nn.RNN default expects (T, B, C)
        x = x.permute(1, 0, 2)

        rnn_out, h_n = self.rnn(x, self.hidden)
        # h_n shape: (num_layers * 2, B, H)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.fc(x)
        return x
