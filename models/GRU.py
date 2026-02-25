import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28               # 画像の Height を時系列長として使う
        self.feature_size = 28          # 画像の Width を各時刻の特徴次元として使う
        self.hidden_size = 128          # GRU の隠れ状態次元
        self.gru_layers = 1             # GRU を何層重ねるか

        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.gru_layers,
        )
        self.fc = nn.Linear(self.hidden_size, 10)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, x, device):
        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size, device)

        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.view(batch_size, self.seq_len, self.feature_size)
        # nn.GRU default expects (T, B, C)
        x = x.permute(1, 0, 2)

        gru_out, h_n = self.gru(x, self.hidden)
        # 最終レイヤーの最終 hidden state を分類に使う
        x = h_n[-1, :, :]
        x = self.fc(x)

        return x
