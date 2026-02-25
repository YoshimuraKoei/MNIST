import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28
        self.feature_size = 28
        self.hidden_size = 128    # 隠れ層の次元
        self.rnn_layers = 1     # RNN を何層重ねるか

        self.simple_rnn = nn.RNN(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers)
        self.fc = nn.Linear(self.hidden_size, 10)

    def init_hidden(self, batch_size, device):  # RNNの隠れ層hiddenを初期化
        hidden = torch.zeros(self.rnn_layers, batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, x, device):
        batch_size = x.shape[0]

        self.hidden = self.init_hidden(batch_size, device)

        x = x.view(batch_size, self.seq_len, self.feature_size)     # (Batch, Channel, Height, Width) -> (Batch, Height, Width) = (Batch, Sequence, Feature)
                                                                    # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてRNNに入力する
        x = x.permute(1, 0, 2)                                      # (Batch, Sequence, Feature) -> (Sequence, Batch, Feature)
        rnn_out, h_n = self.simple_rnn(x, self.hidden)           # RNNの入力データのShapeは(Seqence, Batch, Feature)
                                                                # (h_n) のShapeは (rnn_layers, batch, hidden_size)
        x = h_n[-1,:,:]                                          # RNNの最後のレイヤーを取り出す (rnn_layers, batch, hidden_size)-> (batch, hidden_size)
        x = self.fc(x)                                           # (batch, hidden_size) -> (batch, 10)
        
        return x
