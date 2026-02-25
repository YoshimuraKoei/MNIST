import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 28              # 画像の Height を時系列のSequenceとしてLSTMに入力する
        self.feature_size = 28         # 画像の Width を特徴量の次元としてLSTMに入力する
        self.hidden_size = 128   # 隠れ層のサイズ
        self.lstm_layers = 1           # LSTMのレイヤー数　(LSTMを何層重ねるか)
       
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=self.lstm_layers)   
        self.fc = nn.Linear(self.hidden_size, 10)

    def init_hidden(self, batch_size, device): # LSTMの隠れ層 hidden と記憶セル cell を初期化
        hidden = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)     
        return (hidden, cell)
        
    def forward(self, x, device):
        batch_size = x.shape[0]
        hidden, cell = self.init_hidden(batch_size, device)
        self.hidden = (hidden, cell)
       
        x = x.view(batch_size, self.seq_len, self.feature_size)  # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Sequence, Feature)
                                                                # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてLSTMに入力する
        x = x.permute(1, 0, 2)                                   # (Batch, Sequence, Feature) -> (Sequence, Batch, Feature)
       
        lstm_out, (h_n, c_n) = self.lstm(x, self.hidden)    # LSTMの入力データのShapeは(Sequence, Batch, Feature)
                                                                    # (h_n) のShapeは (num_layers, batch, hidden_size)
        x = h_n[-1,:,:]                                          # lstm_layersの最後のレイヤーを取り出す  (B, h)
        x = self.fc(x)                                           # (B, h) -> (B, 10)
       
        return x