import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import *
from scripts.download_mnist import dataloaders_dict
from scripts.train_model import train_model

# 使用デバイスの確認
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "RNN": RNNModel,
    "BiRNN": BiRNNModel,
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel,
    "GRU": GRUModel,
    "BiGRU": BiGRUModel,
    "Transformer": TransformerModel,
}

net = MODEL_REGISTRY["Transformer"]().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# 学習・検証を実行する
num_epochs = 50
train_model(net, dataloaders_dict, criterion, optimizer, device, num_epochs=num_epochs)

batch_iterator = iter(dataloaders_dict["test"])  # イテレータに変換
images, labels = next(batch_iterator)  # 1番目の要素を取り出す

net.eval() #推論モード
with torch.set_grad_enabled(False):   # 推論モードでは勾配を算出しない
   # GPUを使用する場合は明示的に指定
   images = images.to(device)
   labels = labels.to(device)
   # 出力
   outputs = net(images, device)     # 順伝播
   _, preds = torch.max(outputs, 1)  # ラベルを予測

if __name__ == "__main__":
    #テストデータの予測結果を描画
    # GPUを使用した場合は，'.detach().cpu().clone().numpy()'でarray型に変換できる
    plt.imshow(images[0].detach().cpu().clone().numpy().reshape(28,28), cmap='gray')
    plt.title("Label: Target={}, Predict={}".format(labels[0], preds[0].detach().cpu().clone().numpy()))
    plt.show()