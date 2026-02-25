import sys, os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#データ前処理 transform を設定
transform = transforms.Compose(
   [transforms.ToTensor(),                      # Tensor変換とshape変換 [H, W, C] -> [C, H, W]
    transforms.Normalize((0.5, ), (0.5, ))])    # 標準化 平均:0.5  標準偏差:0.5

#訓練用(train + validation)のデータセット サイズ:(channel, height, width) = (1,28,28) 60000枚
trainval_dataset = datasets.MNIST(root='./data', 
                                       train=True,  # True:訓練用60,000枚, False:テスト用10,000枚
                                       download=True,
                                       transform=transform)                                                     
                    
# 訓練データを検証データと分ける時の比率
TRAIN_VAL_SPLIT_RATIO = 2 / 3

n_total = len(trainval_dataset)
n_train = int(n_total * TRAIN_VAL_SPLIT_RATIO)
n_val = n_total - n_train

train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [n_train, n_val])
print("train_dataset size = {}".format(len(train_dataset)))
print("val_dataset size = {}".format(len(val_dataset)))

#テスト(test)用のデータセット サイズ:(channel, height, width) = (1,28,28) 10000枚
test_dataset = datasets.MNIST(root='./data', 
                                       train=False, 
                                       download=True, 
                                       transform=transform)


#訓練用 Dataloder
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)    
                                           
#検証用 Dataloder
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
                                           
#テスト用 Dataloder
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
                                           
# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


if __name__ == "__main__":
    batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
    images, labels = next(batch_iterator)  # 1番目の要素を取り出す
    print("images size = ", images.size())
    print("labels size = ", labels.size())
    #試しに1枚 plot してみる
    plt.imshow(images[0].numpy().reshape(28,28), cmap='gray')
    plt.title("label = {}".format(labels[0].numpy()))
    plt.show()