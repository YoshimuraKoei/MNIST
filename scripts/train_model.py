import torch
from tqdm import tqdm

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, device, num_epochs):
   for epoch in range(num_epochs):
       print('Epoch {}/{}'.format(epoch+1, num_epochs))
       print('-------------')
       # epochごとの学習と検証のループ
       for phase in ['train', 'val']:
           if phase == 'train':
               net.train()  # モデルを訓練モードに
           else:
               net.eval()   # モデルを検証モードに
           epoch_loss = 0.0  # epochの損失和
           epoch_corrects = 0  # epochの正解数
           # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
           if (epoch == 0) and (phase == 'train'):
               continue
           # データローダーからミニバッチを取り出すループ
           for i , (inputs, labels) in tqdm(enumerate(dataloaders_dict[phase])):
               
               # GPUを使用する場合は明示的に指定
               inputs = inputs.to(device)
               labels = labels.to(device)
               # optimizerを初期化
               optimizer.zero_grad()
               # 順伝搬（forward）計算
               with torch.set_grad_enabled(phase == 'train'):  # 訓練モードのみ勾配を算出
                   outputs = net(inputs, device)    # 順伝播
                   loss = criterion(outputs, labels)  # 損失を計算
                   _, preds = torch.max(outputs, 1)   # ラベルを予測
                   
 
                   # 訓練時はバックプロパゲーション
                   if phase == 'train':
                       loss.backward()
                       optimizer.step()
                   # イタレーション結果の計算
                   # lossの合計を更新
                   epoch_loss += loss.item() * inputs.size(0)  
                   # 正解数の合計を更新
                   epoch_corrects += torch.sum(preds == labels.data)
           # epochごとのlossと正解率を表示
           epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
           epoch_acc = epoch_corrects.float() / len(dataloaders_dict[phase].dataset)
           
           print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))