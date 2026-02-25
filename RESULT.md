# MNIST 結果

## 設定

- エポック数： 50
- 損失関数: `nn.CrossEntropyLoss()`
- 最適化: Adam
- 訓練データと検証データの比率: 2:1
- 記載: 50エポック目の検証損失と正解率

## RNN

- ファイル： `models/RNN.py`
    - 検証損失: 0.1187
    - 正解率: 0.9703

## BiRNN

- ファイル: `models/BiRNN.py`
    - 検証損失: 0.1214
    - 正解率: 0.9663

## LSTM

- ファイル: `models/LSTM.py`
    - 検証損失: 0.0641
    - 正解率: 0.9880

## BiLSTM

- ファイル: `models/BiLSTM.py`
    - 検証損失: 0.0584
    - 正解率: 0.9879


## GRU

- ファイル: `models/GRU.py`
    - 検証損失: 0.0592
    - 正解率: 0.9872

## BiGRU

- ファイル: `models/BiGRU.py`
    - 検証損失: 0.0611
    - 正解率: 0.9880

## Transformer

- ファイル: `models/Transformer.py`
    - 検証損失: 0.0659
    - 正解率: 0.9845