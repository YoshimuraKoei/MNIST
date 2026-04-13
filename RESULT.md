# MNIST 結果

## 初期メモの設定

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

## 2026-04-13 実験

### 設定

- seed: `42`
- optimizer: `Adam`
- loss: `CrossEntropyLoss`
- train/val split: `2:1`
- 実行環境: CPU
- 学習ランナー: `scripts/run_experiments.py`
- 保存先: `artifacts/experiments/`

### スクリーニング 1: 既存モデル再ベースライン

条件:

- train subset: `12000`
- val subset: `4000`
- test: `10000` 全件
- epochs: `6`
- batch size: `256`

| Model | Val Acc | Test Acc | Params |
| --- | ---: | ---: | ---: |
| TransformerMean | 0.9537 | 0.9609 | 273,802 |
| BiLSTM | 0.9460 | 0.9484 | 164,362 |
| BiGRU | 0.9357 | 0.9438 | 123,914 |
| GRU | 0.9293 | 0.9329 | 61,962 |
| LSTM | 0.9240 | 0.9252 | 82,186 |
| BiRNN | 0.8932 | 0.9015 | 43,018 |
| RNN | 0.7660 | 0.7742 | 21,514 |

観察:

- 既存実装では `TransformerMean` が最良。
- 双方向化は一貫して効いている。
- 単純 `RNN` はかなり弱い。

### スクリーニング 2: 拡張系列モデル

条件:

- train subset: `12000`
- val subset: `4000`
- test: `10000` 全件
- epochs: `6`
- batch size: `256`

| Model | Val Acc | Test Acc | Params |
| --- | ---: | ---: | ---: |
| TCN | 0.9800 | 0.9809 | 203,082 |
| BiGRU-2Layer | 0.9657 | 0.9696 | 420,362 |
| BiGRU-2Layer-Attn | 0.9655 | 0.9702 | 420,619 |
| BiLSTM-2Layer-Attn | 0.9615 | 0.9678 | 559,883 |
| BiLSTM-2Layer | 0.9540 | 0.9608 | 559,626 |
| Transformer-CLS | 0.9520 | 0.9590 | 274,058 |
| BiLSTM-Columns | 0.9477 | 0.9538 | 164,362 |
| Transformer-Deep | 0.9447 | 0.9518 | 538,762 |

観察:

- `TCN` が明確に強く、6 epoch 時点で既存群を大きく上回った。
- RNN 系では `2-layer BiGRU` が最も伸びた。
- `attention pooling` は `BiLSTM` では効き、`BiGRU` では小差。
- `CLS pooling` と深い Transformer は、今回の設定では mean pooling の既存 Transformer を越えなかった。
- 列方向入力 (`BiLSTM-Columns`) は悪くないが、行方向よりわずかに不利。

### 最終追試: フルデータ

条件:

- train: `40000`
- val: `20000`
- test: `10000` 全件
- epochs: `10`
- batch size: `256`

| Rank | Model | Best Epoch | Val Acc | Test Acc | Params |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | TCN | 10 | 0.9902 | 0.9912 | 203,082 |
| 2 | BiGRU-2Layer-Attn | 10 | 0.9888 | 0.9886 | 420,619 |
| 3 | BiGRU-2Layer | 10 | 0.9879 | 0.9875 | 420,362 |
| 4 | BiLSTM-2Layer-Attn | 10 | 0.9856 | 0.9868 | 559,883 |
| 5 | TransformerMean | 9 | 0.9797 | 0.9794 | 273,802 |

結論:

- このリポジトリの「MNIST を時系列として扱う」設定では、最終的な勝ち筋は `TCN`。
- 再帰系の最良は `BiGRU-2Layer-Attn`。Transformer より高精度だった。
- 既存の 1 層 recurrent 系は比較対象としては十分だが、性能を見るなら 2 層化の効果が大きい。
- Transformer は悪くないが、この設定では最良ではなかった。
