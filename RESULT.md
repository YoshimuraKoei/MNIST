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

## 2026-04-13 PDCA ラウンド 2

### 出発点

- 前ラウンドでは `TCN` が最良で、フルデータ追試で `val 0.9902 / test 0.9912`。

### 仮説 1

- `TCN` の強さは、長距離依存そのものより「局所パターン抽出」と「集約方法」にある。
- もしそうなら、列方向単独や行列融合、pooling の変更で差が出るはず。

### 実験 1

条件:

- train subset: `12000`
- val subset: `4000`
- test: `10000` 全件
- epochs: `6`
- batch size: `256`

| Model | Val Acc | Test Acc | Params |
| --- | ---: | ---: | ---: |
| TCN-MaxAvg | 0.9828 | 0.9824 | 204,362 |
| TCN | 0.9800 | 0.9809 | 203,082 |
| TCN-WideK5 | 0.9795 | 0.9802 | 329,546 |
| DualTCN-MaxAvg | 0.9778 | 0.9826 | 470,538 |
| DualTCN | 0.9745 | 0.9804 | 437,770 |
| TCN-Columns | 0.9663 | 0.9736 | 203,082 |

### 解釈 1

- 列方向単独 (`TCN-Columns`) は明確に弱い。主情報は行方向系列に多い。
- 行列融合 (`DualTCN`) は悪くないが、パラメータ増に対して改善が小さい。
- `max+avg pooling` は一貫して効いており、今回の `TCN` の改善余地は orientation fusion より readout 側にある。

### 仮説 2

- 局所畳み込み特徴は有効なので、その後段に recurrent readout を載せれば、単純 pooling より少し伸びる可能性がある。

### 実験 2a: サブセット探索

条件:

- train subset: `12000`
- val subset: `4000`
- test: `10000` 全件
- epochs: `6`
- batch size: `256`

| Model | Val Acc | Test Acc | Params |
| --- | ---: | ---: | ---: |
| TCN-MaxAvg | 0.9828 | 0.9824 | 204,362 |
| TCN-BiGRU-WideK5 | 0.9822 | 0.9817 | 364,619 |
| TCN-BiGRU | 0.9792 | 0.9828 | 303,691 |
| BiGRU-2Layer-Attn | 0.9655 | 0.9702 | 420,619 |

### 解釈 2a

- サブセットでは `TCN-BiGRU` 系は強いが、`TCN-MaxAvg` を明確には超えなかった。
- ただし差は小さいので、フルデータでは順位が入れ替わる余地がある。

### 実験 2b: フルデータ追試

条件:

- train: `40000`
- val: `20000`
- test: `10000` 全件
- epochs: `10`
- batch size: `256`

| Model | Best Epoch | Val Acc | Test Acc | Params |
| --- | ---: | ---: | ---: | ---: |
| TCN-BiGRU-WideK5 | 9 | 0.9916 | 0.9914 | 364,619 |
| TCN-BiGRU | 9 | 0.9909 | 0.9913 | 303,691 |
| TCN-MaxAvg | 9 | 0.9906 | 0.9916 | 204,362 |

### 解釈 2b

- best validation では `TCN-BiGRU-WideK5` が最良。
- best test では `TCN-MaxAvg` がわずかに最良。
- 差は非常に小さく、実質的には `TCN-MaxAvg` と `TCN-BiGRU-WideK5` はほぼ同格。
- そのうえで、パラメータ数と構成の単純さを考えると、実用上の第一候補は `TCN-MaxAvg`。

### ここまでの暫定結論

- 「TCN が強い理由」は、行列融合よりも局所畳み込みと適切な pooling にある。
- recurrent を後段に足すと validation は少し伸びうるが、1 seed では決定的な優位を示していない。
- この時点では `TCN-MaxAvg` と `TCN-BiGRU-WideK5` の複数 seed 比較が最優先。

## 2026-04-14 多 seed 判定

### 判定方針

- 仮説の採択・棄却は、単発 seed の勝敗ではなく複数 seed の平均と多数決で判断する。
- 仮説 1 は subset 条件 (`12000/4000`, 6 epoch) で 3 seed 比較。
- 仮説 2 はフルデータ条件 (`40000/20000`, 10 epoch) で 3 seed 比較。

使用 seed:

- `42`
- `123`
- `777`

### 仮説 1 の再判定

対象:

- `TCN`
- `TCN-MaxAvg`
- `TCN-Columns`
- `DualTCN`
- `DualTCN-MaxAvg`
- `TCN-WideK5`

3 seed 集約結果:

| Model | Mean Val Acc | Std | Mean Test Acc | Std |
| --- | ---: | ---: | ---: | ---: |
| TCN-MaxAvg | 0.982500 | 0.000736 | 0.983267 | 0.000736 |
| TCN | 0.981000 | 0.001242 | 0.982000 | 0.001098 |
| DualTCN | 0.977750 | 0.002354 | 0.980167 | 0.000634 |
| DualTCN-MaxAvg | 0.977500 | 0.001137 | 0.979300 | 0.002736 |
| TCN-WideK5 | 0.977167 | 0.004028 | 0.977867 | 0.004177 |
| TCN-Columns | 0.968333 | 0.001559 | 0.972800 | 0.001349 |

判定:

- 仮説 1 のうち「pooling 改善が効く」は採択。
- 仮説 1 のうち「行列融合が本質的に効く」は棄却。

理由:

- `TCN-MaxAvg` は 3 seed 平均で素の `TCN` を上回った。
- `DualTCN` 系はパラメータ増に対して平均性能で優位を示せなかった。
- `TCN-Columns` は一貫して弱く、列方向単独は主役ではない。

### 仮説 2 の再判定

対象:

- `TCN-MaxAvg`
- `TCN-BiGRU`
- `TCN-BiGRU-WideK5`

3 seed 集約結果:

| Model | Mean Val Acc | Std | Mean Test Acc | Std | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| TCN-BiGRU | 0.990117 | 0.000592 | 0.991867 | 0.000492 | 303,691 |
| TCN-MaxAvg | 0.989867 | 0.000487 | 0.991333 | 0.000249 | 204,362 |
| TCN-BiGRU-WideK5 | 0.989300 | 0.001628 | 0.989933 | 0.001226 | 364,619 |

多数決:

- `TCN-BiGRU` は `TCN-MaxAvg` に対して、validation で 2/3 seed 勝ち
- `TCN-BiGRU` は `TCN-MaxAvg` に対して、test で 2/3 seed 勝ち

判定:

- 仮説 2 は採択。
- ただし採択されるのは `TCN-BiGRU` であって、`WideK5` 版ではない。

理由:

- `TCN-BiGRU` は 3 seed 平均で validation/test ともに `TCN-MaxAvg` をわずかに上回った。
- 差は大きくないが、平均と多数決の両方で `TCN-BiGRU` 側に寄っている。
- 一方 `TCN-BiGRU-WideK5` は 1 seed では強く見えたが、3 seed 平均では最良ではなかった。

### 現時点の最終結論

- 仮説 1:
  pooling 改善は採択、行列融合の本質性は棄却。
- 仮説 2:
  `TCN` の後段に recurrent readout を足すと、平均的にはわずかに改善するので採択。
- 現時点の研究上の best model:
  `TCN-BiGRU`
- 現時点の実用上の best trade-off:
  `TCN-MaxAvg`
