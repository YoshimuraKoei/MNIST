# MNIST Sequence Model Experiments

MNIST を通常の画像分類としてではなく、`28 x 28` 画像を `28` ステップの系列として見なして比較する実験リポジトリです。

既存実装として以下を含みます。

- `RNN`
- `BiRNN`
- `LSTM`
- `BiLSTM`
- `GRU`
- `BiGRU`
- `Transformer`

追加で、研究用の実験ランナーと拡張モデルを入れています。

- `2-layer BiLSTM / BiGRU`
- `attention pooling`
- `column-wise sequence` の比較
- `TCN`
- `Transformer CLS pooling / deeper encoder`

## Run

既存の単体学習スクリプト:

```bash
.venv/bin/python scripts/train.py
```

実験ランナー:

```bash
.venv/bin/python scripts/run_experiments.py --suite baseline --device cpu
.venv/bin/python scripts/run_experiments.py --suite extended --device cpu
.venv/bin/python scripts/run_experiments.py --suite finalists --device cpu
.venv/bin/python scripts/run_experiments.py --suite hypothesis1 --device cpu
.venv/bin/python scripts/run_experiments.py --suite hypothesis2 --device cpu
.venv/bin/python scripts/run_experiments.py --suite hypothesis2_final --device cpu
```

探索用にデータ数を絞る場合:

```bash
.venv/bin/python scripts/run_experiments.py \
  --suite baseline \
  --device cpu \
  --batch-size 256 \
  --train-limit 12000 \
  --val-limit 4000
```

結果は `artifacts/experiments/` に JSON と CSV で保存されます。要約は [`RESULT.md`](./RESULT.md) に残しています。

## Hypothesis Loops

第 2 ラウンドでは、単純なモデル比較だけでなく仮説駆動で実験を回しています。

- `hypothesis1`: TCN の強さが行列方向の局所性や pooling に由来するかを検証
- `hypothesis2`: 局所畳み込み特徴に recurrent readout を重ねるとさらに伸びるかを検証
