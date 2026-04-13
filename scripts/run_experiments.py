import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from models import (
    BiGRUModel,
    BiLSTMModel,
    BiRNNModel,
    GRUModel,
    LSTMModel,
    RNNModel,
    TransformerModel,
)
from models.experiment_models import RecurrentClassifier, TCNClassifier, TransformerClassifier


@dataclass
class ExperimentSpec:
    name: str
    family: str
    epochs: int
    learning_rate: float = 1e-3
    hidden_size: int | None = None
    num_layers: int | None = None
    bidirectional: bool | None = None
    pooling: str | None = None
    transpose_input: bool = False
    d_model: int | None = None
    num_heads: int | None = None
    ff_dim: int | None = None
    dropout: float = 0.1


BASELINE_SPECS = [
    ExperimentSpec(name="RNN", family="baseline_rnn", epochs=6),
    ExperimentSpec(name="BiRNN", family="baseline_birnn", epochs=6),
    ExperimentSpec(name="LSTM", family="baseline_lstm", epochs=6),
    ExperimentSpec(name="BiLSTM", family="baseline_bilstm", epochs=6),
    ExperimentSpec(name="GRU", family="baseline_gru", epochs=6),
    ExperimentSpec(name="BiGRU", family="baseline_bigru", epochs=6),
    ExperimentSpec(name="TransformerMean", family="baseline_transformer", epochs=6),
]

EXTENDED_SPECS = [
    ExperimentSpec(name="BiLSTM-2Layer", family="recurrent", epochs=6, hidden_size=128, num_layers=2, bidirectional=True, pooling="last", dropout=0.1),
    ExperimentSpec(name="BiLSTM-2Layer-Attn", family="recurrent", epochs=6, hidden_size=128, num_layers=2, bidirectional=True, pooling="attention", dropout=0.1),
    ExperimentSpec(name="BiGRU-2Layer", family="recurrent", epochs=6, hidden_size=128, num_layers=2, bidirectional=True, pooling="last", dropout=0.1),
    ExperimentSpec(name="BiGRU-2Layer-Attn", family="recurrent", epochs=6, hidden_size=128, num_layers=2, bidirectional=True, pooling="attention", dropout=0.1),
    ExperimentSpec(name="BiLSTM-Columns", family="recurrent", epochs=6, hidden_size=128, num_layers=1, bidirectional=True, pooling="last", transpose_input=True),
    ExperimentSpec(name="Transformer-CLS", family="transformer", epochs=6, d_model=128, num_heads=8, num_layers=2, ff_dim=256, pooling="cls", dropout=0.1),
    ExperimentSpec(name="Transformer-Deep", family="transformer", epochs=6, d_model=128, num_heads=8, num_layers=4, ff_dim=256, pooling="mean", dropout=0.1),
    ExperimentSpec(name="TCN", family="tcn", epochs=6, dropout=0.1),
]

FINALISTS = [
    ExperimentSpec(name="TransformerMean", family="baseline_transformer", epochs=10),
    ExperimentSpec(name="BiGRU-2Layer", family="recurrent", epochs=10, hidden_size=128, num_layers=2, bidirectional=True, pooling="last", dropout=0.1),
    ExperimentSpec(name="BiGRU-2Layer-Attn", family="recurrent", epochs=10, hidden_size=128, num_layers=2, bidirectional=True, pooling="attention", dropout=0.1),
    ExperimentSpec(name="BiLSTM-2Layer-Attn", family="recurrent", epochs=10, hidden_size=128, num_layers=2, bidirectional=True, pooling="attention", dropout=0.1),
    ExperimentSpec(name="TCN", family="tcn", epochs=10, dropout=0.1),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(batch_size: int, seed: int, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    trainval_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed)

    train_size = int(len(trainval_dataset) * (2 / 3))
    val_size = len(trainval_dataset) - train_size
    train_subset, val_subset = random_split(trainval_dataset, [train_size, val_size], generator=generator)

    if train_limit is not None:
        train_subset = Subset(train_subset, list(range(min(train_limit, len(train_subset)))))
    if val_limit is not None:
        val_subset = Subset(val_subset, list(range(min(val_limit, len(val_subset)))))

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return {
        "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def forward_model(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    try:
        return model(inputs, device)
    except TypeError:
        return model(inputs)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = forward_model(model, inputs, device)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_one_experiment(spec: ExperimentSpec, dataloaders: dict[str, DataLoader], device: torch.device, seed: int, epochs_override: int | None = None) -> dict:
    set_seed(seed)
    model = build_model(spec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=spec.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_epoch = 0
    epoch_logs = []
    start_time = time.perf_counter()
    epochs = epochs_override or spec.epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_examples = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = forward_model(model, inputs, device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_examples += batch_size

        train_loss /= train_examples
        train_acc = train_correct / train_examples
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[{spec.name}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    best_val_loss, best_val_acc = evaluate(model, dataloaders["val"], criterion, device)
    test_loss, test_acc = evaluate(model, dataloaders["test"], criterion, device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "name": spec.name,
        "family": spec.family,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "parameter_count": count_parameters(model),
        "elapsed_seconds": elapsed_seconds,
        "spec": asdict(spec),
        "epoch_logs": epoch_logs,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(spec: ExperimentSpec) -> nn.Module:
    if spec.family == "baseline_rnn":
        return RNNModel()
    if spec.family == "baseline_birnn":
        return BiRNNModel()
    if spec.family == "baseline_lstm":
        return LSTMModel()
    if spec.family == "baseline_bilstm":
        return BiLSTMModel()
    if spec.family == "baseline_gru":
        return GRUModel()
    if spec.family == "baseline_bigru":
        return BiGRUModel()
    if spec.family == "baseline_transformer":
        return TransformerModel()
    if spec.family == "recurrent":
        name = spec.name.lower()
        cell_type = "lstm" if "lstm" in name else "gru" if "gru" in name else "rnn"
        return RecurrentClassifier(
            cell_type=cell_type,
            hidden_size=spec.hidden_size or 128,
            num_layers=spec.num_layers or 1,
            bidirectional=bool(spec.bidirectional),
            dropout=spec.dropout,
            pooling=spec.pooling or "last",
            transpose_input=spec.transpose_input,
        )
    if spec.family == "transformer":
        return TransformerClassifier(
            d_model=spec.d_model or 128,
            num_heads=spec.num_heads or 8,
            num_layers=spec.num_layers or 2,
            ff_dim=spec.ff_dim or 256,
            dropout=spec.dropout,
            pooling=spec.pooling or "mean",
            transpose_input=spec.transpose_input,
        )
    if spec.family == "tcn":
        return TCNClassifier(dropout=spec.dropout, transpose_input=spec.transpose_input)

    raise ValueError(f"Unsupported family: {spec.family}")


def resolve_suite(name: str) -> list[ExperimentSpec]:
    if name == "baseline":
        return BASELINE_SPECS
    if name == "extended":
        return EXTENDED_SPECS
    if name == "finalists":
        return FINALISTS
    if name == "all":
        return BASELINE_SPECS + EXTENDED_SPECS + FINALISTS
    raise ValueError(f"Unknown suite: {name}")


def save_results(results: list[dict], output_dir: Path, suite: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{suite}_results.json"
    csv_path = output_dir / f"{suite}_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    summary_rows = []
    for item in results:
        summary_rows.append(
            {
                "name": item["name"],
                "family": item["family"],
                "epochs": item["epochs"],
                "best_epoch": item["best_epoch"],
                "best_val_loss": f"{item['best_val_loss']:.6f}",
                "best_val_acc": f"{item['best_val_acc']:.6f}",
                "test_loss": f"{item['test_loss']:.6f}",
                "test_acc": f"{item['test_acc']:.6f}",
                "parameter_count": item["parameter_count"],
                "elapsed_seconds": f"{item['elapsed_seconds']:.2f}",
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["baseline", "extended", "finalists", "all"], default="baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Suite: {args.suite}")
    print(f"Seed: {args.seed}")
    if args.train_limit or args.val_limit:
        print(f"Subset limits: train={args.train_limit} val={args.val_limit}")
    if args.epochs_override is not None:
        print(f"Epoch override: {args.epochs_override}")

    dataloaders = build_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    specs = resolve_suite(args.suite)

    results = []
    for index, spec in enumerate(specs, start=1):
        print(f"\n=== Experiment {index}/{len(specs)}: {spec.name} ===")
        if args.epochs_override is not None:
            result = train_one_experiment(spec, dataloaders, device, args.seed, epochs_override=args.epochs_override)
        else:
            result = train_one_experiment(spec, dataloaders, device, args.seed)
        results.append(result)
        save_results(results, args.output_dir, args.suite)

    ranked = sorted(results, key=lambda item: item["best_val_acc"], reverse=True)
    print("\n=== Ranked by validation accuracy ===")
    for item in ranked:
        print(
            f"{item['name']:<22} val_acc={item['best_val_acc']:.4f} "
            f"test_acc={item['test_acc']:.4f} params={item['parameter_count']}"
        )


if __name__ == "__main__":
    main()
