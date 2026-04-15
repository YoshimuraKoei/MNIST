from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
MULTISEED_DIR = ARTIFACTS_DIR / "multiseed"
REPORT_RUNS_DIR = ARTIFACTS_DIR / "report_runs" / "hypothesis2_final"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def epoch_range(result: dict) -> list[int]:
    return [row["epoch"] for row in result["epoch_logs"]]


def metric_curve(result: dict, metric: str) -> list[float]:
    return [row[metric] for row in result["epoch_logs"]]


def avg_epoch_time(result: dict) -> float:
    return float(result["elapsed_seconds"]) / int(result["epochs"])


def plot_line_chart(results: list[dict], title: str, output_name: str, metric: str = "val_acc") -> None:
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            epoch_range(result),
            metric_curve(result, metric),
            marker="o",
            linewidth=2,
            label=result["name"],
        )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def plot_bar_chart(results: list[dict], title: str, ylabel: str, output_name: str) -> None:
    ordered = sorted(results, key=avg_epoch_time, reverse=True)
    labels = [result["name"] for result in ordered]
    values = [avg_epoch_time(result) for result in ordered]

    plt.figure(figsize=(11, 6))
    plt.bar(labels, values, color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def plot_aggregate_chart(rows: list[dict], title: str, metric: str, error: str, output_name: str) -> None:
    ordered = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
    labels = [row["name"] for row in ordered]
    values = [float(row[metric]) for row in ordered]
    errors = [float(row[error]) for row in ordered]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, yerr=errors, capsize=5, color="#55A868")
    plt.title(title)
    plt.ylabel(metric)
    plt.ylim(min(values) - 0.01, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def plot_group_curves(results: list[dict], model_names: list[str], title: str, output_name: str, metric: str) -> None:
    selected = [result for result in results if result["name"] in model_names]
    selected.sort(key=lambda result: model_names.index(result["name"]))
    plot_line_chart(selected, title, output_name, metric=metric)


def load_report_run_results() -> dict[str, list[dict]]:
    per_seed: dict[str, list[dict]] = {}
    for seed_dir in sorted(REPORT_RUNS_DIR.glob("seed*/hypothesis2_final_results.json")):
        per_seed[seed_dir.parent.name] = load_results(seed_dir)
    return per_seed


def collect_multiseed_curves(per_seed: dict[str, list[dict]], model_names: list[str], metric: str) -> dict[str, np.ndarray]:
    curves: dict[str, list[list[float]]] = defaultdict(list)
    for results in per_seed.values():
        by_name = {result["name"]: result for result in results}
        for model_name in model_names:
            curves[model_name].append(metric_curve(by_name[model_name], metric))
    return {name: np.array(series) for name, series in curves.items()}


def plot_multiseed_band(per_seed: dict[str, list[dict]], model_names: list[str], metric: str, title: str, output_name: str, ylabel: str) -> None:
    curves = collect_multiseed_curves(per_seed, model_names, metric)
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, next(iter(curves.values())).shape[1] + 1)

    for model_name in model_names:
        values = curves[model_name]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        plt.plot(x_axis, mean, marker="o", linewidth=2, label=model_name)
        plt.fill_between(x_axis, mean - std, mean + std, alpha=0.18)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def plot_epoch_time_band(per_seed: dict[str, list[dict]], model_names: list[str], title: str, output_name: str) -> None:
    plot_multiseed_band(
        per_seed,
        model_names,
        metric="epoch_seconds",
        title=title,
        output_name=output_name,
        ylabel="Seconds / Epoch",
    )


def plot_accuracy_time_tradeoff(per_seed: dict[str, list[dict]], title: str, output_name: str) -> None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"test_acc": [], "epoch_seconds": []})
    for results in per_seed.values():
        for result in results:
            grouped[result["name"]]["test_acc"].append(float(result["test_acc"]))
            grouped[result["name"]]["epoch_seconds"].append(np.mean(metric_curve(result, "epoch_seconds")))

    plt.figure(figsize=(8, 6))
    for model_name, metrics in grouped.items():
        mean_test = float(np.mean(metrics["test_acc"]))
        mean_epoch = float(np.mean(metrics["epoch_seconds"]))
        plt.scatter(mean_epoch, mean_test, s=90)
        plt.annotate(model_name, (mean_epoch, mean_test), textcoords="offset points", xytext=(6, 4))

    plt.title(title)
    plt.xlabel("Mean Seconds / Epoch")
    plt.ylabel("Mean Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def plot_top_baseline_diagnostics(results: list[dict], model_names: list[str], output_name: str) -> None:
    selected = [result for result in results if result["name"] in model_names]
    selected.sort(key=lambda result: model_names.index(result["name"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("val_acc", "Validation Accuracy"),
    ]

    for axis, (metric, label) in zip(axes, metrics):
        for result in selected:
            axis.plot(epoch_range(result), metric_curve(result, metric), marker="o", linewidth=2, label=result["name"])
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Loss / Accuracy")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("Top Baseline Models: Convergence Diagnostics")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_results(EXPERIMENTS_DIR / "baseline_results.json")
    extended = load_results(EXPERIMENTS_DIR / "extended_results.json")
    hypothesis1 = load_results(EXPERIMENTS_DIR / "hypothesis1_results.json")
    hypothesis2 = load_results(EXPERIMENTS_DIR / "hypothesis2_results.json")
    hypothesis2_final = load_results(EXPERIMENTS_DIR / "hypothesis2_final_results.json")
    per_seed_finalists = load_report_run_results()

    plot_line_chart(
        baseline,
        "Baseline Core Models: Validation Accuracy by Epoch",
        "baseline_core_val_curves.png",
    )
    plot_line_chart(
        extended,
        "Extended Models: Validation Accuracy by Epoch",
        "baseline_extended_val_curves.png",
    )
    plot_bar_chart(
        baseline + extended,
        "Baseline and Extended Models: Average Time per Epoch",
        "Seconds / Epoch",
        "baseline_avg_epoch_time.png",
    )
    plot_top_baseline_diagnostics(
        baseline + extended,
        ["TransformerMean", "BiGRU-2Layer-Attn", "TCN"],
        "baseline_top_diagnostics.png",
    )
    plot_line_chart(
        hypothesis1,
        "Hypothesis 1 Models: Validation Accuracy by Epoch",
        "hypothesis1_val_curves.png",
    )
    plot_aggregate_chart(
        load_csv(MULTISEED_DIR / "hypothesis1_aggregate.csv"),
        "Hypothesis 1: Mean Test Accuracy Across Seeds",
        "mean_test_acc",
        "std_test_acc",
        "hypothesis1_multiseed_test_acc.png",
    )
    plot_line_chart(
        hypothesis2,
        "Hypothesis 2 Subset Screen: Validation Accuracy by Epoch",
        "hypothesis2_subset_val_curves.png",
    )
    plot_line_chart(
        hypothesis2_final,
        "Hypothesis 2 Finalists: Validation Accuracy by Epoch",
        "hypothesis2_final_val_curves.png",
    )
    plot_aggregate_chart(
        load_csv(MULTISEED_DIR / "hypothesis2_final_aggregate.csv"),
        "Hypothesis 2: Mean Test Accuracy Across Seeds",
        "mean_test_acc",
        "std_test_acc",
        "hypothesis2_multiseed_test_acc.png",
    )
    top_models = ["TCN-MaxAvg", "TCN-BiGRU"]
    plot_multiseed_band(
        per_seed_finalists,
        top_models,
        metric="val_acc",
        title="Top Finalists: Mean Validation Accuracy Across Seeds",
        output_name="hypothesis2_top_multiseed_val_band.png",
        ylabel="Validation Accuracy",
    )
    plot_multiseed_band(
        per_seed_finalists,
        top_models,
        metric="val_loss",
        title="Top Finalists: Mean Validation Loss Across Seeds",
        output_name="hypothesis2_top_multiseed_valloss_band.png",
        ylabel="Validation Loss",
    )
    plot_multiseed_band(
        per_seed_finalists,
        top_models,
        metric="train_loss",
        title="Top Finalists: Mean Train Loss Across Seeds",
        output_name="hypothesis2_top_multiseed_trainloss_band.png",
        ylabel="Train Loss",
    )
    plot_epoch_time_band(
        per_seed_finalists,
        top_models,
        title="Top Finalists: Mean Seconds per Epoch Across Seeds",
        output_name="hypothesis2_top_epoch_seconds_band.png",
    )
    plot_accuracy_time_tradeoff(
        per_seed_finalists,
        title="Top Finalists: Accuracy vs Time Trade-off",
        output_name="hypothesis2_accuracy_time_tradeoff.png",
    )


if __name__ == "__main__":
    main()
