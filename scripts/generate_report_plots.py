from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
MULTISEED_DIR = ARTIFACTS_DIR / "multiseed"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def epoch_range(result: dict) -> list[int]:
    return [row["epoch"] for row in result["epoch_logs"]]


def val_curve(result: dict) -> list[float]:
    return [row["val_acc"] for row in result["epoch_logs"]]


def avg_epoch_time(result: dict) -> float:
    return float(result["elapsed_seconds"]) / int(result["epochs"])


def plot_line_chart(results: list[dict], title: str, output_name: str, names: list[str] | None = None) -> None:
    if names is not None:
        name_set = set(names)
        results = [result for result in results if result["name"] in name_set]
        results.sort(key=lambda result: names.index(result["name"]))

    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            epoch_range(result),
            val_curve(result),
            marker="o",
            linewidth=2,
            label=result["name"],
        )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.5, 1.0)
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


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_results(EXPERIMENTS_DIR / "baseline_results.json")
    extended = load_results(EXPERIMENTS_DIR / "extended_results.json")
    hypothesis1 = load_results(EXPERIMENTS_DIR / "hypothesis1_results.json")
    hypothesis2 = load_results(EXPERIMENTS_DIR / "hypothesis2_results.json")
    hypothesis2_final = load_results(EXPERIMENTS_DIR / "hypothesis2_final_results.json")

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


if __name__ == "__main__":
    main()
