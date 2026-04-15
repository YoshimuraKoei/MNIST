import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row: dict, key: str) -> float:
    return float(row[key])


def to_int(row: dict, key: str) -> int:
    return int(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pattern", required=True, help="Glob pattern for per-seed summary csv files")
    parser.add_argument("--output-prefix", required=True, help="Prefix path for output csv/json")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_pattern}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for row in load_rows(path):
            row = dict(row)
            row["source_file"] = str(path)
            grouped[row["name"]].append(row)

    summary_rows = []
    details = {}
    for model_name, rows in sorted(grouped.items()):
        val_accs = [to_float(row, "best_val_acc") for row in rows]
        test_accs = [to_float(row, "test_acc") for row in rows]
        val_losses = [to_float(row, "best_val_loss") for row in rows]
        test_losses = [to_float(row, "test_loss") for row in rows]
        params = [to_int(row, "parameter_count") for row in rows]
        epochs = [to_int(row, "epochs") for row in rows]

        summary_rows.append(
            {
                "name": model_name,
                "runs": len(rows),
                "mean_val_acc": f"{mean(val_accs):.6f}",
                "std_val_acc": f"{pstdev(val_accs):.6f}",
                "mean_test_acc": f"{mean(test_accs):.6f}",
                "std_test_acc": f"{pstdev(test_accs):.6f}",
                "mean_val_loss": f"{mean(val_losses):.6f}",
                "mean_test_loss": f"{mean(test_losses):.6f}",
                "parameter_count": params[0],
                "epochs": epochs[0],
            }
        )
        details[model_name] = rows

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary_rows, "details": details}, file, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
