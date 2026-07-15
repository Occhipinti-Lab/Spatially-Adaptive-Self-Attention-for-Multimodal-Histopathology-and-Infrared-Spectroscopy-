import re
from pathlib import Path


CLASS_METRIC_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_ ./()+-]+?)\s{2,}"
    r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+\d+"
)
ACCURACY_PATTERN = re.compile(r"^\s*accuracy\s+.*?\s+([\d.]+)\s+\d+")
AVERAGE_ROWS = {"micro avg", "macro avg", "weighted avg"}


def parse_classification_report(file_content: str) -> dict[str, float] | None:
    metrics: dict[str, float] = {}

    for line in file_content.strip().splitlines():
        stripped_line = line.strip()
        if any(label in stripped_line for label in AVERAGE_ROWS):
            continue

        class_match = CLASS_METRIC_PATTERN.match(line)
        if class_match:
            class_name = class_match.group(1).strip().replace(" ", "_")
            metrics[f"{class_name}_precision"] = float(class_match.group(2))
            metrics[f"{class_name}_recall"] = float(class_match.group(3))
            metrics[f"{class_name}_f1-score"] = float(class_match.group(4))
            continue

        accuracy_match = ACCURACY_PATTERN.match(line)
        if accuracy_match:
            metrics["accuracy"] = float(accuracy_match.group(1))

    return metrics or None


def find_report_files(reports_dir: Path) -> list[Path]:
    if not reports_dir.is_dir():
        return []

    return sorted(reports_dir.glob("*.txt"), key=lambda path: natural_sort_key(path.name))


def load_reports(reports_dir: Path, modality: str) -> list[dict[str, float | str]]:
    report_paths = find_report_files(reports_dir)
    records: list[dict[str, float | str]] = []

    if not report_paths:
        print(f"No report files found for {modality}: {reports_dir}")
        return records

    for report_path in report_paths:
        try:
            parsed_metrics = parse_classification_report(report_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            parsed_metrics = parse_classification_report(report_path.read_text(encoding="latin-1"))
        except OSError as exc:
            print(f"Could not read {report_path}: {exc}")
            continue

        if parsed_metrics is None:
            print(f"Could not parse {report_path}")
            continue

        parsed_metrics["_file"] = report_path.name
        records.append(parsed_metrics)

    print(f"{modality}: parsed {len(records)} report files")
    return records


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

