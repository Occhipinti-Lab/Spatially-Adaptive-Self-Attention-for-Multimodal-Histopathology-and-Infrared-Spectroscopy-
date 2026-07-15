from pathlib import Path

import pandas as pd

from .statistics import effect_size_label, significance_label


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        print(f"Skipped empty table: {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def save_t_test_results(t_test_df: pd.DataFrame, path: Path) -> None:
    if t_test_df.empty:
        print("No t-test results to save")
        return

    output_df = t_test_df.copy()
    output_df["Significance_Level"] = output_df["p_value"].apply(significance_label)
    output_df["Interpretation"] = output_df["p_value"].apply(
        lambda value: "Significant difference" if value < 0.05 else "No significant difference"
    )
    output_df["Effect_Size_Interpretation"] = output_df["Cohen_d"].apply(effect_size_label)
    output_df = output_df.sort_values("p_value").reset_index(drop=True)

    for column in ["t_stat", "p_value", "Cohen_d"]:
        if column in output_df:
            output_df[column] = output_df[column].round(6)

    save_dataframe(output_df, path)


def write_summary_report(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    path: Path,
) -> None:
    if summary_df.empty:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "CLASSIFICATION METRICS ANALYSIS SUMMARY",
        "=" * 46,
        "",
        f"Modalities analysed: {summary_df['Modality'].nunique()}",
        f"Classes analysed: {count_classes(summary_df)}",
        f"Metrics analysed: {summary_df['Metric'].nunique()}",
        "",
        "BEST PERFORMANCE BY METRIC",
        "-" * 26,
    ]

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        metric_df = summary_df[summary_df["Metric"] == metric]
        if metric_df.empty:
            continue
        best_row = metric_df.loc[metric_df["Mean"].idxmax()]
        lines.append(
            f"{metric}: {best_row['Modality']} "
            f"({best_row['Mean']:.4f} +/- {best_row['StdDev']:.4f})"
        )

    lines.extend(["", "OVERALL SCORE DISTRIBUTION", "-" * 26])

    if not raw_df.empty:
        grouped = raw_df.groupby("Modality")["Score"].agg(["mean", "std", "count"])
        for modality, row in grouped.iterrows():
            lines.append(f"{modality}: {row['mean']:.4f} +/- {row['std']:.4f} (n={int(row['count'])})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def count_classes(summary_df: pd.DataFrame) -> int:
    class_rows = summary_df[~summary_df["Category"].str.contains("Overall", case=False, na=False)]
    return int(class_rows["Category"].nunique())

