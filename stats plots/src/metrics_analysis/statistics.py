import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(values: Iterable[float]) -> tuple[float, float, float, float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean_value = float(np.mean(values))
    std_dev = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    median_value = float(np.median(values))
    return mean_value, std_dev, median_value, float(min(values)), float(max(values))


def create_summary_dataframe(
    reports_data: list[dict[str, float | str]],
    modality: str,
) -> pd.DataFrame:
    if not reports_data:
        return pd.DataFrame()

    grouped_metrics: dict[str, list[float]] = {}
    for report_data in reports_data:
        for metric_name, value in report_data.items():
            if metric_name == "_file":
                continue
            grouped_metrics.setdefault(metric_name, []).append(float(value))

    rows = []
    for metric_name, values in grouped_metrics.items():
        mean_value, std_dev, median_value, min_value, max_value = descriptive_stats(values)
        count = len(values)
        std_error = std_dev / np.sqrt(count) if count > 1 else 0.0
        confidence_interval = 1.96 * std_error if count > 1 else 0.0

        if metric_name == "accuracy":
            category = "Overall"
            metric = "Accuracy"
        else:
            category, metric = metric_name.rsplit("_", 1)
            category = category.replace("_", " ").title()
            metric = metric.replace("-", " ").title()

        rows.append(
            {
                "Modality": modality,
                "Category": category,
                "Metric": metric,
                "Mean": round(mean_value, 4),
                "StdDev": round(std_dev, 4),
                "StdError": round(std_error, 4),
                "ConfidenceInterval": round(confidence_interval, 4),
                "Median": round(median_value, 4),
                "Min": round(min_value, 4),
                "Max": round(max_value, 4),
                "Count": count,
                "Range": round(max_value - min_value, 4),
            }
        )

    return pd.DataFrame(rows)


def create_raw_metrics_dataframe(
    reports_data: list[dict[str, float | str]],
    modality: str,
) -> pd.DataFrame:
    rows = []

    for run_index, report_data in enumerate(reports_data, start=1):
        file_name = str(report_data.get("_file", f"report_{run_index}"))

        for metric_name, value in report_data.items():
            if metric_name == "_file":
                continue

            if metric_name == "accuracy":
                rows.append(
                    {
                        "Modality": modality,
                        "Run": run_index,
                        "File": file_name,
                        "Class": "Overall",
                        "Metric": "Accuracy",
                        "Score": float(value),
                    }
                )
                continue

            if "_" not in metric_name:
                continue

            class_name, metric = metric_name.rsplit("_", 1)
            metric = metric.replace("-", " ").title()
            if metric not in {"Precision", "Recall", "F1 Score"}:
                continue

            rows.append(
                {
                    "Modality": modality,
                    "Run": run_index,
                    "File": file_name,
                    "Class": class_name.replace("_", " ").title(),
                    "Metric": metric,
                    "Score": float(value),
                }
            )

    return pd.DataFrame(rows)


def perform_t_tests_across_modalities(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty or raw_df["Modality"].nunique() < 2:
        return pd.DataFrame()

    results = []
    modalities = list(raw_df["Modality"].dropna().unique())
    modality_pairs = list(itertools.combinations(modalities, 2))

    for metric in raw_df["Metric"].dropna().unique():
        metric_df = raw_df[raw_df["Metric"] == metric]
        for class_name in metric_df["Class"].dropna().unique():
            class_df = metric_df[metric_df["Class"] == class_name]

            for first_modality, second_modality in modality_pairs:
                first_df = class_df[class_df["Modality"] == first_modality].sort_values("Run")
                second_df = class_df[class_df["Modality"] == second_modality].sort_values("Run")

                first_scores = first_df["Score"].to_numpy()
                second_scores = second_df["Score"].to_numpy()
                if len(first_scores) < 2 or len(second_scores) < 2:
                    continue

                first_runs = first_df["Run"].to_numpy()
                second_runs = second_df["Run"].to_numpy()
                if len(first_scores) == len(second_scores) and np.array_equal(first_runs, second_runs):
                    t_statistic, p_value = stats.ttest_rel(first_scores, second_scores)
                    test_type = "paired"
                else:
                    t_statistic, p_value = stats.ttest_ind(
                        first_scores,
                        second_scores,
                        equal_var=False,
                    )
                    test_type = "independent"

                p_value = float(max(p_value, 1e-300)) if not np.isnan(p_value) else 1.0
                cohens_d = calculate_cohens_d(first_scores, second_scores)

                results.append(
                    {
                        "Metric": metric,
                        "Class": class_name,
                        "Modality1": first_modality,
                        "Modality2": second_modality,
                        "t_stat": float(t_statistic) if not np.isnan(t_statistic) else 0.0,
                        "p_value": p_value,
                        "Test_Type": test_type,
                        "Cohen_d": cohens_d,
                        "n1": len(first_scores),
                        "n2": len(second_scores),
                    }
                )

    return pd.DataFrame(results)


def calculate_cohens_d(first_scores: np.ndarray, second_scores: np.ndarray) -> float:
    n1 = len(first_scores)
    n2 = len(second_scores)
    if n1 < 2 or n2 < 2:
        return 0.0

    variance_1 = np.var(first_scores, ddof=1)
    variance_2 = np.var(second_scores, ddof=1)
    denominator = n1 + n2 - 2
    if denominator <= 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * variance_1 + (n2 - 1) * variance_2) / denominator)
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0

    return float((np.mean(first_scores) - np.mean(second_scores)) / pooled_std)


def calculate_descriptive_statistics(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    stats_df = (
        raw_df.groupby(["Modality", "Class", "Metric"])["Score"]
        .agg(["mean", "std", "median", "min", "max", "count", "sem"])
        .round(4)
        .reset_index()
    )
    stats_df = stats_df.rename(
        columns={
            "mean": "Score_mean",
            "std": "Score_std",
            "median": "Score_median",
            "min": "Score_min",
            "max": "Score_max",
            "count": "Score_count",
            "sem": "Score_sem",
        }
    )

    stats_df["Score_range"] = stats_df["Score_max"] - stats_df["Score_min"]
    stats_df["Score_cv"] = (stats_df["Score_std"] / stats_df["Score_mean"]) * 100
    stats_df["Score_ci_lower"] = stats_df["Score_mean"] - (1.96 * stats_df["Score_sem"])
    stats_df["Score_ci_upper"] = stats_df["Score_mean"] + (1.96 * stats_df["Score_sem"])
    return stats_df.round(4)


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def effect_size_label(cohens_d: float) -> str:
    absolute_value = abs(cohens_d)
    if absolute_value >= 0.8:
        return "Large"
    if absolute_value >= 0.5:
        return "Medium"
    if absolute_value >= 0.2:
        return "Small"
    return "Negligible"

