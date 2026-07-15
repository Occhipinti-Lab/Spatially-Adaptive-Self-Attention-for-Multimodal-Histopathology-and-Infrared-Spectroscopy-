from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .statistics import significance_label


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {png_path} and {pdf_path}")


def generate_box_plots(
    raw_df: pd.DataFrame,
    output_dir: Path,
    modality: str,
    palette: dict[str, str],
) -> None:
    if raw_df.empty:
        return

    modality_dir = output_dir / modality
    for metric in ["Precision", "Recall", "F1 Score", "Accuracy"]:
        metric_df = raw_df[raw_df["Metric"] == metric]
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))

        if metric == "Accuracy":
            sns.boxplot(
                data=metric_df,
                y="Score",
                color=palette.get(modality),
                linewidth=0.8,
                fliersize=4,
                ax=ax,
            )
            mean_value = metric_df["Score"].mean()
            ax.axhline(mean_value, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_value:.3f}")
            ax.legend(frameon=False)
            ax.set_xlabel("")
        else:
            sns.boxplot(
                data=metric_df,
                x="Class",
                y="Score",
                hue="Class",
                palette="viridis",
                linewidth=0.8,
                fliersize=4,
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Class")
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(f"{modality}: {metric}")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, modality_dir / f"{slugify(modality)}_{slugify(metric)}_boxplot")
        plt.close(fig)


def generate_line_plot(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    palette: dict[str, str],
) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_df["Category"] = plot_df["Category"].apply(lambda value: textwrap.fill(str(value), width=15))

    fig, ax = plt.subplots(figsize=(6, 4))
    lineplot_args = {
        "data": plot_df,
        "x": "Category",
        "y": "Mean",
        "hue": "Modality",
        "style": "Modality",
        "markers": True,
        "dashes": False,
        "palette": palette,
        "linewidth": 1.0,
        "markersize": 6,
        "ax": ax,
    }
    try:
        sns.lineplot(**lineplot_args, errorbar=("ci", 95))
    except (AttributeError, TypeError):
        sns.lineplot(**lineplot_args, ci=95)
    ax.set_title(f"Mean {metric} per Class")
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Modality", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def generate_radar_plot(
    summary_df: pd.DataFrame,
    output_dir: Path,
    palette: dict[str, str],
) -> None:
    plot_df = summary_df[~summary_df["Category"].str.contains("Overall", case=False, na=False)]
    if plot_df.empty:
        return

    metrics = ["Precision", "Recall", "F1 Score"]
    angles = [index / len(metrics) * 2 * np.pi for index in range(len(metrics))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    for modality in plot_df["Modality"].dropna().unique():
        values = []
        for metric in metrics:
            subset = plot_df[(plot_df["Modality"] == modality) & (plot_df["Metric"] == metric)]
            values.append(float(subset["Mean"].mean()) if not subset.empty else 0.0)
        values += values[:1]
        ax.plot(angles, values, linewidth=1.0, label=modality, color=palette.get(modality), marker="o")
        ax.fill(angles, values, alpha=0.2, color=palette.get(modality))

    ax.set_title("Average Class Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "radar_plot")
    plt.close(fig)


def generate_performance_heatmaps(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df[~summary_df["Category"].str.contains("Overall", case=False, na=False)]
    if plot_df.empty:
        return

    metrics = ["Precision", "Recall", "F1 Score"]
    modalities = list(plot_df["Modality"].dropna().unique())
    classes = sorted(plot_df["Category"].dropna().unique())
    wrapped_classes = [textwrap.fill(value, width=12) for value in classes]

    for metric in metrics:
        values = []
        labels = []
        for modality in modalities:
            row_values = []
            row_labels = []
            for class_name in classes:
                subset = plot_df[
                    (plot_df["Modality"] == modality)
                    & (plot_df["Category"] == class_name)
                    & (plot_df["Metric"] == metric)
                ]
                if subset.empty:
                    row_values.append(0.0)
                    row_labels.append("")
                else:
                    mean_value = float(subset["Mean"].iloc[0])
                    std_value = float(subset["StdDev"].iloc[0])
                    row_values.append(mean_value)
                    row_labels.append(f"{mean_value:.3f}\n+/- {std_value:.3f}")

            values.append(row_values)
            labels.append(row_labels)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.heatmap(
            values,
            annot=np.array(labels),
            fmt="",
            cmap="YlOrRd",
            xticklabels=wrapped_classes,
            yticklabels=modalities,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": f"{metric} Score"},
            annot_kws={"size": 9},
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{slugify(metric)}_heatmap_with_std")
        plt.close(fig)


def generate_boxplots_with_significance(
    raw_df: pd.DataFrame,
    t_test_df: pd.DataFrame,
    output_dir: Path,
    modality_order: list[str],
    palette: dict[str, str],
) -> None:
    if raw_df.empty or t_test_df.empty:
        return

    significance_dir = output_dir / "significance_plots"
    accuracy_df = raw_df[raw_df["Metric"] == "Accuracy"]
    if not accuracy_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=accuracy_df,
            x="Modality",
            y="Score",
            hue="Modality",
            order=modality_order,
            palette=palette,
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=accuracy_df,
            x="Modality",
            y="Score",
            order=modality_order,
            color="black",
            alpha=0.6,
            jitter=True,
            size=4,
            ax=ax,
        )
        add_significance_brackets(
            ax=ax,
            tests_df=t_test_df[(t_test_df["Metric"] == "Accuracy") & (t_test_df["Class"] == "Overall")],
            score_df=accuracy_df,
            modality_order=modality_order,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy Score")
        fig.tight_layout()
        save_figure(fig, significance_dir / "accuracy_with_significance")
        plt.close(fig)

    for metric in ["Precision", "Recall", "F1 Score"]:
        metric_df = raw_df[raw_df["Metric"] == metric]
        if metric_df.empty:
            continue

        classes = list(metric_df["Class"].dropna().unique())
        n_cols = min(2, len(classes))
        n_rows = (len(classes) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes_array = np.array(axes).reshape(-1)

        for index, class_name in enumerate(classes):
            ax = axes_array[index]
            class_df = metric_df[metric_df["Class"] == class_name]
            sns.boxplot(
                data=class_df,
                x="Modality",
                y="Score",
                hue="Modality",
                order=modality_order,
                palette=palette,
                linewidth=0.8,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=class_df,
                x="Modality",
                y="Score",
                order=modality_order,
                color="black",
                alpha=0.6,
                jitter=True,
                size=4,
                ax=ax,
            )
            add_group_means(ax, class_df, modality_order)
            add_significance_brackets(
                ax=ax,
                tests_df=t_test_df[(t_test_df["Metric"] == metric) & (t_test_df["Class"] == class_name)],
                score_df=class_df,
                modality_order=modality_order,
            )
            ax.set_title(class_name)
            ax.set_xlabel("")
            ax.set_ylabel(metric)

        for index in range(len(classes), len(axes_array)):
            axes_array[index].set_visible(False)

        fig.tight_layout()
        save_figure(fig, significance_dir / f"{slugify(metric)}_with_significance")
        plt.close(fig)


def generate_significance_plots(t_test_df: pd.DataFrame, output_dir: Path) -> None:
    if t_test_df.empty:
        return

    significance_dir = output_dir / "significance_plots"
    pivot_data = t_test_df.pivot_table(
        values="p_value",
        index=["Metric", "Class"],
        columns=["Modality1", "Modality2"],
        aggfunc="mean",
    )
    if not pivot_data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            -np.log10(pivot_data.clip(lower=1e-300)),
            cmap="viridis",
            annot=True,
            fmt=".2f",
            mask=pivot_data > 0.05,
            cbar_kws={"label": "-log10(p-value)"},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Statistical Significance")
        fig.tight_layout()
        save_figure(fig, significance_dir / "significance_heatmap")
        plt.close(fig)

    significant_df = t_test_df[t_test_df["p_value"] < 0.05].copy()
    if not significant_df.empty:
        significant_df["Modality Pair"] = significant_df["Modality1"] + " vs " + significant_df["Modality2"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=significant_df,
            x="Modality Pair",
            y="Cohen_d",
            hue="Metric",
            palette="viridis",
            ax=ax,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", rotation=90, fontsize=8)
        ax.set_title("Effect Sizes for Significant Results")
        ax.set_xlabel("Modality Pair")
        ax.set_ylabel("Cohen's d")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        save_figure(fig, significance_dir / "effect_sizes")
        plt.close(fig)

    volcano_df = t_test_df.copy()
    volcano_df["neg_log10_p"] = -np.log10(np.maximum(volcano_df["p_value"], 1e-300))
    colors = np.where(volcano_df["p_value"] < 0.05, "red", "gray")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        volcano_df["Cohen_d"],
        volcano_df["neg_log10_p"],
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p = 0.05")
    ax.axvline(x=0, color="black", linewidth=0.8, alpha=0.5)

    labelled_df = volcano_df[(volcano_df["p_value"] < 0.05) & (volcano_df["Cohen_d"].abs() > 0.5)]
    for _, row in labelled_df.iterrows():
        ax.annotate(
            f"{row['Metric']}\n{row['Class']}",
            (row["Cohen_d"], row["neg_log10_p"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_title("Effect Size vs Statistical Significance")
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, significance_dir / "volcano_plot")
    plt.close(fig)


def add_group_means(ax: plt.Axes, class_df: pd.DataFrame, modality_order: list[str]) -> None:
    means = class_df.groupby("Modality")["Score"].mean()
    for index, modality in enumerate(modality_order):
        if modality in means.index:
            ax.scatter(index, means[modality], color="red", marker="D", s=45, alpha=0.8)


def add_significance_brackets(
    ax: plt.Axes,
    tests_df: pd.DataFrame,
    score_df: pd.DataFrame,
    modality_order: list[str],
) -> None:
    if tests_df.empty or score_df.empty:
        return

    y_max = float(score_df["Score"].max())
    y_min = float(score_df["Score"].min())
    y_range = y_max - y_min
    if y_range == 0:
        y_range = max(abs(y_max) * 0.05, 0.01)

    pair_tests: dict[tuple[str, str], list[float]] = {}
    for _, row in tests_df.iterrows():
        pair = tuple(sorted([row["Modality1"], row["Modality2"]], key=modality_order.index))
        pair_tests.setdefault(pair, []).append(float(row["p_value"]))

    sorted_pairs = sorted(
        pair_tests,
        key=lambda pair: abs(modality_order.index(pair[1]) - modality_order.index(pair[0])),
    )

    bar_height = 0.02 * y_range
    start_y = y_max + 0.06 * y_range
    step_y = 0.11 * y_range

    for index, (first_modality, second_modality) in enumerate(sorted_pairs):
        x1 = modality_order.index(first_modality)
        x2 = modality_order.index(second_modality)
        y = start_y + index * step_y
        draw_significance_bracket(
            ax=ax,
            x1=x1,
            x2=x2,
            y=y,
            h=bar_height,
            text=significance_label(min(pair_tests[(first_modality, second_modality)])),
        )

    ax.set_ylim(top=start_y + len(sorted_pairs) * step_y + 0.08 * y_range)


def draw_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="black", clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="center",
        fontsize=12,
        color="darkred",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")
