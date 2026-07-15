import pandas as pd

from .config import AnalysisConfig, configure_matplotlib
from .parser import load_reports
from .plots import (
    generate_box_plots,
    generate_boxplots_with_significance,
    generate_line_plot,
    generate_performance_heatmaps,
    generate_radar_plot,
    generate_significance_plots,
)
from .reporting import save_dataframe, save_t_test_results, write_summary_report
from .statistics import (
    calculate_descriptive_statistics,
    create_raw_metrics_dataframe,
    create_summary_dataframe,
    perform_t_tests_across_modalities,
)


def run_analysis(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    configure_matplotlib()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    raw_frames = []

    for modality, relative_dir in config.modalities.items():
        reports_dir = config.input_root / relative_dir
        reports_data = load_reports(reports_dir, modality)
        if not reports_data:
            continue

        summary_df = create_summary_dataframe(reports_data, modality)
        raw_df = create_raw_metrics_dataframe(reports_data, modality)

        if not summary_df.empty:
            summary_frames.append(summary_df)
        if not raw_df.empty:
            raw_frames.append(raw_df)
            generate_box_plots(raw_df, config.output_dir, modality, config.palette)

    if not summary_frames or not raw_frames:
        print("No valid classification reports were available for analysis")
        return {
            "summary": pd.DataFrame(),
            "raw": pd.DataFrame(),
            "t_tests": pd.DataFrame(),
            "descriptive": pd.DataFrame(),
        }

    summary_df = pd.concat(summary_frames, ignore_index=True)
    raw_df = pd.concat(raw_frames, ignore_index=True)

    save_dataframe(summary_df, config.output_dir / config.summary_csv)

    class_metrics_df = summary_df[
        summary_df["Metric"].str.lower().isin(["precision", "recall", "f1 score"])
    ]
    for metric in ["Precision", "Recall", "F1 Score"]:
        metric_df = class_metrics_df[class_metrics_df["Metric"] == metric]
        generate_line_plot(
            df=metric_df,
            output_path=config.output_dir / f"mean_{metric.lower().replace(' ', '_')}",
            metric=metric,
            palette=config.palette,
        )

    generate_radar_plot(summary_df, config.output_dir, config.palette)
    generate_performance_heatmaps(summary_df, config.output_dir)

    descriptive_df = calculate_descriptive_statistics(raw_df)
    save_dataframe(descriptive_df, config.output_dir / config.descriptive_csv)

    t_test_df = perform_t_tests_across_modalities(raw_df)
    save_t_test_results(t_test_df, config.output_dir / config.t_test_csv)

    modality_order = list(config.modalities.keys())
    generate_boxplots_with_significance(raw_df, t_test_df, config.output_dir, modality_order, config.palette)
    generate_significance_plots(t_test_df, config.output_dir)
    write_summary_report(summary_df, raw_df, config.output_dir / config.report_name)

    print(f"Analysis complete. Results saved to {config.output_dir}")
    return {
        "summary": summary_df,
        "raw": raw_df,
        "t_tests": t_test_df,
        "descriptive": descriptive_df,
    }

