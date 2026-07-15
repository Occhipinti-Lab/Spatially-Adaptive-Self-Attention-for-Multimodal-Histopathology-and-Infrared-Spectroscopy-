from argparse import ArgumentParser
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from metrics_analysis import AnalysisConfig, run_analysis


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Summarise classification reports and generate analysis figures.")
    parser.add_argument("--input-root", type=Path, default=Path("raw_reports"), help="Directory containing report folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for tables and figures.")
    parser.add_argument("--summary-csv", default="classification_metrics_summary.csv", help="Summary CSV filename.")
    parser.add_argument("--t-test-csv", default="t_test_results.csv", help="T-test CSV filename.")
    parser.add_argument("--descriptive-csv", default="descriptive_statistics.csv", help="Descriptive statistics CSV filename.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    config = AnalysisConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        t_test_csv=args.t_test_csv,
        descriptive_csv=args.descriptive_csv,
    )
    run_analysis(config)


if __name__ == "__main__":
    main()

