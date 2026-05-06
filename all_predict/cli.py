"""Command line interface for all_predict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .classification import AllClassifier
from .exceptions import DataValidationError, InvalidTaskError
from .regression import AllRegressor
from .utils import infer_task_from_target, parse_model_list, require_known_task

SUBCOMMANDS = {"classify", "regress", "infer"}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="all-predict",
        description="Compare many classification or regression models against a tabular dataset.",
    )
    parser.add_argument(
        "--task",
        choices=["classify", "classification", "regress", "regression", "infer", "auto"],
        help="Legacy task selector when not using subcommands.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command, help_text in (
        ("classify", "Run classification comparisons."),
        ("regress", "Run regression comparisons."),
        ("infer", "Infer the task from the target column."),
    ):
        subparser = subparsers.add_parser(command, help=help_text, description=help_text)
        _add_common_arguments(subparser)

    return parser


def build_legacy_parser() -> argparse.ArgumentParser:
    """Build the legacy `--task` parser."""

    parser = argparse.ArgumentParser(
        prog="all-predict",
        description="Compare many classification or regression models against a tabular dataset.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["classify", "classification", "regress", "regression", "infer", "auto"],
        help="Task selector for the legacy flat CLI mode.",
    )
    _add_common_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_legacy_parser() if _should_use_legacy_parser(argv) else build_parser()
    args = parser.parse_args(argv)

    try:
        task = _resolve_requested_task(args)
        frame = pd.read_csv(args.file)
        if args.target not in frame.columns:
            raise DataValidationError(
                f"Target column '{args.target}' was not found in {args.file}."
            )

        y = frame[args.target]
        X = frame.drop(columns=[args.target])
        if task == "infer":
            task = infer_task_from_target(y)

        output_dir = (
            Path(args.output) if args.output else Path("runs") / f"{Path(args.file).stem}_{task}"
        )
        runner = _build_runner(args, task, output_dir)
        results, predictions = runner.fit(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

        if results.empty:
            print("No models completed successfully.", file=sys.stderr)
            return 1

        print(results.head(min(10, len(results))).to_string(index=False))
        print(f"\nResults written to {output_dir}")
        if args.predictions and not predictions.empty:
            print(f"Predictions saved to {output_dir / 'predictions.csv'}")
        if not runner.skipped_models_.empty:
            print(f"Skipped models: {len(runner.skipped_models_)}")
        if not runner.failed_models_.empty:
            print(f"Failed models: {len(runner.failed_models_)}")
        return 0
    except (DataValidationError, InvalidTaskError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


def _should_use_legacy_parser(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in SUBCOMMANDS:
        return False
    return any(argument == "--task" or argument.startswith("--task=") for argument in argv)


def _resolve_requested_task(args: argparse.Namespace) -> str:
    if args.command:
        return require_known_task(args.command)
    if args.task:
        return require_known_task(args.task)
    raise DataValidationError(
        "A subcommand or --task is required. Use classify, regress, or infer."
    )


def _build_runner(args: argparse.Namespace, task: str, output_dir: Path):
    common_kwargs = {
        "verbose": args.verbose,
        "ignore_warnings": args.ignore_warnings,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "sort_by": args.sort_by,
        "predictions": args.predictions,
        "include_models": parse_model_list(args.include_models),
        "exclude_models": parse_model_list(args.exclude_models),
        "max_models": args.max_models,
        "preprocess": True,
        "tune": args.tune,
        "tune_top_n": args.tune_top_n,
        "tuner": args.tuner,
        "cv": args.cv,
        "timeout": args.timeout,
        "save_best": args.save_best,
        "output_dir": output_dir,
        "progress": args.progress,
        "fail_fast": args.fail_fast,
    }
    if task == "classification":
        return AllClassifier(**common_kwargs)
    if task == "regression":
        return AllRegressor(**common_kwargs)
    raise InvalidTaskError(f"Unsupported task: {task}")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", required=True, help="CSV file containing the dataset.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of rows to reserve for testing."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed used for splits and estimators."
    )
    parser.add_argument("--sort-by", default=None, help="Metric used to sort the results table.")
    parser.add_argument(
        "--include-models", default=None, help="Comma-separated list of model names to include."
    )
    parser.add_argument(
        "--exclude-models", default=None, help="Comma-separated list of model names to exclude."
    )
    parser.add_argument(
        "--tune", action=argparse.BooleanOptionalAction, default=False, help="Tune the top models."
    )
    parser.add_argument("--tune-top-n", type=int, default=3, help="How many top models to tune.")
    parser.add_argument(
        "--tuner", choices=["grid", "randomized"], default="randomized", help="Search strategy."
    )
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for tuning.")
    parser.add_argument(
        "--predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-model predictions for the test split.",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist the best fitted model to disk.",
    )
    parser.add_argument(
        "--output", default=None, help="Directory used for CSV, JSON, and model artifacts."
    )
    parser.add_argument(
        "--ignore-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress estimator warnings during fitting.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress and summary output.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs value used where supported.")
    parser.add_argument(
        "--timeout", type=float, default=None, help="Best-effort runtime budget in seconds."
    )
    parser.add_argument(
        "--max-models", type=int, default=None, help="Limit the number of attempted models."
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep progress logging enabled.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise immediately when a model fails instead of recording the failure.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
