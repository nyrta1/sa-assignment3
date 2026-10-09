"""Command-line interface: ``oulad-risk {run,audit,synth}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from oulad_risk import __version__
from oulad_risk.config import MODEL_CHOICES, ExperimentConfig
from oulad_risk.data import load_tables
from oulad_risk.features import build_features
from oulad_risk.split import audit_overlap, group_split, naive_stratified_split


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oulad-risk",
        description="Leakage-resistant early academic risk prediction on OULAD.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train and evaluate a model, write metrics and figures")
    run.add_argument("--data-dir", required=True, help="folder with the OULAD CSV files")
    run.add_argument("--out-dir", default="reports", help="where to write results")
    run.add_argument("--model", choices=MODEL_CHOICES, default="rf")
    run.add_argument("--cutoff", type=int, default=30, help="prediction day (default 30)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--min-auc",
        type=float,
        default=None,
        help="quality gate: exit with code 1 if test ROC-AUC is below this value",
    )

    audit = sub.add_parser("audit", help="compare student overlap of naive vs group split")
    audit.add_argument("--data-dir", required=True)
    audit.add_argument("--cutoff", type=int, default=30)
    audit.add_argument("--seed", type=int, default=42)

    synth = sub.add_parser("synth", help="write a small synthetic OULAD-shaped dataset")
    synth.add_argument("--out-dir", required=True)
    synth.add_argument("--students", type=int, default=600)
    synth.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "synth":
        from oulad_risk.synthetic import write

        print(f"Synthetic data written to {write(args.out_dir, args.students, args.seed)}")
        return 0

    tables = load_tables(args.data_dir)

    if args.command == "audit":
        fs = build_features(tables, args.cutoff)
        for name, (tr, te) in {
            "naive row-level split": naive_stratified_split(fs.y, 0.2, args.seed),
            "group (student) split": group_split(fs.groups, 0.2, args.seed),
        }.items():
            rep = audit_overlap(fs.groups, tr, te)
            print(f"{name:24s} shared students={rep.n_shared_students:6d}  "
                  f"leaked test rows={rep.leaked_row_share:6.2%}")  # fmt: skip
        return 0

    from oulad_risk.pipeline import run_experiment  # heavy import only when needed

    config = ExperimentConfig(cutoff_day=args.cutoff, model=args.model, random_state=args.seed)
    results = run_experiment(tables, config, args.out_dir)
    test = results["metrics"]["test_tuned_threshold"]
    print(json.dumps({"data": results["data"], "test_tuned_threshold": test}, indent=2))
    if args.min_auc is not None and test["roc_auc"] < args.min_auc:
        print(f"QUALITY GATE FAILED: ROC-AUC {test['roc_auc']} < {args.min_auc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
