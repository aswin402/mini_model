"""Run the explicit deterministic-versus-hybrid perception benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from little.evaluation.hybrid_benchmark import (
    build_adapter,
    evaluate_adapter,
    load_cases,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "benchmarks" / "hybrid_perception_cases.json"


def run(
    *,
    mode: str,
    cases_path: Path = DEFAULT_CASES,
    model_id_or_path: str | None = None,
    device: str | None = None,
    subfolder: str | None = None,
    runtime: str | None = None,
    model_version: str | None = None,
    package_version: str | None = None,
) -> dict[str, object]:
    benchmark_version, cases = load_cases(cases_path)
    selected_modes = (
        ("deterministic", "hybrid-laya") if mode == "both" else (mode,)
    )
    reports = []
    for selected_mode in selected_modes:
        adapter = build_adapter(
            selected_mode,
            model_id_or_path=model_id_or_path,
            device=device,
            subfolder=subfolder,
            runtime=runtime,
            model_version=model_version,
            package_version=package_version,
        )
        reports.append(evaluate_adapter(selected_mode, adapter, cases).to_dict())
    return {
        "benchmark": "hybrid_perception",
        "version": benchmark_version,
        "mode": mode,
        "reports": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("deterministic", "hybrid-laya", "both"),
        default="deterministic",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--model-id-or-path",
        help="Explicit local Laya model identifier or checkpoint path.",
    )
    parser.add_argument("--device")
    parser.add_argument("--subfolder")
    parser.add_argument("--runtime", choices=("agent", "router"))
    parser.add_argument("--model-version")
    parser.add_argument("--package-version")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run(
            mode=args.mode,
            cases_path=args.cases,
            model_id_or_path=args.model_id_or_path,
            device=args.device,
            subfolder=args.subfolder,
            runtime=args.runtime,
            model_version=args.model_version,
            package_version=args.package_version,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
