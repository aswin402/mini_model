"""Run the data-driven verified System 2 proof benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from little.evaluation.system2_benchmark import evaluate_cases, load_cases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "benchmarks" / "system2_reasoning_cases.json"


def run(cases_path: Path = DEFAULT_CASES) -> dict[str, object]:
    version, cases = load_cases(cases_path)
    report = evaluate_cases(cases)
    return {
        "benchmark": "system2_proof",
        "version": version,
        "report": report.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rendered = json.dumps(run(args.cases), indent=2, sort_keys=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
