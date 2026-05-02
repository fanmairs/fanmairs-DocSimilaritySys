from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from engines.traditional.system import PlagiarismDetectorSystem  # noqa: E402


DEFAULT_SAMPLE_DIR = ROOT / "experiments" / "threshold_sensitivity_samples_final"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "threshold_sensitivity_results"


@dataclass(frozen=True)
class SampleCase:
    case_id: str
    sample_type: str
    label: int
    target_file: Path
    reference_file: Path
    domain: str


@dataclass(frozen=True)
class ThresholdSetting:
    threshold_id: str
    experiment_stage: str
    window_threshold: float
    fine_threshold: float
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run threshold sensitivity experiments for traditional mode."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory containing labels.csv, threshold_grid.csv and samples/.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=Path,
        default=None,
        help="CSV with experiment_stage, window_threshold and fine_threshold columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for result CSV files. Defaults to a timestamped run directory.",
    )
    parser.add_argument(
        "--window-thresholds",
        default=None,
        help="Comma-separated window thresholds. Example: 0.04,0.06,0.08,0.1",
    )
    parser.add_argument(
        "--fine-thresholds",
        default=None,
        help="Comma-separated fine trigger thresholds. Example: 0.2,0.25,0.3,0.35",
    )
    parser.add_argument("--lsa-components", type=int, default=None)
    parser.add_argument("--semantic-threshold", type=float, default=None)
    parser.add_argument("--semantic-weight", type=float, default=None)
    parser.add_argument("--min-window-chars", type=int, default=40)
    parser.add_argument("--body-mode", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--limit-thresholds", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Run the detector separately for every threshold setting. Slower.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    return [float(item) for item in items]


def load_cases(sample_dir: Path, limit: Optional[int]) -> List[SampleCase]:
    labels_path = sample_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    cases: List[SampleCase] = []
    for row in read_csv_rows(labels_path):
        target_file = sample_dir / row["target_file"]
        reference_file = sample_dir / row["reference_file"]
        if not target_file.is_file():
            raise FileNotFoundError(f"target file not found: {target_file}")
        if not reference_file.is_file():
            raise FileNotFoundError(f"reference file not found: {reference_file}")
        cases.append(
            SampleCase(
                case_id=row["case_id"],
                sample_type=row["sample_type"],
                label=int(row["label"]),
                target_file=target_file,
                reference_file=reference_file,
                domain=row.get("domain", ""),
            )
        )

    if limit is not None:
        cases = cases[: max(0, limit)]
    return cases


def load_thresholds(args: argparse.Namespace) -> List[ThresholdSetting]:
    custom_window_thresholds = parse_float_list(args.window_thresholds)
    custom_fine_thresholds = parse_float_list(args.fine_thresholds)
    thresholds: List[ThresholdSetting] = []

    if custom_window_thresholds is not None or custom_fine_thresholds is not None:
        window_values = custom_window_thresholds or [0.08]
        fine_values = custom_fine_thresholds or [0.3]
        for window_threshold in window_values:
            for fine_threshold in fine_values:
                thresholds.append(
                    ThresholdSetting(
                        threshold_id=f"T{len(thresholds) + 1:03d}",
                        experiment_stage="custom",
                        window_threshold=window_threshold,
                        fine_threshold=fine_threshold,
                        description="Command-line threshold setting.",
                    )
                )
    else:
        grid_path = args.threshold_grid or (args.sample_dir / "threshold_grid.csv")
        if not grid_path.is_file():
            raise FileNotFoundError(f"threshold grid not found: {grid_path}")
        for row in read_csv_rows(grid_path):
            thresholds.append(
                ThresholdSetting(
                    threshold_id=f"T{len(thresholds) + 1:03d}",
                    experiment_stage=row.get("experiment_stage", ""),
                    window_threshold=float(row["window_threshold"]),
                    fine_threshold=float(row["fine_threshold"]),
                    description=row.get("description", ""),
                )
            )

    if args.limit_thresholds is not None:
        thresholds = thresholds[: max(0, args.limit_thresholds)]
    return thresholds


def build_system(
    setting: ThresholdSetting,
    args: argparse.Namespace,
    *,
    verbose: bool,
) -> PlagiarismDetectorSystem:
    settings = get_settings()
    kwargs = {
        "stopwords_path": settings.stopwords_path,
        "lsa_components": args.lsa_components or settings.lsa_components,
        "synonyms_path": settings.synonyms_path,
        "semantic_embeddings_path": settings.semantic_embeddings_path,
        "semantic_threshold": (
            args.semantic_threshold
            if args.semantic_threshold is not None
            else settings.semantic_threshold
        ),
        "semantic_weight": (
            args.semantic_weight
            if args.semantic_weight is not None
            else settings.semantic_weight
        ),
        "window_threshold": setting.window_threshold,
        "fine_trigger_threshold": setting.fine_threshold,
        "min_window_chars": args.min_window_chars,
    }
    if verbose:
        return PlagiarismDetectorSystem(**kwargs)
    with redirect_stdout(io.StringIO()):
        return PlagiarismDetectorSystem(**kwargs)


def run_detector(
    system: PlagiarismDetectorSystem,
    case: SampleCase,
    *,
    body_mode: bool,
    verbose: bool,
) -> List[Dict[str, object]]:
    if verbose:
        return system.check_similarity(
            str(case.target_file),
            [str(case.reference_file)],
            body_mode=body_mode,
        )
    with redirect_stdout(io.StringIO()):
        return system.check_similarity(
            str(case.target_file),
            [str(case.reference_file)],
            body_mode=body_mode,
        )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify(label: int, detected: int) -> str:
    if label == 1 and detected == 1:
        return "TP"
    if label == 0 and detected == 1:
        return "FP"
    if label == 1 and detected == 0:
        return "FN"
    return "TN"


def build_raw_row(
    *,
    setting: ThresholdSetting,
    case: SampleCase,
    result: Dict[str, object],
    elapsed: float,
    error: str,
    sample_dir: Path,
    min_window_chars: int,
    simulate_window_filter: bool,
) -> Dict[str, object]:
    sim_tfidf = safe_float(result.get("sim_tfidf"))
    sim_lsa = safe_float(result.get("sim_lsa"))
    sim_soft = safe_float(result.get("sim_soft"))
    sim_hybrid = safe_float(result.get("sim_hybrid"))
    risk_score = safe_float(result.get("risk_score"))
    coarse_max_score = max(sim_tfidf, sim_lsa, sim_soft, sim_hybrid, risk_score)
    fine_triggered = 1 if coarse_max_score > setting.fine_threshold else 0

    raw_parts = result.get("plagiarized_parts") or []
    if not isinstance(raw_parts, list):
        raw_parts = []
    if not fine_triggered:
        parts = []
    elif simulate_window_filter:
        parts = [
            part
            for part in raw_parts
            if isinstance(part, dict)
            and safe_float(part.get("score")) > setting.window_threshold
            and int(part.get("length", 0) or 0) > min_window_chars
        ]
    else:
        parts = [part for part in raw_parts if isinstance(part, dict)]

    part_scores = [safe_float(part.get("score")) for part in parts]
    part_lengths = [int(part.get("length", 0) or 0) for part in parts]
    hit_count = len(parts)
    detected = 1 if hit_count > 0 else 0
    max_part_score = max(part_scores, default=0.0)
    avg_part_score = statistics.mean(part_scores) if part_scores else 0.0
    max_part_length = max(part_lengths, default=0)

    return {
        "threshold_id": setting.threshold_id,
        "experiment_stage": setting.experiment_stage,
        "window_threshold": setting.window_threshold,
        "fine_threshold": setting.fine_threshold,
        "case_id": case.case_id,
        "sample_type": case.sample_type,
        "domain": case.domain,
        "label": case.label,
        "detected": detected,
        "outcome": classify(case.label, detected),
        "fine_triggered": fine_triggered,
        "hit_count": hit_count,
        "max_part_score": round(max_part_score, 6),
        "avg_part_score": round(avg_part_score, 6),
        "max_part_length": max_part_length,
        "sim_tfidf": round(sim_tfidf, 6),
        "sim_lsa": round(sim_lsa, 6),
        "sim_soft": round(sim_soft, 6),
        "sim_hybrid": round(sim_hybrid, 6),
        "risk_score": round(risk_score, 6),
        "elapsed_seconds": round(elapsed, 4),
        "target_file": str(case.target_file.relative_to(sample_dir)),
        "reference_file": str(case.reference_file.relative_to(sample_dir)),
        "error": error,
    }


def run_cached_experiment(
    cases: List[SampleCase],
    thresholds: List[ThresholdSetting],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    min_window_threshold = min(setting.window_threshold for setting in thresholds)
    min_fine_threshold = min(setting.fine_threshold for setting in thresholds)
    base_setting = ThresholdSetting(
        threshold_id="BASE",
        experiment_stage="base",
        window_threshold=min_window_threshold,
        fine_threshold=min_fine_threshold,
        description="Base run used for threshold filtering.",
    )
    system = build_system(base_setting, args, verbose=args.verbose)
    rows: List[Dict[str, object]] = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] BASE "
            f"window={min_window_threshold:g} fine={min_fine_threshold:g} "
            f"case={case.case_id}"
        )
        started = time.perf_counter()
        error = ""
        result: Dict[str, object] = {}
        try:
            results = run_detector(
                system,
                case,
                body_mode=args.body_mode,
                verbose=args.verbose,
            )
            result = dict(results[0]) if results else {}
        except Exception as exc:  # pragma: no cover - recorded in CSV for diagnosis
            error = repr(exc)
        elapsed = time.perf_counter() - started

        for setting in thresholds:
            rows.append(
                build_raw_row(
                    setting=setting,
                    case=case,
                    result=result,
                    elapsed=elapsed,
                    error=error,
                    sample_dir=args.sample_dir,
                    min_window_chars=args.min_window_chars,
                    simulate_window_filter=True,
                )
            )

    return rows


def run_experiment(
    cases: List[SampleCase],
    thresholds: List[ThresholdSetting],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if not args.exact:
        return run_cached_experiment(cases, thresholds, args)

    rows: List[Dict[str, object]] = []
    total = len(cases) * len(thresholds)
    done = 0

    for setting in thresholds:
        system = build_system(setting, args, verbose=args.verbose)
        for case in cases:
            done += 1
            print(
                f"[{done}/{total}] {setting.threshold_id} "
                f"window={setting.window_threshold:g} fine={setting.fine_threshold:g} "
                f"case={case.case_id}"
            )

            started = time.perf_counter()
            error = ""
            result: Dict[str, object] = {}
            try:
                results = run_detector(
                    system,
                    case,
                    body_mode=args.body_mode,
                    verbose=args.verbose,
                )
                result = dict(results[0]) if results else {}
            except Exception as exc:  # pragma: no cover - recorded in CSV for diagnosis
                error = repr(exc)
            elapsed = time.perf_counter() - started

            parts = result.get("plagiarized_parts") or []
            if not isinstance(parts, list):
                parts = []
            part_scores = [safe_float(part.get("score")) for part in parts if isinstance(part, dict)]
            part_lengths = [int(part.get("length", 0) or 0) for part in parts if isinstance(part, dict)]
            hit_count = len(parts)
            detected = 1 if hit_count > 0 else 0
            max_part_score = max(part_scores, default=0.0)
            avg_part_score = statistics.mean(part_scores) if part_scores else 0.0
            max_part_length = max(part_lengths, default=0)

            sim_tfidf = safe_float(result.get("sim_tfidf"))
            sim_lsa = safe_float(result.get("sim_lsa"))
            sim_soft = safe_float(result.get("sim_soft"))
            sim_hybrid = safe_float(result.get("sim_hybrid"))
            risk_score = safe_float(result.get("risk_score"))
            coarse_max_score = max(sim_tfidf, sim_lsa, sim_soft, sim_hybrid, risk_score)
            fine_triggered = 1 if coarse_max_score > setting.fine_threshold else 0

            rows.append(
                {
                    "threshold_id": setting.threshold_id,
                    "experiment_stage": setting.experiment_stage,
                    "window_threshold": setting.window_threshold,
                    "fine_threshold": setting.fine_threshold,
                    "case_id": case.case_id,
                    "sample_type": case.sample_type,
                    "domain": case.domain,
                    "label": case.label,
                    "detected": detected,
                    "outcome": classify(case.label, detected),
                    "fine_triggered": fine_triggered,
                    "hit_count": hit_count,
                    "max_part_score": round(max_part_score, 6),
                    "avg_part_score": round(avg_part_score, 6),
                    "max_part_length": max_part_length,
                    "sim_tfidf": round(sim_tfidf, 6),
                    "sim_lsa": round(sim_lsa, 6),
                    "sim_soft": round(sim_soft, 6),
                    "sim_hybrid": round(sim_hybrid, 6),
                    "risk_score": round(risk_score, 6),
                    "elapsed_seconds": round(elapsed, 4),
                    "target_file": str(case.target_file.relative_to(args.sample_dir)),
                    "reference_file": str(case.reference_file.relative_to(args.sample_dir)),
                    "error": error,
                }
            )

    return rows


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(raw_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        groups[str(row["threshold_id"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for threshold_id, rows in groups.items():
        first = rows[0]
        tp = sum(1 for row in rows if row["outcome"] == "TP")
        fp = sum(1 for row in rows if row["outcome"] == "FP")
        fn = sum(1 for row in rows if row["outcome"] == "FN")
        tn = sum(1 for row in rows if row["outcome"] == "TN")
        positives = tp + fn
        negatives = tn + fp
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        accuracy = divide(tp + tn, len(rows))
        false_positive_rate = divide(fp, negatives)
        false_negative_rate = divide(fn, positives)
        hit_counts = [int(row["hit_count"]) for row in rows]
        max_scores = [safe_float(row["max_part_score"]) for row in rows]

        summary_rows.append(
            {
                "threshold_id": threshold_id,
                "experiment_stage": first["experiment_stage"],
                "window_threshold": first["window_threshold"],
                "fine_threshold": first["fine_threshold"],
                "sample_count": len(rows),
                "positive_count": positives,
                "negative_count": negatives,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "accuracy": round(accuracy, 6),
                "false_positive_rate": round(false_positive_rate, 6),
                "false_negative_rate": round(false_negative_rate, 6),
                "avg_hit_count": round(statistics.mean(hit_counts), 6),
                "avg_max_part_score": round(statistics.mean(max_scores), 6),
                "error_count": sum(1 for row in rows if row.get("error")),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -safe_float(row["f1"]),
            safe_float(row["false_positive_rate"]),
            -safe_float(row["recall"]),
            safe_float(row["window_threshold"]),
            safe_float(row["fine_threshold"]),
        )
    )
    return summary_rows


def write_report(output_dir: Path, summary_rows: List[Dict[str, object]]) -> None:
    report_path = output_dir / "experiment_report.md"
    best = summary_rows[0] if summary_rows else None
    lines = [
        "# 阈值敏感性实验报告",
        "",
        "本报告由 `tools/run_threshold_sensitivity.py` 自动生成。",
        "",
    ]
    if best:
        lines.extend(
            [
                "## F1 最优阈值组合",
                "",
                f"- 阈值编号：{best['threshold_id']}",
                f"- 窗口命中阈值：{best['window_threshold']}",
                f"- 细粒度触发阈值：{best['fine_threshold']}",
                f"- 准确率 precision：{best['precision']}",
                f"- 召回率 recall：{best['recall']}",
                f"- F1 值：{best['f1']}",
                f"- 误报率 false_positive_rate：{best['false_positive_rate']}",
                "",
                "该结果可作为阈值设置的样本实验依据，但不是绝对证明。"
                "结论会受到当前样本文档数量、领域分布和标注方式的影响。",
                "",
            ]
        )
    lines.extend(
        [
            "## 输出文件说明",
            "",
            "- `raw_results.csv`：每一行对应一组样本文档对和一组阈值的检测结果。",
            "- `summary_by_threshold.csv`：按阈值组合汇总后的 precision、recall、F1、accuracy 等指标。",
            "- `experiment_report.md`：便于阅读的实验摘要。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_output_dir(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / stamp


def main() -> int:
    args = parse_args()
    args.sample_dir = args.sample_dir.resolve()
    output_dir = resolve_output_dir(args.output_dir).resolve()

    cases = load_cases(args.sample_dir, args.limit_cases)
    thresholds = load_thresholds(args)
    if not cases:
        raise RuntimeError("No sample cases loaded.")
    if not thresholds:
        raise RuntimeError("No threshold settings loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sample cases: {len(cases)}")
    print(f"Threshold settings: {len(thresholds)}")
    print(f"Run mode: {'exact' if args.exact else 'cached threshold filtering'}")
    print(f"Output directory: {output_dir}")

    raw_rows = run_experiment(cases, thresholds, args)
    summary_rows = summarize(raw_rows)

    raw_fields = [
        "threshold_id",
        "experiment_stage",
        "window_threshold",
        "fine_threshold",
        "case_id",
        "sample_type",
        "domain",
        "label",
        "detected",
        "outcome",
        "fine_triggered",
        "hit_count",
        "max_part_score",
        "avg_part_score",
        "max_part_length",
        "sim_tfidf",
        "sim_lsa",
        "sim_soft",
        "sim_hybrid",
        "risk_score",
        "elapsed_seconds",
        "target_file",
        "reference_file",
        "error",
    ]
    summary_fields = [
        "threshold_id",
        "experiment_stage",
        "window_threshold",
        "fine_threshold",
        "sample_count",
        "positive_count",
        "negative_count",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "avg_hit_count",
        "avg_max_part_score",
        "error_count",
    ]

    write_csv_rows(output_dir / "raw_results.csv", raw_rows, raw_fields)
    write_csv_rows(output_dir / "summary_by_threshold.csv", summary_rows, summary_fields)
    write_report(output_dir, summary_rows)

    print("Done.")
    if summary_rows:
        best = summary_rows[0]
        print(
            "Best by F1: "
            f"{best['threshold_id']} "
            f"window={best['window_threshold']} "
            f"fine={best['fine_threshold']} "
            f"precision={best['precision']} "
            f"recall={best['recall']} "
            f"f1={best['f1']}"
        )
    print(f"Raw results: {output_dir / 'raw_results.csv'}")
    print(f"Summary: {output_dir / 'summary_by_threshold.csv'}")
    print(f"Report: {output_dir / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
