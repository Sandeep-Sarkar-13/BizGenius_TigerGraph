#!/usr/bin/env python3
"""
TRACE benchmark runner.

Expected project structure:

TRACE/
├── agentic_graphrag_local.py
├── requirements_agentic_graphrag.txt
├── .env
├── data/
│   └── corpus.jsonl
├── benchmarks/
│   ├── public_eval.jsonl
│   └── benchmark.py
└── results/
    └── trace_results.json

This benchmark evaluates the public evaluation set by calling the local
TRACE Agentic GraphRAG runner.

The script supports common public-eval JSONL field names:
    question / query
    answer / gold_answer / expected_answer
    qtype / question_type / type

If your evaluation file contains only questions and no gold answers,
the script will still run TRACE and save predictions, but exact accuracy
cannot be calculated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "benchmarks" / "public_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "trace_results.json"
AGENT_FILE = ROOT / "agentic_graphrag_local.py"


def load_agent_module():
    """Load agentic_graphrag_local.py from the project root."""
    if not AGENT_FILE.exists():
        raise FileNotFoundError(
            f"TRACE runner not found: {AGENT_FILE}\n"
            "Place agentic_graphrag_local.py in the project root."
        )

    spec = importlib.util.spec_from_file_location(
        "agentic_graphrag_local",
        AGENT_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load agentic_graphrag_local.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} of {path} is not a JSON object."
                )

            rows.append(obj)

    return rows


def first_value(row: dict[str, Any], keys: list[str], default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def extract_question(row: dict[str, Any]) -> str:
    question = first_value(
        row,
        ["question", "query", "prompt", "input"],
    )

    if question is None:
        raise ValueError(
            f"Could not find a question field. Available fields: {list(row)}"
        )

    return str(question).strip()


def extract_gold(row: dict[str, Any]):
    return first_value(
        row,
        [
            "answer",
            "gold_answer",
            "expected_answer",
            "reference_answer",
            "gold",
            "target",
        ],
    )


def extract_qtype(row: dict[str, Any]) -> str:
    value = first_value(
        row,
        [
            "qtype",
            "question_type",
            "type",
            "category",
        ],
        default="unknown",
    )
    return str(value).strip().lower()


def normalize_answer(value: Any) -> str:
    """
    Conservative normalization for exact-match evaluation.

    This intentionally does not use an LLM to judge correctness.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    # Remove markdown emphasis/backticks.
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\*+", "", text)

    # Normalize common Unicode punctuation.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove a final sentence-ending punctuation mark.
    text = re.sub(r"[.!?]+$", "", text)

    return text.strip()


def extract_prediction(result: Any) -> str:
    """
    Extract the final answer from different TRACE runner return formats.

    The benchmark first checks common dictionary fields, then falls back
    to string conversion.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in [
            "answer",
            "final_answer",
            "response",
            "output",
            "prediction",
            "result",
        ]:
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        # Some TRACE versions store the answer in state.
        state = result.get("state")

        if isinstance(state, dict):
            for key in [
                "answer",
                "final_answer",
                "answer_candidate",
                "response",
            ]:
                value = state.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

    return str(result).strip()


def run_trace(agent, question: str) -> Any:
    """
    Call the local TRACE runner.

    Preferred public interface:
        graph = graph_rag(question)
        trace_run_agentic_graphrag_v5(question)

    The benchmark prefers the final TRACE V5 runner if available.
    """
    candidates = [
        "trace_run_agentic_graphrag_v5",
        "trace_run_agentic_graphrag_v4",
        "trace_run_agentic_graphrag_v3",
        "trace_run_agentic_graphrag_v2",
        "trace_run_agentic_graphrag",
    ]

    for name in candidates:
        fn = getattr(agent, name, None)

        if callable(fn):
            return fn(question)

    # Fallback for the cleaned local script if it exposes graph_rag().
    fn = getattr(agent, "graph_rag", None)

    if callable(fn):
        return fn(question)

    raise AttributeError(
        "No TRACE runner was found in agentic_graphrag_local.py. "
        f"Expected one of: {', '.join(candidates)} or graph_rag."
    )


def safe_route(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None

    for key in [
        "route",
        "initial_route",
        "strategy",
        "question_type",
    ]:
        value = result.get(key)
        if value:
            return str(value)

    state = result.get("state")

    if isinstance(state, dict):
        for key in [
            "route",
            "initial_route",
            "strategy",
            "question_type",
        ]:
            value = state.get(key)
            if value:
                return str(value)

    return None


def safe_steps(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None

    for key in ["steps", "step_count", "num_steps"]:
        value = result.get(key)
        if isinstance(value, int):
            return value

    state = result.get("state")

    if isinstance(state, dict):
        for key in ["steps", "step_count", "num_steps"]:
            value = state.get(key)
            if isinstance(value, int):
                return value

        history = state.get("tool_history")
        if isinstance(history, list):
            return len(history)

    history = result.get("tool_history")
    if isinstance(history, list):
        return len(history)

    return None


def evaluate(
    agent,
    rows: list[dict[str, Any]],
    limit: int | None = None,
    delay: float = 0.0,
) -> dict[str, Any]:

    if limit is not None:
        rows = rows[:limit]

    results = []

    total = len(rows)
    exact_count = 0
    evaluated_count = 0

    for index, row in enumerate(rows, start=1):
        question = extract_question(row)
        gold = extract_gold(row)
        qtype = extract_qtype(row)

        print(f"\n[{index}/{total}] {question}")

        started = time.perf_counter()

        try:
            raw = run_trace(agent, question)
            prediction = extract_prediction(raw)

            error = None

        except Exception as exc:
            raw = None
            prediction = ""
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started

        gold_norm = normalize_answer(gold)
        pred_norm = normalize_answer(prediction)

        exact = None

        if gold is not None:
            evaluated_count += 1
            exact = pred_norm == gold_norm

            if exact:
                exact_count += 1

        item = {
            "id": first_value(
                row,
                ["id", "question_id", "qid"],
                default=index,
            ),
            "question": question,
            "qtype": qtype,
            "gold_answer": gold,
            "prediction": prediction,
            "exact_match": exact,
            "latency_seconds": round(elapsed, 4),
            "route": safe_route(raw),
            "steps": safe_steps(raw),
            "error": error,
        }

        results.append(item)

        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  Prediction: {prediction}")
            if gold is not None:
                print(f"  Gold:       {gold}")
                print(f"  Exact:      {exact}")

        if delay > 0:
            time.sleep(delay)

    # Overall metrics.
    metrics: dict[str, Any] = {
        "total_questions": total,
        "questions_with_gold": evaluated_count,
        "exact_matches": exact_count,
        "exact_match_accuracy": (
            exact_count / evaluated_count
            if evaluated_count
            else None
        ),
    }

    # Per-question-type metrics.
    by_type: dict[str, dict[str, Any]] = {}

    for item in results:
        qtype = item["qtype"]

        if qtype not in by_type:
            by_type[qtype] = {
                "total": 0,
                "evaluated": 0,
                "correct": 0,
            }

        by_type[qtype]["total"] += 1

        if item["exact_match"] is not None:
            by_type[qtype]["evaluated"] += 1

            if item["exact_match"]:
                by_type[qtype]["correct"] += 1

    for qtype, stats in by_type.items():
        evaluated = stats["evaluated"]

        stats["accuracy"] = (
            stats["correct"] / evaluated
            if evaluated
            else None
        )

    metrics["by_question_type"] = by_type

    # Error count.
    metrics["errors"] = sum(
        1 for item in results if item["error"] is not None
    )

    return {
        "benchmark": "TRACE Agentic GraphRAG Public Evaluation",
        "metrics": metrics,
        "results": results,
    }


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]

    print("\n" + "=" * 70)
    print("TRACE BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"Total questions:       {metrics['total_questions']}")
    print(f"Questions with gold:   {metrics['questions_with_gold']}")
    print(f"Exact matches:         {metrics['exact_matches']}")

    accuracy = metrics["exact_match_accuracy"]

    if accuracy is None:
        print("Exact-match accuracy:  N/A")
    else:
        print(f"Exact-match accuracy:  {accuracy:.4f}")

    print(f"Errors:                {metrics['errors']}")

    print("\nPer question type:")
    print(
        f"{'Type':<18}"
        f"{'Total':>8}"
        f"{'Correct':>10}"
        f"{'Accuracy':>12}"
    )

    print("-" * 50)

    for qtype, stats in sorted(
        metrics["by_question_type"].items()
    ):
        acc = stats["accuracy"]

        acc_text = (
            f"{acc:.4f}"
            if acc is not None
            else "N/A"
        )

        print(
            f"{qtype:<18}"
            f"{stats['total']:>8}"
            f"{stats['correct']:>10}"
            f"{acc_text:>12}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the local TRACE Agentic GraphRAG."
    )

    parser.add_argument(
        "--eval",
        type=Path,
        default=DEFAULT_EVAL,
        help="Public evaluation JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between questions in seconds.",
    )

    args = parser.parse_args()

    eval_path = args.eval.resolve()
    output_path = args.output.resolve()

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_path}"
        )

    print("Loading TRACE Agentic GraphRAG...")
    agent = load_agent_module()

    print(f"Evaluation file: {eval_path}")
    print(f"Output file:     {output_path}")

    rows = load_jsonl(eval_path)

    print(f"Loaded {len(rows)} evaluation questions.")

    report = evaluate(
        agent=agent,
        rows=rows,
        limit=args.limit,
        delay=args.delay,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(report)

    print(
        f"\nDetailed results saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()
