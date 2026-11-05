"""Оценка открытого набора по правилам задания."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from common import LIMITS, RATIONALE_LIMIT, f1, read_jsonl

CATEGORIES = ("conditions", "comparison", "application", "insufficient")


def cost_of(trace: dict) -> float:
    return trace["input_tokens"] + 4 * trace["output_tokens"] + 256 * trace["llm_calls"]


def over_limit(trace: dict) -> bool:
    return (
        trace["llm_calls"] > LIMITS["llm_calls"]
        or trace["input_tokens"] > LIMITS["input_tokens"]
        or trace["output_tokens"] > LIMITS["output_tokens"]
    )


def format_ok(row: dict, require_citations: bool) -> bool:
    if not isinstance(row.get("abstain"), bool):
        return False
    answers = row.get("answer_ids")
    citations = row.get("citations")
    rationale = row.get("rationale")
    if not isinstance(answers, list) or not isinstance(citations, list) or not isinstance(rationale, str):
        return False
    if rationale.strip() == "" or len(rationale) > RATIONALE_LIMIT:
        return False
    if len(answers) != len(set(answers)) or len(citations) != len(set(citations)):
        return False
    if len(citations) > 4:
        return False
    if row["abstain"]:
        return answers == [] and citations == []
    if not answers:
        return False
    if require_citations and not citations:
        return False
    return True


def option_match(predicted: dict, gold: dict) -> int:
    if gold["abstain"]:
        return int(predicted["abstain"] and predicted["answer_ids"] == [])
    return int((not predicted["abstain"]) and set(predicted["answer_ids"]) == set(gold["answer_ids"]))


def question_score(predicted: dict, gold: dict, trace: dict, require_citations: bool) -> dict:
    if over_limit(trace) or not format_ok(predicted, require_citations):
        return {"A": 0, "s": 0.0, "F": 0.0}
    matched = option_match(predicted, gold)
    if gold["abstain"]:
        return {"A": matched, "s": float(matched), "F": 0.0}
    if not matched:
        return {"A": 0, "s": 0.0, "F": 0.0}
    best = max((f1(predicted["citations"], item) for item in gold["citation_sets"]), default=0.0)
    return {"A": 1, "s": 0.8 + 0.2 * best, "F": best}


def score_run(gold_rows: list[dict], answers: list[dict], traces: list[dict], require_citations: bool) -> dict:
    by_answer = {row["id"]: row for row in answers}
    by_trace = {row["id"]: row for row in traces}
    per_question = []
    by_category: dict[str, list[float]] = {name: [] for name in CATEGORIES}
    for gold in gold_rows:
        predicted = by_answer.get(gold["id"])
        trace = by_trace.get(gold["id"])
        if predicted is None or trace is None:
            item = {"id": gold["id"], "category": gold["category"], "A": 0, "s": 0.0, "F": 0.0, "missing": True}
        else:
            scored = question_score(predicted, gold, trace, require_citations)
            item = {"id": gold["id"], "category": gold["category"], **scored, "missing": False}
        per_question.append(item)
        by_category[gold["category"]].append(item["s"])
    category_means = {
        name: (sum(values) / len(values) if values else None) for name, values in by_category.items()
    }
    present = [value for value in category_means.values() if value is not None]
    macro = sum(present) / len(present) if present else 0.0
    accuracy_values = [item["A"] for item in per_question]
    used = [by_trace[item["id"]] for item in per_question if item["id"] in by_trace]
    def mean(field: str) -> float:
        if not used:
            return 0.0
        return sum(row[field] for row in used) / len(used)
    latencies = sorted(row["elapsed_ms"] for row in used)
    median = statistics.median(latencies) if latencies else 0
    return {
        "answer_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0,
        "Q": None if not require_citations else macro,
        "category_means": category_means,
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_llm_calls": mean("llm_calls"),
        "mean_cost": sum(cost_of(row) for row in used) / len(used) if used else 0.0,
        "median_latency_ms": median,
        "efficiency_zero": any(over_limit(row) for row in used),
        "n": len(gold_rows),
        "per_question": per_question,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Оценка answers/trace по gold")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    metrics = score_run(
        read_jsonl(args.gold),
        read_jsonl(args.answers),
        read_jsonl(args.trace),
        require_citations=not args.accuracy_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    q_text = "" if metrics["Q"] is None else f"{metrics['Q']:.4f}"
    print(
        f"accuracy={metrics['answer_accuracy']:.4f} Q={q_text} "
        f"cost={metrics['mean_cost']:.1f} calls={metrics['mean_llm_calls']:.2f}"
    )


if __name__ == "__main__":
    main()
