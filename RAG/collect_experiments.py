"""Сводит пять прогонов в experiments.csv и блок отчёта."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import read_jsonl
from evaluate import score_run

ORDER = (
    ("no_context", False),
    ("single", True),
    ("final", True),
    ("no_rerank", True),
    ("no_dense", True),
)
COLUMNS = [
    "name",
    "answer_accuracy",
    "Q",
    "mean_input_tokens",
    "mean_output_tokens",
    "mean_llm_calls",
    "mean_cost",
    "median_latency_ms",
]
BEGIN = "<!-- BEGIN_GENERATED -->"
END = "<!-- END_GENERATED -->"


def fmt(value, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def worst_cases(metrics: dict, gold_rows: list[dict], answers: list[dict], traces: list[dict], corpus: dict[str, dict]) -> list[str]:
    gold = {row["id"]: row for row in gold_rows}
    predicted = {row["id"]: row for row in answers}
    traced = {row["id"]: row for row in traces}
    ranked = sorted(metrics["per_question"], key=lambda item: (item["s"], item["id"]))
    lines = []
    taken = 0
    for item in ranked:
        if item["s"] >= 1 or taken >= 3:
            continue
        taken += 1
        truth = gold[item["id"]]
        guess = predicted.get(item["id"], {})
        retrieved = (traced.get(item["id"]) or {}).get("retrieved_block_ids") or []
        quotes = []
        for block_id in retrieved[:4]:
            block = corpus.get(block_id)
            if block:
                quotes.append(f"{block_id}: {block['text'][:180].replace(chr(10), ' ')}")
        if item["A"] == 0 and truth["abstain"]:
            reason = "Модель не отказалась, хотя условие не задаёт единственное значение."
        elif item["A"] == 0 and guess.get("abstain"):
            reason = "Модель отказалась, хотя фрагмент с нужной формулировкой есть в корпусе."
        elif item["A"] == 0:
            reason = "Набор вариантов не совпал с эталоном."
        else:
            reason = "Варианты верны, но набор ссылок не совпал ни с одним допустимым."
        lines.append(
            f"- **{item['id']}**. Эталон: abstain={truth['abstain']}, варианты {truth['answer_ids']}. "
            f"Ответ: abstain={guess.get('abstain')}, варианты {guess.get('answer_ids')}, "
            f"ссылки {guess.get('citations')}. {reason} "
            + ("Найденные фрагменты: " + " | ".join(quotes) if quotes else "Поиск не вернул фрагменты.")
        )
    if not lines:
        lines.append("- На открытом наборе все 15 вопросов получили полный балл, трёх ошибок нет.")
    return lines


def ablation_sentence(title: str, full: float, other: float) -> str:
    delta = full - other
    if delta > 0.01:
        return f"Отключение {title} снизило Q на {delta:.3f}: на этом наборе компонент полезен."
    if delta < -0.01:
        return (
            f"Отключение {title} повысило Q на {-delta:.3f}. На 15 вопросах компонент не окупился: "
            "он меняет порядок уже найденных фрагментов и может вытеснить короткую формулировку теоремы."
        )
    return f"Отключение {title} изменило Q на {delta:.3f}, то есть почти не повлияло."


def generated_section(rows: dict[str, dict], notes: list[str], stats: dict | None) -> str:
    final = rows["final"]["Q"]
    parts = [
        "### Результаты прогона",
        "",
        ablation_sentence("reranker", final, rows["no_rerank"]["Q"]),
        ablation_sentence("dense-поиск", final, rows["no_dense"]["Q"]),
        "",
        "### Три ошибки итоговой системы",
        "",
        *notes,
        "",
    ]
    if stats:
        parts.extend(
            [
                "### Сборка индекса",
                "",
                f"Сборка заняла {stats['elapsed_sec']} с, пиковая RAM {stats['peak_ram_mb']} МБ, "
                f"размер индекса {stats['index_bytes']} байт. Платформа: {stats['platform']}, "
                f"процессор: {stats['processor']}, всего RAM {stats['ram_total_gb']} ГБ. "
                f"Блоков {stats['blocks']}, дочерних окон {stats['children']}.",
                "",
            ]
        )
    else:
        parts.extend(["### Сборка индекса", "", "Файл index/stats.json рядом с отчётом не найден.", ""])
    return "\n".join(parts)


def replace_report(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"В {path} нет маркеров {BEGIN} и {END}")
    before, rest = text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    path.write_text(before + BEGIN + "\n" + section.rstrip() + "\n" + END + after, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Собрать experiments.csv и раздел отчёта")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()
    gold_rows = read_jsonl(args.gold)
    corpus = {}
    if args.corpus and args.corpus.is_file():
        corpus = {row["block_id"]: row for row in read_jsonl(args.corpus)}
    table = []
    stored = {}
    for name, full_q in ORDER:
        answers = read_jsonl(args.runs / f"{name}_answers.jsonl")
        trace = read_jsonl(args.runs / f"{name}_trace.jsonl")
        metrics = score_run(gold_rows, answers, trace, require_citations=full_q)
        stored[name] = metrics
        table.append(
            {
                "name": name,
                "answer_accuracy": fmt(metrics["answer_accuracy"], 4),
                "Q": fmt(metrics["Q"], 4),
                "mean_input_tokens": fmt(metrics["mean_input_tokens"], 1),
                "mean_output_tokens": fmt(metrics["mean_output_tokens"], 1),
                "mean_llm_calls": fmt(metrics["mean_llm_calls"], 2),
                "mean_cost": fmt(metrics["mean_cost"], 1),
                "median_latency_ms": fmt(metrics["median_latency_ms"], 1),
            }
        )
        print(name, table[-1]["answer_accuracy"], table[-1]["Q"], "efficiency_zero" if metrics["efficiency_zero"] else "ok")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    final_answers = read_jsonl(args.runs / "final_answers.jsonl")
    final_trace = read_jsonl(args.runs / "final_trace.jsonl")
    notes = worst_cases(stored["final"], gold_rows, final_answers, final_trace, corpus)
    stats = json.loads(args.stats.read_text(encoding="utf-8")) if args.stats and args.stats.is_file() else None
    replace_report(args.report, generated_section(stored, notes, stats))
    print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
