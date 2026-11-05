"""Общие функции индекса и оценки. Генеративных вызовов здесь нет."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r"\d+(?:\.\d+)?|[A-Za-zА-Яа-яЁё]+(?:\u0305)?", re.UNICODE)
STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "к", "у", "о", "об", "от", "из", "для",
    "по", "что", "как", "а", "но", "или", "же", "бы", "то", "это", "этот",
    "эта", "эти", "при", "до", "над", "под", "между", "также", "того",
}
LIMITS = {
    "llm_calls": 4,
    "input_tokens": 16384,
    "output_tokens": 4096,
    "request_budget": 8192,
}
RATIONALE_LIMIT = 1200


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def resolve_from(config: dict, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    config = read_json(path)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def tokenize(text: str) -> list[str]:
    normalized = text.replace("z\u0305", " zbar ").replace("Z\u0305", " zbar ")
    normalized = normalized.lower().replace("ё", "е")
    tokens = []
    for token in TOKEN.findall(normalized):
        token = token.replace("\u0305", "")
        if token in STOPWORDS or len(token) == 1:
            continue
        tokens.append(token)
    return tokens


class BM25:
    """Okapi BM25, idf как в rank_bm25: log(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.n = len(documents)
        self.lengths = [len(doc) for doc in documents]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(doc))
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        result = [0.0] * self.n
        if not query or not self.n:
            return result
        for index, doc in enumerate(self.documents):
            if not doc:
                continue
            counts = Counter(doc)
            norm = self.k1 * (1.0 - self.b + self.b * self.lengths[index] / self.avgdl)
            score = 0.0
            for term in query:
                freq = counts.get(term)
                if not freq:
                    continue
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1.0) / (freq + norm)
            result[index] = score
        return result

    def top(self, query: list[str], k: int) -> list[int]:
        values = self.scores(query)
        ranked = sorted(range(self.n), key=lambda index: values[index], reverse=True)
        return [index for index in ranked if values[index] > 0][:k]


def rrf(rank_lists: list[list[str]], c: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for listing in rank_lists:
        seen: set[str] = set()
        for rank, doc_id in enumerate(listing, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (c + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def f1(predicted: list[str], gold: list[str]) -> float:
    pred, truth = set(predicted), set(gold)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    overlap = len(pred & truth)
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def windows(text: str, size: int, stride: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += stride
    return chunks
