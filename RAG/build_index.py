"""Сборка BM25 и векторного индекса. Вызовов LLM нет."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from common import load_config, read_jsonl, windows


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def peak_ram_mb() -> float:
    info = psutil.Process().memory_full_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return round(peak / (1024 * 1024), 1)


def build_children(blocks: list[dict], size: int, stride: int) -> list[dict]:
    children = []
    for block in blocks:
        for piece in windows(block["text"], size, stride):
            children.append(
                {
                    "child_id": f"c_{len(children):05d}",
                    "parent_id": block["block_id"],
                    "text": piece,
                }
            )
    return children


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Индекс корпуса RAG")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    retrieval = config["retrieval"]
    blocks = read_jsonl(args.corpus)
    required = {"block_id", "page", "section", "text"}
    for row in blocks:
        missing = required - set(row)
        if missing:
            raise SystemExit(f"В корпусе нет полей {sorted(missing)}")
    if len({row["block_id"] for row in blocks}) != len(blocks):
        raise SystemExit("block_id в корпусе должны быть уникальны")
    children = build_children(blocks, int(retrieval["child_chars"]), int(retrieval["child_stride"]))
    from embed import Encoder

    encoder = Encoder(config["embedding_model"], config["embedding_revision"])
    prefixed = ["passage: " + child["text"] for child in children]
    matrix = encoder.encode(prefixed, int(retrieval["batch_size"]))
    args.output.mkdir(parents=True, exist_ok=True)
    corpus_bytes = args.corpus.read_bytes()
    meta = {
        "embedding_model": config["embedding_model"],
        "embedding_revision": config["embedding_revision"],
        "embedding_parameters": config["embedding_parameters"],
        "reranker_model": config["reranker_model"],
        "reranker_revision": config["reranker_revision"],
        "reranker_parameters": config["reranker_parameters"],
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "retrieval": retrieval,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "blocks": len(blocks),
        "children": len(children),
    }
    (args.output / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "blocks.json").write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    (args.output / "children.json").write_text(json.dumps(children, ensure_ascii=False), encoding="utf-8")
    np.save(args.output / "embeddings.npy", matrix)
    elapsed = round(time.perf_counter() - started, 1)
    stats = {
        "elapsed_sec": elapsed,
        "peak_ram_mb": peak_ram_mb(),
        "index_bytes": directory_bytes(args.output),
        "blocks": len(blocks),
        "children": len(children),
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cpu_only": True,
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"index={args.output} blocks={len(blocks)} children={len(children)} "
        f"sec={elapsed} peak_ram_mb={stats['peak_ram_mb']} bytes={stats['index_bytes']}"
    )


if __name__ == "__main__":
    main()
