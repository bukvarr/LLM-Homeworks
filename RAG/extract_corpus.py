"""Извлечение корпуса ТФКП из PDF в corpus.jsonl.

Строка: block_id, page, section, text. page — номер страницы PDF с 1.
block_id стабильны при той же версии PyMuPDF и этом скрипте.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

HEADING = re.compile(
    r"^(?:§\s*\d+|Теорема\s+\d|Определение\s+\d|Следствие\s+\d|"
    r"Замечание\s+\d|Пример\s+\d|Лемма\s+\d|Доказательство\b)"
)
HEADER_LINE = re.compile(
    r"^(?:лекции по тфкп|в\.\s*в\.\s*горяйнов,\s*е\.\s*с\.\s*половинкин)\.?$",
    re.IGNORECASE,
)
HYPHEN_END = re.compile(r"[-‐­]$")


def decode_char(font: str, ch: str) -> str:
    if font.startswith(("CM", "MS", "EU")):
        return ch
    code = ord(ch)
    if code < 256:
        return bytes([code]).decode("cp1251")
    return ch


def overline_indexes(page, chars: list[dict]) -> set[int]:
    """Короткая черта у верхней кромки буквы — это \\bar, не дробная черта.

    Дробная черта лежит над знаменателем, то есть выше его bbox.
    Черта \\bar пересекает верхнюю часть самого глифа.
    """
    bars = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        if rect.height < 1.5 and 2.5 <= rect.width <= 9:
            bars.append(rect)
    marked: set[int] = set()
    for index, char in enumerate(chars):
        box = char["bbox"]
        height = max(box[3] - box[1], 0.1)
        center = (box[0] + box[2]) / 2
        for rect in bars:
            if not (rect.x0 - 1.5 <= center <= rect.x1 + 1.5):
                continue
            if not (box[1] <= rect.y0 <= box[3]):
                continue
            if rect.y0 - box[1] <= 0.5 * height:
                marked.add(index)
                break
    return marked


def page_lines(page) -> list[dict]:
    raw = page.get_text("rawdict")
    chars: list[dict] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                font = span.get("font", "")
                for item in span.get("chars", []):
                    chars.append(
                        {
                            "ch": decode_char(font, item["c"]),
                            "bbox": item["bbox"],
                        }
                    )
    barred = overline_indexes(page, chars)
    for index in barred:
        if chars[index]["ch"].isalpha():
            chars[index]["ch"] += "\u0305"
    chars.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    grouped: list[dict] = []
    for char in chars:
        y0 = char["bbox"][1]
        if grouped and abs(y0 - grouped[-1]["y0"]) <= 2.2:
            grouped[-1]["chars"].append(char)
            grouped[-1]["y0"] = min(grouped[-1]["y0"], y0)
            grouped[-1]["y1"] = max(grouped[-1]["y1"], char["bbox"][3])
        else:
            grouped.append({"y0": y0, "y1": char["bbox"][3], "chars": [char]})
    lines: list[dict] = []
    for group in grouped:
        ordered = sorted(group["chars"], key=lambda item: item["bbox"][0])
        text = ""
        previous_right = None
        for char in ordered:
            if previous_right is not None and char["bbox"][0] - previous_right > 1.6 and not text.endswith(" "):
                text += " "
            text += char["ch"]
            previous_right = char["bbox"][2]
        lines.append(
            {
                "text": text,
                "y0": group["y0"],
                "y1": group["y1"],
                "x0": ordered[0]["bbox"][0],
                "x1": ordered[-1]["bbox"][2],
            }
        )
    return lines


def is_running_header(line: dict) -> bool:
    if line["y0"] < 46:
        return True
    compact = re.sub(r"\s+", " ", line["text"]).strip()
    if HEADER_LINE.match(compact):
        return True
    return bool(re.fullmatch(r"\d{1,3}", compact) and line["y0"] < 60)


def starts_paragraph(line: dict) -> bool:
    text = line["text"].strip()
    if HEADING.match(text):
        return True
    # Первая строка абзаца сдвинута примерно на 55 pt.
    # Продолжение пункта списка сдвинуто сильнее, около 73 pt, и абзац не начинает.
    if 50 <= line["x0"] <= 66 and re.search(r"[А-Яа-яЁё(]", text):
        return True
    return False


def join_lines(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not text:
            text = part
            continue
        if HYPHEN_END.search(text) and re.match(r"[а-яё]", part):
            text = HYPHEN_END.sub("", text) + part
        else:
            text += " " + part
    text = text.replace("\u00ad", "")
    text = text.replace("\x12", "")
    text = text.replace("\x16", "—")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?□]\s*$", text))


def merge_continuations(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for text in paragraphs:
        if not merged:
            merged.append(text)
            continue
        previous = merged[-1]
        if HEADING.match(text):
            merged.append(text)
            continue
        if HYPHEN_END.search(previous) or not ends_sentence(previous) or re.match(r"[а-яё(]", text):
            merged[-1] = join_lines([previous, text])
            continue
        merged.append(text)
    return merged


def page_paragraphs(page) -> list[str]:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in page_lines(page):
        if is_running_header(line):
            continue
        text = re.sub(r"\s+", " ", line["text"]).strip()
        if not text:
            continue
        if current and starts_paragraph(line):
            paragraphs.append(current)
            current = []
        current.append(text)
    if current:
        paragraphs.append(current)
    merged: list[str] = []
    for parts in paragraphs:
        text = join_lines(parts)
        if not text:
            continue
        if merged and len(text) < 80 and not re.search(r"[А-Яа-яЁё]", text):
            merged[-1] = join_lines([merged[-1], text])
        else:
            merged.append(text)
    return merge_continuations(merged)


def section_by_page(toc: list[list], page_count: int) -> dict[int, str]:
    sections = {page: "" for page in range(1, page_count + 1)}
    current = ""
    cursor = 1
    for _level, title, start in toc:
        start = max(1, int(start))
        for page in range(cursor, min(start, page_count + 1)):
            sections[page] = current
        current = re.sub(r"\s+", " ", str(title)).strip()
        cursor = start
    for page in range(cursor, page_count + 1):
        sections[page] = current
    return sections


def extract(pdf_path: Path) -> list[dict]:
    document = pymupdf.open(pdf_path)
    sections = section_by_page(document.get_toc(), document.page_count)
    rows = []
    for index, page in enumerate(document, start=1):
        if index == 1:
            continue
        section = sections.get(index, "")
        paragraphs = page_paragraphs(page)
        if rows and paragraphs and not HEADING.match(paragraphs[0]):
            previous = rows[-1]["text"]
            nxt = paragraphs[0]
            continued = bool(HYPHEN_END.search(previous) or re.match(r"[а-яё]", nxt))
            if continued:
                rows[-1]["text"] = join_lines([previous, nxt])
                paragraphs = paragraphs[1:]
        for text in paragraphs:
            rows.append(
                {
                    "block_id": f"b_{len(rows) + 1:04d}",
                    "page": index,
                    "section": section,
                    "text": text,
                }
            )
    document.close()
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="PDF учебника ТФКП → corpus.jsonl")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF не найден: {args.pdf}")
    rows = extract(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    pages = max(row["page"] for row in rows)
    print(f"blocks={len(rows)} pages={pages} sha256={digest}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
