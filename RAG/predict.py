"""Ответы RAG. Параметры генерации читаются из runtime.json и не подменяются."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from common import BM25, LIMITS, RATIONALE_LIMIT, load_config, read_json, read_jsonl, rrf, tokenize, write_jsonl

SYSTEMS = {
    "no_context": {"bm25": False, "dense": False, "rerank": False, "neighbor": False, "adaptive": False},
    "single": {"bm25": True, "dense": False, "rerank": False, "neighbor": False, "adaptive": False},
    "final": {"bm25": True, "dense": True, "rerank": True, "neighbor": True, "adaptive": True},
    "no_rerank": {"bm25": True, "dense": True, "rerank": False, "neighbor": True, "adaptive": True},
    "no_dense": {"bm25": True, "dense": False, "rerank": True, "neighbor": True, "adaptive": True},
}

ANSWER_RULES = """Верни один JSON-объект и ничего больше. Ключи:
"decision": "answer" или "search",
"search_query": строка, иначе "",
"answer_ids": список ID вариантов,
"abstain": true или false,
"rationale": обоснование короче 900 символов,
"citations": от 1 до 4 block_id из фрагментов при abstain=false, иначе [].
Несколько вариантов могут быть верны. Бери ID только из вопроса.
Если фрагменты подтверждают ответ, decision="answer", abstain=false, citations — минимальный набор.
Если из условия нельзя определить единственное значение (не заданы константа, ориентация, точный радиус или значение в точке), сразу decision="answer", abstain=true и оба списка пустые. Напиши, какого сведения не хватает.
Не добавляй факт, которого нет во фрагментах."""

SEARCH_RULE = """decision="search" допустим только если нужной формулировки нет во фрагментах, но она может быть в другой главе этого учебника. Тогда answer_ids и citations оставь пустыми, а search_query сделай коротким."""

ANSWER_ONLY = """decision обязан быть "answer". Повторный поиск недоступен."""

NO_CONTEXT = """Фрагментов учебника нет. Верни JSON с decision="answer" и citations=[].
Выбери все верные варианты по формулировке вопроса. Если однозначное значение из условия не следует, abstain=true и answer_ids=[].
rationale короче 900 символов."""


def load_index(path: Path) -> dict:
    meta = read_json(path / "meta.json")
    blocks = json.loads((path / "blocks.json").read_text(encoding="utf-8"))
    children = json.loads((path / "children.json").read_text(encoding="utf-8"))
    matrix = np.load(path / "embeddings.npy")
    return {"meta": meta, "blocks": blocks, "children": children, "matrix": matrix}


def query_text(question: dict) -> str:
    options = "\n".join(f"{item['id']}. {item['text']}" for item in question["options"])
    return question["question"].strip() + "\n" + options


def parent_lists(child_indexes: list[int], children: list[dict]) -> list[str]:
    ordered = []
    seen = set()
    for index in child_indexes:
        parent = children[index]["parent_id"]
        if parent in seen:
            continue
        seen.add(parent)
        ordered.append(parent)
    return ordered


class Retriever:
    def __init__(self, index: dict, config: dict, system: dict):
        self.blocks = {row["block_id"]: row for row in index["blocks"]}
        self.order = [row["block_id"] for row in index["blocks"]]
        self.children = index["children"]
        self.matrix = index["matrix"]
        self.meta = index["meta"]
        self.retrieval = config["retrieval"]
        self.system = system
        documents = [tokenize(child["text"]) for child in self.children]
        self.bm25 = BM25(documents, float(self.retrieval["bm25_k1"]), float(self.retrieval["bm25_b"]))
        self.encoder = None
        self.reranker = None
        if system["dense"]:
            self._check_meta(config)
            from embed import Encoder

            self.encoder = Encoder(config["embedding_model"], config["embedding_revision"])
        if system["rerank"]:
            self._check_meta(config)
            from embed import Reranker

            self.reranker = Reranker(config["reranker_model"], config["reranker_revision"])

    def _check_meta(self, config: dict) -> None:
        meta = self.meta
        if meta["embedding_model"] != config["embedding_model"] or meta["embedding_revision"] != config["embedding_revision"]:
            raise SystemExit("Индекс собран другой embedding-моделью. Пересоберите build_index.py.")
        if meta["reranker_model"] != config["reranker_model"] or meta["reranker_revision"] != config["reranker_revision"]:
            raise SystemExit("Индекс записан с другим reranker. Пересоберите build_index.py.")

    def search(self, text: str) -> list[dict]:
        retrieval = self.retrieval
        lists = []
        query_tokens = tokenize(text)
        if self.system["bm25"]:
            lists.append(parent_lists(self.bm25.top(query_tokens, int(retrieval["bm25_top_k"])), self.children))
        if self.system["dense"] and self.encoder is not None:
            vector = self.encoder.encode([self.meta["query_prefix"] + text[:2000]], 1)[0]
            scores = self.matrix @ vector
            top = np.argsort(-scores)[: int(retrieval["dense_top_k"])]
            lists.append(parent_lists([int(item) for item in top if scores[item] > 0], self.children))
        if not lists:
            return []
        fused = rrf(lists, int(retrieval["rrf_c"])) if len(lists) > 1 else lists[0]
        if self.system["rerank"] and self.reranker is not None and fused:
            pool = fused[: int(retrieval["rerank_candidates"])]
            passages = [self.blocks[item]["text"][:1800] for item in pool]
            values = self.reranker.scores(text[:1800], passages, int(retrieval["batch_size"]))
            order = sorted(range(len(pool)), key=lambda index: values[index], reverse=True)
            fused = [pool[index] for index in order] + [item for item in fused if item not in pool]
        chosen = fused[: int(retrieval["context_k"])]
        if self.system["neighbor"] and chosen:
            chosen = self._with_neighbor(chosen)
        excerpts = []
        for block_id in chosen:
            block = self.blocks[block_id]
            excerpts.append(
                {
                    "block_id": block_id,
                    "page": block["page"],
                    "section": block["section"],
                    "text": block["text"][: int(retrieval["excerpt_chars"])],
                }
            )
        return excerpts

    def _with_neighbor(self, chosen: list[str]) -> list[str]:
        position = {block_id: index for index, block_id in enumerate(self.order)}
        best = chosen[0]
        nxt_index = position[best] + 1
        if nxt_index >= len(self.order):
            return chosen
        nxt = self.order[nxt_index]
        if nxt in chosen or self.blocks[nxt]["page"] != self.blocks[best]["page"]:
            return chosen
        limit = int(self.retrieval["context_k"])
        body = chosen[: max(limit - 1, 1)]
        if nxt not in body:
            body.append(nxt)
        return body[:limit]


def render_context(excerpts: list[dict]) -> str:
    if not excerpts:
        return "Фрагменты не передавались."
    chunks = []
    for item in excerpts:
        chunks.append(
            f"[{item['block_id']} | с. {item['page']} | {item['section']}]\n{item['text']}"
        )
    return "\n\n".join(chunks)


def message_pair(question: dict, excerpts: list[dict], allow_search: bool, no_context: bool) -> list[dict]:
    if no_context:
        system = NO_CONTEXT
    else:
        system = ANSWER_RULES + "\n" + (SEARCH_RULE if allow_search else ANSWER_ONLY)
    user = "Вопрос:\n" + question["question"].strip() + "\n\nВарианты:\n"
    user += "\n".join(f"{item['id']}. {item['text']}" for item in question["options"])
    user += "\n\nФрагменты:\n" + render_context(excerpts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Budget:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def room_for_call(self) -> bool:
        return self.calls < LIMITS["llm_calls"] and self.input_tokens < LIMITS["input_tokens"] and self.output_tokens < LIMITS["output_tokens"]


def estimate_tokens(tokenizer, messages: list[dict]) -> int:
    text = "\n".join(item["content"] for item in messages)
    if tokenizer is None:
        return max(1, len(text) // 2)
    return len(tokenizer.encode(text))


def fit_messages(tokenizer, messages: list[dict], excerpts: list[dict], ceiling: int) -> tuple[list[dict], int]:
    """Ужимает фрагменты, пока вход + потолок выхода не превышают 8192."""
    current = messages
    use = list(excerpts)
    if len(messages) != 2:
        prompt = estimate_tokens(tokenizer, current)
        return current, min(ceiling, LIMITS["request_budget"] - prompt)
    while True:
        prompt = estimate_tokens(tokenizer, current)
        allowed = min(ceiling, LIMITS["request_budget"] - prompt)
        if allowed >= 256 or len(use) <= 1:
            return current, allowed
        use = use[:-1]
        current = message_pair_from(messages, use)


def message_pair_from(messages: list[dict], excerpts: list[dict]) -> list[dict]:
    # Пересобирает пользовательское сообщение, сохраняя вопрос до маркера фрагментов.
    user = messages[1]["content"]
    head = user.split("\n\nФрагменты:\n", 1)[0]
    rebuilt = head + "\n\nФрагменты:\n" + render_context(excerpts)
    return [messages[0], {"role": "user", "content": rebuilt}]


def output_tokens_of(usage: dict) -> int:
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    if reasoning > completion:
        return completion + reasoning
    return max(completion, reasoning)


def parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("` \n")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе нет JSON")
    return json.loads(raw[start : end + 1])


def normalize(parsed: dict, question: dict, allowed: set[str], require_citations: bool) -> dict:
    option_ids = [item["id"] for item in question["options"]]
    known = set(option_ids)
    answer_ids = [item for item in parsed.get("answer_ids") or [] if item in known]
    answer_ids = list(dict.fromkeys(answer_ids))
    citations = [item for item in parsed.get("citations") or [] if item in allowed]
    citations = list(dict.fromkeys(citations))[:4]
    abstain = bool(parsed.get("abstain"))
    rationale = str(parsed.get("rationale") or "").strip()
    if abstain:
        answer_ids, citations = [], []
        if not rationale:
            rationale = "В условии или фрагментах не хватает сведения для однозначного ответа."
    elif not answer_ids or (require_citations and not citations):
        raise ValueError("ответ без вариантов или без ссылок")
    if len(rationale) > RATIONALE_LIMIT:
        rationale = rationale[:RATIONALE_LIMIT]
    if not rationale:
        rationale = "Ответ выбран по переданным фрагментам."
    return {
        "answer_ids": answer_ids,
        "abstain": abstain,
        "rationale": rationale,
        "citations": citations,
    }


def call_llm(client, model: str, runtime: dict, messages: list[dict], max_tokens: int, question_id: str, submission_id: str) -> dict:
    sampling = runtime["sampling"]
    thinking = bool(runtime["thinking"])
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        presence_penalty=sampling["presence_penalty"],
        max_tokens=max_tokens,
        extra_body={
            "top_k": sampling["top_k"],
            "min_p": sampling["min_p"],
            "repetition_penalty": sampling["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
        extra_headers={"X-Question-ID": question_id, "X-Submission-ID": submission_id},
    )
    payload = response.model_dump()
    usage = payload.get("usage") or {}
    message = payload["choices"][0]["message"]
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "content": content,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": output_tokens_of(usage),
    }


def ask(client, model, runtime, tokenizer, budget: Budget, messages, excerpts, question_id, submission_id):
    ceiling = int(runtime["sampling"]["max_tokens"])
    remaining_out = LIMITS["output_tokens"] - budget.output_tokens
    remaining_in = LIMITS["input_tokens"] - budget.input_tokens
    fitted, allowed_out = fit_messages(tokenizer, messages, excerpts, min(ceiling, remaining_out))
    prompt_estimate = estimate_tokens(tokenizer, fitted)
    if prompt_estimate > remaining_in or allowed_out < 64 or not budget.room_for_call():
        raise RuntimeError("бюджет вопроса не позволяет ещё один вызов")
    result = call_llm(client, model, runtime, fitted, allowed_out, question_id, submission_id)
    budget.calls += 1
    budget.input_tokens += result["input_tokens"]
    budget.output_tokens += result["output_tokens"]
    if budget.input_tokens > LIMITS["input_tokens"] or budget.output_tokens > LIMITS["output_tokens"] or budget.calls > LIMITS["llm_calls"]:
        raise RuntimeError("превышен лимит вопроса")
    parsed = parse_json(result["content"])
    return parsed


def answer_question(client, model, runtime, tokenizer, retriever: Retriever | None, system: dict, question: dict, submission_id: str) -> tuple[dict, dict]:
    started = time.perf_counter()
    budget = Budget()
    decisions = []
    shown: list[str] = []
    try:
        return _answer(client, model, runtime, tokenizer, retriever, system, question, submission_id, started, budget, decisions, shown)
    except Exception as exc:
        print(f"{question['id']}: {exc}")
        answer, trace = failure(question["id"], f"Сбой обработки: {exc}", started)
        trace["llm_calls"] = budget.calls
        trace["input_tokens"] = budget.input_tokens
        trace["output_tokens"] = budget.output_tokens
        trace["retrieved_block_ids"] = list(dict.fromkeys(shown))
        return answer, trace


def _answer(client, model, runtime, tokenizer, retriever, system, question, submission_id, started, budget, decisions, shown) -> tuple[dict, dict]:
    excerpts: list[dict] = []
    no_context = retriever is None or not (system["bm25"] or system["dense"])
    if not no_context:
        excerpts = retriever.search(query_text(question))
        shown.extend(item["block_id"] for item in excerpts)
    allowed = {item["block_id"] for item in excerpts}
    require_citations = not no_context
    messages = message_pair(question, excerpts, system["adaptive"], no_context)
    parsed = ask(
        client, model, runtime, tokenizer, budget, messages, excerpts, question["id"], submission_id,
    )
    decision = str(parsed.get("decision") or "answer")
    search_query = str(parsed.get("search_query") or "").strip()
    if system["adaptive"] and decision == "search" and search_query and budget.room_for_call():
        decisions.append({"llm_call": budget.calls, "decision": "search", "search_query": search_query})
        extra = retriever.search(search_query)
        for item in extra:
            if item["block_id"] not in shown:
                shown.append(item["block_id"])
        excerpts = extra[: int(retriever.retrieval["context_k"])] or excerpts
        allowed = {item["block_id"] for item in excerpts}
        messages = message_pair(question, excerpts, False, False)
        parsed = ask(
            client, model, runtime, tokenizer, budget, messages, excerpts, question["id"], submission_id,
        )
    try:
        body = normalize(parsed, question, allowed, require_citations)
    except ValueError:
        if not budget.room_for_call():
            raise
        repair = messages + [
            {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)[:2000]},
            {"role": "user", "content": "Предыдущий JSON неполон. Верни только исправленный JSON по схеме из системной инструкции. decision=\"answer\"."},
        ]
        parsed = ask(
            client, model, runtime, tokenizer, budget, repair, excerpts, question["id"], submission_id,
        )
        decisions.append({"llm_call": budget.calls, "decision": "repair"})
        body = normalize(parsed, question, allowed, require_citations)
    answer = {"id": question["id"], **body}
    trace = {
        "id": question["id"],
        "llm_calls": budget.calls,
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "retrieved_block_ids": list(dict.fromkeys(shown)),
        "decisions": decisions,
    }
    return answer, trace


def failure(question_id: str, reason: str, started: float, budget: Budget | None = None) -> tuple[dict, dict]:
    answer = {
        "id": question_id,
        "answer_ids": [],
        "abstain": True,
        "rationale": reason[:RATIONALE_LIMIT],
        "citations": [],
    }
    trace = {
        "id": question_id,
        "llm_calls": 0 if budget is None else budget.calls,
        "input_tokens": 0 if budget is None else budget.input_tokens,
        "output_tokens": 0 if budget is None else budget.output_tokens,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "retrieved_block_ids": [],
        "decisions": [],
    }
    return answer, trace


def load_tokenizer(config: dict):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            config["llm_tokenizer"], revision=config["llm_tokenizer_revision"], trust_remote_code=True
        )
    except Exception as exc:
        print(f"Токенайзер Qwen недоступен, оценка длины приблизительная: {exc}")
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Предсказания RAG")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default=None)
    args = parser.parse_args()
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "SUBMISSION_ID"):
        if not os.environ.get(name):
            raise SystemExit(f"Не задана переменная {name}")
    config = load_config(args.config)
    runtime = read_json(Path(config["_config_dir"]) / config["runtime"] if not Path(config["runtime"]).is_absolute() else Path(config["runtime"]))
    system_name = args.system or config["system"]
    system = SYSTEMS[system_name]
    questions = read_jsonl(args.questions)
    ids = [row["id"] for row in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("В вопросах есть повторяющиеся id")
    retriever = None if system_name == "no_context" else Retriever(load_index(args.index), config, system)
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"].rstrip("/"),
        api_key=os.environ["LLM_API_KEY"],
        timeout=float(runtime["timeouts_sec"]["request"]),
    )
    tokenizer = load_tokenizer(config)
    answers, traces = [], []
    for question in questions:
        answer, trace = answer_question(
            client, os.environ["LLM_MODEL"], runtime, tokenizer, retriever, system, question, os.environ["SUBMISSION_ID"]
        )
        answers.append(answer)
        traces.append(trace)
        print(f"{question['id']} calls={trace['llm_calls']} abstain={answer['abstain']} answers={answer['answer_ids']}", flush=True)
        if trace["llm_calls"] == 0:
            raise SystemExit(f"Модель не ответила на {question['id']}. Прогон остановлен.\n{answer['rationale']}")
    write_jsonl(args.output, answers)
    write_jsonl(args.trace, traces)
    print(f"answers={args.output} trace={args.trace} system={system_name}")


if __name__ == "__main__":
    main()
