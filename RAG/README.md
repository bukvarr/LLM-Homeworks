# Доказательный RAG по ТФКП

Шамсутдинов Данис Маратович, группа М05-603.

Модель генерации: Qwen3.5-4B, только через OpenAI-совместимый endpoint. Поиск, эмбеддинги и reranker считаются на CPU.

## Что лежит в репозитории

Код пересобирает `corpus.jsonl` из PDF, индекс и ответы. Сами `corpus.jsonl`, индекс и векторную базу загружать не нужно. `dev_questions.jsonl` и `dev_gold.jsonl` лежат в корне: это 15 вопросов из `tfkp_15_qa.md`, привязанные к `block_id` этого извлечения. `dev_answers.jsonl` и `dev_trace.jsonl` появляются после прогона итоговой системы. `experiments.csv` и числовой блок `report.md` тоже.

Официального `runtime.json` в материалах курса не было. Файл в корне — параметры non-thinking режима с карточки Qwen3.5-4B. `predict.py` читает их и не меняет temperature, top_p, top_k, min_p, presence_penalty и repetition_penalty.

Контрольная сумма корпуса этой версии извлечения: `f86eecd5ad64c235aad728869b7dccf25025644e560f4f2ca9bcaf2a622c4521`.

## Секреты

В Colab → Secrets задайте:

- `LLM_BASE_URL` — адрес OpenAI-совместимого API без хвоста `/chat/completions`;
- `LLM_MODEL` — id Qwen3.5-4B на этом endpoint;
- `LLM_API_KEY`;
- `SUBMISSION_ID`.

Ключи в репозиторий не класть. Каждый запрос получает заголовки `X-Question-ID` и `X-Submission-ID`.

## Запуск

Ноутбук `colab_run.ipynb` повторяет те же команды. Из чистого окружения Python 3.11, из любой папки:

```bash
python -m pip install -r requirements.txt
python extract_corpus.py --pdf /path/to/tfkp.pdf --output /path/to/corpus.jsonl
python build_index.py --corpus /path/to/corpus.jsonl --output /path/to/index --config /path/to/config.json
python predict.py --index /path/to/index --questions /path/to/dev_questions.jsonl --output /path/to/dev_answers.jsonl --trace /path/to/dev_trace.jsonl --config /path/to/config.json
```

PDF: <https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf>

Пять прогонов пишутся в `runs/` и собираются так:

```bash
python predict.py --index /path/to/index --questions /path/to/dev_questions.jsonl --output runs/final_answers.jsonl --trace runs/final_trace.jsonl --config /path/to/config.json --system final
python collect_experiments.py --gold dev_gold.jsonl --runs runs --csv experiments.csv --report report.md --corpus /path/to/corpus.jsonl --stats /path/to/index/stats.json
```

Остальные системы: `no_context`, `single`, `no_rerank`, `no_dense`. В `config.json` поле `system` для сдачи равно `final`.

## Архитектура

Индекс без LLM. Абзацы PDF становятся блоками с исходным `block_id`; цитаты ссылаются только на них. Для поиска блок режется на окна по 500 символов с шагом 320. BM25 и `intfloat/multilingual-e5-small` (117 653 760 параметров, ревизия `614241f6…`) ищут по окнам, оценки сливаются через RRF с c = 60, родительский блок берётся один раз. Затем `BAAI/bge-reranker-v2-m3` (567 755 777 параметров, ревизия `953dc6f6…`) переставляет не больше 12 кандидатов. В контекст попадают 4 блока; соседний блок той же страницы может занять последнее место.

Итоговая система делает один вызов Qwen. Второй вызов бывает только если модель сама вернула `decision=search` и новый запрос. Это решение пишется в `decisions` внутри `trace.jsonl`. Третий вызов — только починка сломанного JSON. Потолок из `runtime.json` уменьшается, если иначе вход и допустимый выход превысят 8192.

Абляции: `no_rerank` убирает кросс-энкодер, `no_dense` оставляет BM25. `single` — один BM25-поиск и один вызов. `no_context` отвечает без корпуса; для него в CSV заполняется только точность вариантов.
