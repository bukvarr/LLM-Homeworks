"""Эмбеддинги e5 и кросс-энкодер. Только CPU, без генерации."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def force_cpu() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))


def average_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)


class Encoder:
    def __init__(self, model_id: str, revision: str):
        force_cpu()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision)
        self.model.eval()
        self.model.to("cpu")

    def encode(self, texts: list[str], batch_size: int) -> "torch.Tensor":
        import numpy as np

        parts = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
                pooled = average_pool(hidden, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            parts.append(pooled.cpu().numpy().astype("float32"))
        if not parts:
            return np.zeros((0, 0), dtype="float32")
        return np.concatenate(parts, axis=0)


class Reranker:
    def __init__(self, model_id: str, revision: str):
        force_cpu()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.eval()
        self.model.to("cpu")

    def scores(self, query: str, passages: list[str], batch_size: int) -> list[float]:
        values: list[float] = []
        for start in range(0, len(passages), batch_size):
            chunk = passages[start : start + batch_size]
            tokens = self.tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**tokens).logits.view(-1).float().cpu()
            values.extend(float(item) for item in logits)
        return values
