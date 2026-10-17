"""Inferência textual (NLI): estima se uma manchete implica a afirmação analisada."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

PAD_TOKEN = "[PAD]"


class EntailmentScorer(Protocol):
    def entailment(self, premises: list[str], hypothesis: str) -> list[float]: ...


class EntailmentModel:
    """Modelo ONNX carregado uma vez na inicialização e compartilhado entre as requisições."""

    def __init__(self, session: ort.InferenceSession, tokenizer: Tokenizer, entailment_index: int):
        self.session = session
        self.tokenizer = tokenizer
        self.entailment_index = entailment_index
        self.input_names = {item.name for item in session.get_inputs()}

    @classmethod
    def load(cls, repo: str, revision: str, model_file: str, cache_dir: Path, max_tokens: int) -> "EntailmentModel":
        # Revisão fixa: o arquivo baixado é sempre o mesmo que foi avaliado
        def download(filename: str) -> Path:
            return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir))

        config = json.loads(download("config.json").read_text(encoding="utf-8"))
        labels = {label.lower(): int(index) for index, label in config["id2label"].items()}
        tokenizer = Tokenizer.from_file(str(download("tokenizer.json")))
        tokenizer.enable_truncation(max_tokens)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD_TOKEN) or 0, pad_token=PAD_TOKEN)
        session = ort.InferenceSession(str(download(model_file)), providers=["CPUExecutionProvider"])
        return cls(session, tokenizer, labels["entailment"])

    def entailment(self, premises: list[str], hypothesis: str) -> list[float]:
        """Probabilidade (0 a 1) de cada premissa implicar a hipótese, em um único lote."""
        if not premises:
            return []
        encodings = self.tokenizer.encode_batch([(premise, hypothesis) for premise in premises])
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        logits = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return [float(p) for p in probabilities[:, self.entailment_index]]
