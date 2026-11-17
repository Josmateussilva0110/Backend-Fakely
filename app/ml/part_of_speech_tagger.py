"""Classe gramatical das palavras (spaCy): substantivos e nomes identificam o assunto melhor que adjetivos."""

from threading import Lock
from typing import Protocol

import spacy
from spacy.language import Language

# Só o necessário para a classe gramatical; análise sintática e entidades deixariam cada frase ~2,5x mais lenta
EXCLUDED_COMPONENTS = ("parser", "ner", "lemmatizer")


class WordTagger(Protocol):
    def tag(self, text: str) -> list[tuple[str, str]]: ...


class PartOfSpeechTagger:
    """Modelo carregado uma vez na inicialização e compartilhado entre as requisições."""

    def __init__(self, nlp: Language):
        self.nlp = nlp
        # O spaCy não garante uso simultâneo do mesmo pipeline; cada frase leva ~2 ms
        self.lock = Lock()

    @classmethod
    def load(cls, model_name: str) -> "PartOfSpeechTagger":
        return cls(spacy.load(model_name, exclude=list(EXCLUDED_COMPONENTS)))

    def tag(self, text: str) -> list[tuple[str, str]]:
        """Pares (palavra, classe) no padrão Universal Dependencies: NOUN, PROPN, VERB, ADJ..."""
        with self.lock:
            doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc if not token.is_punct and not token.is_space]
