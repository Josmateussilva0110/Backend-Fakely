"""Pré-processamento do texto (RN04) e extração de características de estilo."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.analysis_rules_service import AnalysisRules

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Palavras com acentos e maiúsculas preservados (antes da normalização)
WORD_PATTERN = re.compile(r"\w+(?:[-']\w+)*")
_REPEATED_PUNCTUATION_PATTERN = re.compile(r"[!?]{2,}")
# Acima dessa proporção de maiúsculas o texto está "gritando" e a caixa não indica nome próprio
SHOUTING_UPPERCASE_RATIO = 0.7
ACRONYM_MIN_LENGTH = 2
ACRONYM_MAX_LENGTH = 5
# "Leão XIV", "século XX": algarismo romano não é sigla
_ROMAN_NUMERAL_PATTERN = re.compile(r"[IVXLCDM]+")


@dataclass(frozen=True)
class TextFeatures:
    word_count: int
    uppercase_ratio: float
    exclamation_count: int
    question_count: int
    repeated_punctuation_count: int
    sensational_terms: tuple[str, ...]
    alarmist_terms: tuple[str, ...]


def remove_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = remove_accents(text.lower())
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(normalized_text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(normalized_text)


def is_shouting(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(char.isupper() for char in letters) / len(letters) > SHOUTING_UPPERCASE_RATIO


def is_acronym(word: str, known: frozenset[str], shouting: bool) -> bool:
    """Sigla conhecida em qualquer caixa, ou palavra curta toda em maiúsculas num texto que não grita."""
    if normalize_text(word) in known:
        return True
    return (
        not shouting
        and ACRONYM_MIN_LENGTH <= len(word) <= ACRONYM_MAX_LENGTH
        and word.isalpha()
        and word.isupper()
        and not _ROMAN_NUMERAL_PATTERN.fullmatch(word)
    )


def find_terms(normalized_text: str, terms: frozenset[str]) -> tuple[str, ...]:
    return tuple(term for term in sorted(terms) if re.search(rf"\b{re.escape(term)}\b", normalized_text))


def extract_features(text: str, rules: AnalysisRules, title: str | None = None) -> TextFeatures:
    raw = f"{title}\n{text}" if title else text
    normalized = normalize_text(raw)

    letters = [char for char in raw if char.isalpha()]
    uppercase = sum(1 for char in letters if char.isupper())

    return TextFeatures(
        word_count=len(tokenize(normalize_text(text))),
        uppercase_ratio=round(uppercase / len(letters), 3) if letters else 0.0,
        exclamation_count=raw.count("!"),
        question_count=raw.count("?"),
        repeated_punctuation_count=len(_REPEATED_PUNCTUATION_PATTERN.findall(raw)),
        sensational_terms=find_terms(normalized, rules.sensational_terms),
        alarmist_terms=find_terms(normalized, rules.alarmist_terms),
    )


def describe_style_signals(features: TextFeatures, rules: AnalysisRules) -> list[str]:
    """Sinais de estilo comuns em desinformação; são alertas, não provas de falsidade."""
    thresholds = rules.style_thresholds
    signals = []
    if features.sensational_terms:
        signals.append(f"Termos sensacionalistas: {', '.join(features.sensational_terms)}")
    if features.alarmist_terms:
        signals.append(f"Afirmações alarmistas: {', '.join(features.alarmist_terms)}")
    if features.repeated_punctuation_count:
        signals.append("Uso de pontuação repetida (ex.: '!!!', '?!')")
    elif features.exclamation_count >= thresholds.excess_exclamation_min:
        signals.append("Excesso de pontos de exclamação")
    if features.uppercase_ratio > thresholds.uppercase_ratio_min:
        signals.append("Uso excessivo de letras maiúsculas")
    return signals


# Quantidades que identificam um fato: "morreu aos 86 anos" e "morre aos 93 anos" são fatos diferentes.
# Só tipos com unidade clara; um número solto ("acima de 5 mil") não diz o que mede
_NUMBER = r"\d+(?:[.,]\d+)*"
_SCALE = r"(?:\s+(mil|milhao|milhoes|bilhao|bilhoes))?"
_SCALES = {None: 1, "mil": 1e3, "milhao": 1e6, "milhoes": 1e6, "bilhao": 1e9, "bilhoes": 1e9}
_QUANTITY_PATTERNS = {
    "age": re.compile(rf"\b(?:aos|com)\s+({_NUMBER})()\s+anos\b"),
    "percent": re.compile(rf"\b({_NUMBER})()\s*(?:%|por cento)"),
    "money": re.compile(rf"(?:r\$|us\$|u\$)\s*({_NUMBER}){_SCALE}"),
    "money_words": re.compile(rf"\b({_NUMBER}){_SCALE}\s+(?:de\s+)?(?:reais|dolares)\b"),
    "cents": re.compile(rf"\b({_NUMBER})()\s+centavos\b"),
}
# Dinheiro escrito com símbolo ou por extenso é a mesma medida
_QUANTITY_KIND = {"money_words": "money"}


def parse_number(raw: str) -> float:
    """Número no formato brasileiro ou americano: "1.518" e "5.000" são milhares; "0,37" e "0.37", decimais."""
    if "," in raw:
        return float(raw.replace(".", "").replace(",", "."))
    groups = raw.split(".")
    if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
        return float("".join(groups))
    return float(raw)


def extract_quantities(normalized_text: str) -> dict[str, set[float]]:
    """Idades, porcentagens e valores em dinheiro do texto normalizado, já padronizados ("5 mil" = 5000)."""
    found: dict[str, set[float]] = {}
    for name, pattern in _QUANTITY_PATTERNS.items():
        for match in pattern.finditer(normalized_text):
            value = parse_number(match.group(1)) * _SCALES[match.group(2) or None]
            found.setdefault(_QUANTITY_KIND.get(name, name), set()).add(round(value, 6))
    return found


def quantities_conflict(first: dict[str, set[float]], second: dict[str, set[float]]) -> bool:
    """Os dois textos citam a mesma medida (idade, porcentagem...) com valores diferentes."""
    return any(not (first[kind] & second[kind]) for kind in first.keys() & second.keys())
