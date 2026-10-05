"""Treina o classificador a partir de um CSV rotulado.

Uso:
    python scripts/train_model.py data.csv --text-column text --label-column label

Rótulos aceitos: 1/0, fake/true, falsa/verdadeira (1/fake/falsa = notícia falsa).
Sugestão de dataset em português: Fake.br Corpus (https://github.com/roneysco/Fake.br-Corpus).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ml.news_classifier import NewsClassifier  # noqa: E402
from app.services.analysis_rules_service import load_analysis_rules  # noqa: E402

LABEL_MAP = {
    "1": 1, "fake": 1, "falsa": 1, "falso": 1,
    "0": 0, "true": 0, "verdadeira": 0, "verdadeiro": 0,
}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", type=Path, default=settings.model_path)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stopwords = load_analysis_rules(get_settings().analysis_rules_path).stopwords

    df = pd.read_csv(args.csv)[[args.text_column, args.label_column]].dropna()
    df["y"] = df[args.label_column].astype(str).str.strip().str.lower().map(LABEL_MAP)
    invalid = int(df["y"].isna().sum())
    if invalid:
        print(f"Ignorando {invalid} linha(s) com rótulo desconhecido.")
        df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        df[args.text_column], df["y"], test_size=args.test_size, stratify=df["y"], random_state=42
    )
    classifier = NewsClassifier().train(x_train, y_train, stopwords)
    predictions = classifier.pipeline.predict(list(x_test))
    print(classification_report(y_test, predictions, target_names=["true", "fake"]))

    # Modelo final treinado com todos os dados
    classifier.train(df[args.text_column], df["y"], stopwords).save(args.output)
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()
