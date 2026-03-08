from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

@dataclass
class TfidfConfig:
    # general
    output_root: str = "outputs/tfidf"
    seed: int = 42
    val_size: float = 0.1
    labels: list[str] = field(
        default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"]
    )

    # vectorizer
    lowercase: bool = True
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 0.95
    max_features: Optional[int] = 100_000
    sublinear_tf: bool = True
    strip_accents: Optional[str] = "unicode"
    stop_words: Optional[str] = "english"

    # classifier
    
    # inverse of regularization strength (1/lambda);
    # smaller C -> stronger regularization
    C: float = 4.0
    max_iter: int = 2000
    n_jobs: int = -1  # use all CPU cores
    random_state: int = 42


def build_vectorizer(cfg: TfidfConfig) -> TfidfVectorizer:
    """
    Build a TfidfVectorizer based on the provided configuration.
    """
    vectorizer = TfidfVectorizer(
        lowercase=cfg.lowercase,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents=cfg.strip_accents,
        stop_words=cfg.stop_words,
    )
    return vectorizer

def build_classifier(cfg: TfidfConfig) -> LogisticRegression:
    """
    Build a LogisticRegression classifier based on the provided configuration.
    """
    clf = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    return clf

def build_pipeline(cfg: TfidfConfig) -> Pipeline:
    """
    Build a scikit-learn Pipeline for TF-IDF vectorization and Logistic Regression classification.
    """
    pipe = Pipeline(
        steps=[
            ("tfidf", build_vectorizer(cfg)),
            ("clf", build_classifier(cfg)),
        ]
    )

    return pipe







