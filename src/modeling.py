"""Treino, predição com limiar e métricas de classificação."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import TARGET

RANDOM_STATE = 42

MODEL_CATALOG: dict[str, str] = {
    "Naive Bayes": "nb",
    "Regressão Logística": "lr",
    "Árvore de Decisão": "dt",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
}


@dataclass
class EvaluationResult:
    modelo: str
    total_predito: int
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float
    precisao: float
    f1: float
    roc_auc: float
    margem_pct: float
    erro_palpite_pct: float
    captura_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_estimator(key: str):
    """Instancia o algoritmo. NB e LR passam por padronização; árvores não."""
    if key == "nb":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        )
    if key == "lr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if key == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if key == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    raise KeyError(f"Algoritmo desconhecido: {key}")


def split_data(
    df: pd.DataFrame,
    features: list[str],
    train_pct: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino/teste com stratify=y para preservar a proporção de lendários."""
    X = df[features]
    y = df[TARGET].astype(int)
    test_size = 1.0 - (train_pct / 100.0)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate_estimator(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float,
) -> EvaluationResult:
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)[:, 1]
    y_hat = (proba >= threshold).astype(int)

    y_true = y_test.to_numpy()
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))

    predito = int(np.sum(y_hat == 1))
    real = int(np.sum(y_true == 1))

    precisao = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precisao * recall, precisao + recall)
    erro_palpite = _safe_div(fp, tp + fp)
    margem = _safe_div(predito - real, real) * 100.0

    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")

    return EvaluationResult(
        modelo=name,
        total_predito=predito,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=recall,
        precisao=precisao,
        f1=f1,
        roc_auc=auc,
        margem_pct=margem,
        erro_palpite_pct=erro_palpite * 100.0,
        captura_pct=recall * 100.0,
    )


def run_experiment(
    df: pd.DataFrame,
    features: list[str],
    model_names: list[str],
    train_pct: int,
    threshold: float,
) -> tuple[list[EvaluationResult], int, int]:
    """Treina os modelos selecionados e devolve resultados + totais do teste."""
    X_train, X_test, y_train, y_test = split_data(df, features, train_pct)
    results: list[EvaluationResult] = []
    for name in model_names:
        key = MODEL_CATALOG[name]
        estimator = build_estimator(key)
        results.append(
            evaluate_estimator(
                name, estimator, X_train, X_test, y_train, y_test, threshold
            )
        )
    return results, int(len(y_test)), int(y_test.sum())
