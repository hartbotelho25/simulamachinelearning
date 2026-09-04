"""Treino, predição com limiar, métricas e impacto das variáveis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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
    acuracia: float
    especificidade: float
    importancias: dict[str, float] = field(default_factory=dict)
    origem_importancia: str = ""
    direcao: dict[str, float] = field(default_factory=dict)
    exemplos_tp: list[str] = field(default_factory=list)
    exemplos_fp: list[str] = field(default_factory=list)
    exemplos_fn: list[str] = field(default_factory=list)

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Separa treino/teste com stratify=y para preservar a proporção de lendários."""
    X = df[features]
    y = df[TARGET].astype(int)
    if "name" in df.columns:
        names = df["name"].astype(str)
    else:
        names = pd.Series([f"#{i}" for i in df.index], index=df.index)
    test_size = 1.0 - (train_pct / 100.0)
    return train_test_split(
        X,
        y,
        names,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _unwrap(estimator):
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def _feature_impact(
    estimator,
    features: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], str, dict[str, float]]:
    """Importância nativa (árvore/LR) ou permutação (Naive Bayes)."""
    clf = _unwrap(estimator)
    direcao: dict[str, float] = {}

    if hasattr(clf, "coef_"):
        raw = np.asarray(clf.coef_[0], dtype=float)
        direcao = {f: float(v) for f, v in zip(features, raw)}
        pesos = {f: float(abs(v)) for f, v in direcao.items()}
        return pesos, "peso absoluto do coeficiente", direcao

    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(clf.feature_importances_, dtype=float)
        return (
            {f: float(v) for f, v in zip(features, vals)},
            "importância na árvore (redução de impureza)",
            direcao,
        )

    if len(features) == 1:
        return {features[0]: 1.0}, "único atributo no treino", direcao

    try:
        perm = permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=8,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=1,
        )
        return (
            {f: float(v) for f, v in zip(features, perm.importances_mean)},
            "queda de ROC AUC ao embaralhar a variável",
            direcao,
        )
    except Exception:
        return {}, "", direcao


def _sample_names(mask: np.ndarray, names: np.ndarray, limit: int = 6) -> list[str]:
    picked = names[mask]
    return [str(n) for n in picked[:limit]]


def evaluate_estimator(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    names_test: pd.Series,
    threshold: float,
) -> EvaluationResult:
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)[:, 1]
    y_hat = (proba >= threshold).astype(int)

    y_true = y_test.to_numpy()
    name_arr = names_test.to_numpy()
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))

    predito = int(np.sum(y_hat == 1))
    real = int(np.sum(y_true == 1))
    n = len(y_true)

    precisao = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precisao * recall, precisao + recall)
    erro_palpite = _safe_div(fp, tp + fp)
    margem = _safe_div(predito - real, real) * 100.0
    acuracia = _safe_div(tp + tn, n)
    especificidade = _safe_div(tn, tn + fp)

    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")

    importancias, origem, direcao = _feature_impact(
        estimator, list(X_train.columns), X_test, y_test
    )

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
        acuracia=acuracia,
        especificidade=especificidade,
        importancias=importancias,
        origem_importancia=origem,
        direcao=direcao,
        exemplos_tp=_sample_names((y_hat == 1) & (y_true == 1), name_arr),
        exemplos_fp=_sample_names((y_hat == 1) & (y_true == 0), name_arr),
        exemplos_fn=_sample_names((y_hat == 0) & (y_true == 1), name_arr),
    )


def run_experiment(
    df: pd.DataFrame,
    features: list[str],
    model_names: list[str],
    train_pct: int,
    threshold: float,
) -> tuple[list[EvaluationResult], int, int]:
    """Treina os modelos selecionados e devolve resultados + totais do teste."""
    X_train, X_test, y_train, y_test, _names_tr, names_te = split_data(
        df, features, train_pct
    )
    results: list[EvaluationResult] = []
    for name in model_names:
        key = MODEL_CATALOG[name]
        estimator = build_estimator(key)
        results.append(
            evaluate_estimator(
                name,
                estimator,
                X_train,
                X_test,
                y_train,
                y_test,
                names_te,
                threshold,
            )
        )
    return results, int(len(y_test)), int(y_test.sum())


def variable_profile(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compara a média de cada variável entre lendários e comuns na base limpa."""
    rows = []
    y = df[TARGET].astype(int)
    for feat in features:
        lend = df.loc[y == 1, feat]
        comum = df.loc[y == 0, feat]
        m_l = float(lend.mean()) if len(lend) else 0.0
        m_c = float(comum.mean()) if len(comum) else 0.0
        rows.append(
            {
                "atributo": feat,
                "media_lendarios": m_l,
                "media_comuns": m_c,
                "diferenca": m_l - m_c,
            }
        )
    return pd.DataFrame(rows)


def ablation_table(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    train_pct: int,
    threshold: float,
) -> pd.DataFrame:
    """Retreina o método sem cada variável para mostrar o impacto em F1 e na contagem."""
    if len(features) < 2:
        return pd.DataFrame()

    baseline_list, _, _ = run_experiment(df, features, [model_name], train_pct, threshold)
    baseline = baseline_list[0]
    X_train, X_test, y_train, y_test, _ntr, names_te = split_data(df, features, train_pct)
    key = MODEL_CATALOG[model_name]
    rows = []
    for feat in features:
        keep = [f for f in features if f != feat]
        estimator = build_estimator(key)
        res = evaluate_estimator(
            model_name,
            estimator,
            X_train[keep],
            X_test[keep],
            y_train,
            y_test,
            names_te,
            threshold,
        )
        rows.append(
            {
                "atributo": feat,
                "f1_sem": res.f1 * 100.0,
                "delta_f1": (res.f1 - baseline.f1) * 100.0,
                "predito_sem": res.total_predito,
                "delta_predito": res.total_predito - baseline.total_predito,
                "recall_sem": res.recall * 100.0,
            }
        )
    return pd.DataFrame(rows)
