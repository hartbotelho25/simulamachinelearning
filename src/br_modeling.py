"""Treino, limiar, métricas e impacto — Brasileirão."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.br_data import COL_CLUB

RANDOM_STATE = 42

MODEL_CATALOG: dict[str, str] = {
    "Naive Bayes": "nb",
    "KNN": "knn",
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
    cv_f1: float | None = None


def impact_rank(result: EvaluationResult) -> list[tuple[str, float]]:
    return sorted(result.importancias.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)


def build_estimator(key: str, k_neighbors: int = 5):
    if key == "nb":
        return Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
    if key == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=max(1, k_neighbors))),
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
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if key == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced",
        )
    if key == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    raise KeyError(key)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _unwrap(estimator):
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def split_data(df: pd.DataFrame, features: list[str], target_col: str, train_pct: int):
    X = df[features]
    y = df[target_col].astype(int)
    names = df[COL_CLUB].astype(str)
    test_size = 1.0 - (train_pct / 100.0)
    kwargs = dict(test_size=test_size, random_state=RANDOM_STATE)
    try:
        return train_test_split(X, y, names, stratify=y, **kwargs)
    except ValueError:
        return train_test_split(X, y, names, **kwargs)


def _feature_impact(estimator, features, X_test, y_test):
    clf = _unwrap(estimator)
    direcao: dict[str, float] = {}
    if hasattr(clf, "coef_"):
        raw = np.asarray(clf.coef_[0], dtype=float)
        direcao = {f: float(v) for f, v in zip(features, raw)}
        return {f: abs(v) for f, v in direcao.items()}, "peso absoluto do coeficiente", direcao
    if hasattr(clf, "feature_importances_"):
        vals = np.asarray(clf.feature_importances_, dtype=float)
        return (
            {f: float(v) for f, v in zip(features, vals)},
            "importância na árvore (impureza)",
            direcao,
        )
    if len(features) == 1:
        return {features[0]: 1.0}, "único atributo no treino", direcao
    try:
        perm = permutation_importance(
            estimator, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE,
            scoring="f1", n_jobs=1,
        )
        return (
            {f: float(v) for f, v in zip(features, perm.importances_mean)},
            "queda de F1 ao embaralhar a variável",
            direcao,
        )
    except Exception:
        return {}, "", direcao


def _names(mask, names, limit=20) -> list[str]:
    return [str(n) for n in names[mask][:limit]]


def evaluate_estimator(name, estimator, X_train, X_test, y_train, y_test, names_test, threshold):
    n_train = len(X_train)
    clf = _unwrap(estimator)
    if isinstance(clf, KNeighborsClassifier):
        clf.set_params(n_neighbors=min(clf.n_neighbors, max(1, n_train)))
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
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")
    imps, origem, direcao = _feature_impact(estimator, list(X_train.columns), X_test, y_test)
    return EvaluationResult(
        modelo=name,
        total_predito=predito,
        tp=tp, fp=fp, fn=fn, tn=tn,
        recall=recall, precisao=precisao, f1=f1, roc_auc=auc,
        margem_pct=_safe_div(predito - real, real) * 100.0,
        erro_palpite_pct=_safe_div(fp, tp + fp) * 100.0,
        captura_pct=recall * 100.0,
        acuracia=_safe_div(tp + tn, n),
        especificidade=_safe_div(tn, tn + fp),
        importancias=imps,
        origem_importancia=origem,
        direcao=direcao,
        exemplos_tp=_names((y_hat == 1) & (y_true == 1), name_arr),
        exemplos_fp=_names((y_hat == 1) & (y_true == 0), name_arr),
        exemplos_fn=_names((y_hat == 0) & (y_true == 1), name_arr),
    )


def run_experiment(df, features, target_col, model_names, train_pct, threshold, k_neighbors=5):
    X_tr, X_te, y_tr, y_te, _ntr, names_te = split_data(df, features, target_col, train_pct)
    results = []
    for name in model_names:
        est = build_estimator(MODEL_CATALOG[name], k_neighbors=k_neighbors)
        results.append(evaluate_estimator(name, est, X_tr, X_te, y_tr, y_te, names_te, threshold))
    return results, int(len(y_te)), int(y_te.sum())


def crossval_f1(df, features, target_col, model_name, k_neighbors=5) -> float | None:
    X = df[features]
    y = df[target_col].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        return None
    est = build_estimator(MODEL_CATALOG[model_name], k_neighbors=k_neighbors)
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(est, X, y, scoring="f1", cv=cv)
        return float(np.mean(scores))
    except Exception:
        return None


def variable_profile(df, features, target_col) -> pd.DataFrame:
    rows = []
    y = df[target_col].astype(int)
    for feat in features:
        pos = df.loc[y == 1, feat]
        neg = df.loc[y == 0, feat]
        rows.append(
            {
                "atributo": feat,
                "media_sim": float(pos.mean()) if len(pos) else 0.0,
                "media_nao": float(neg.mean()) if len(neg) else 0.0,
                "diferenca": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ablation_table(df, features, target_col, model_name, train_pct, threshold, k_neighbors=5):
    if len(features) < 2:
        return pd.DataFrame()
    base, _, _ = run_experiment(df, features, target_col, [model_name], train_pct, threshold, k_neighbors)
    baseline = base[0]
    X_tr, X_te, y_tr, y_te, _n, names = split_data(df, features, target_col, train_pct)
    key = MODEL_CATALOG[model_name]
    rows = []
    for feat in features:
        keep = [f for f in features if f != feat]
        est = build_estimator(key, k_neighbors=k_neighbors)
        res = evaluate_estimator(model_name, est, X_tr[keep], X_te[keep], y_tr, y_te, names, threshold)
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
