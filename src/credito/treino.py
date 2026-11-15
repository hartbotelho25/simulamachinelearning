"""Quatro modelos no mesmo desenho de colunas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .dataset import FEATURES
from .politica import FRACAO_TESTE, SEMENTE

CATALOGO = [
    {
        "slug": "logistica",
        "nome": "Regressão logística",
        "papel": "Referência linear. O peso de cada gatilho fica visível.",
        "tecnica": (
            "Modelo linear. Os números entram padronizados e as categorias abrem em colunas. "
            "Cada peso se lê com sinal: empurra para o calote ou afasta dele. "
            "Uma fatia separada do treino recalibra a chance com regressão isotônica, "
            "para ela acompanhar a taxa observada."
        ),
    },
    {
        "slug": "floresta",
        "nome": "Random Forest",
        "papel": "Ensemble de árvores, estável diante de relação não linear.",
        "tecnica": (
            "Duzentas árvores. Cada uma vê uma amostra e um subconjunto de colunas. "
            "A chance é a média dos votos. Uma fatia separada do treino recalibra essa chance "
            "com regressão isotônica, para ela acompanhar a taxa observada."
        ),
    },
    {
        "slug": "boosting",
        "nome": "Gradient Boosting (XGBoost)",
        "papel": "Árvores em sequência, o padrão forte em tabela de crédito.",
        "tecnica": (
            "XGBoost com árvores rasas. A árvore seguinte corrige o erro da anterior. "
            "Captura não linearidade sem virar uma rede. Uma fatia separada do treino "
            "recalibra a chance com regressão isotônica, para ela acompanhar a taxa observada."
        ),
    },
    {
        "slug": "mlp",
        "nome": "Rede neural MLP",
        "papel": "Interações entre os blocos, em camadas densas.",
        "tecnica": (
            "Perceptron de duas camadas ocultas, 64 e 32 neurônios, ativação ReLU. "
            "Os números entram padronizados. A classe rara pesa mais no treino, "
            "na proporção inversa da frequência, e a rede volta aos coeficientes da melhor validação. "
            "Uma fatia separada recalibra a chance com regressão isotônica, "
            "para ela acompanhar a taxa observada. A faixa de cor sai dessa chance calibrada."
        ),
    },
]


def construir(slug: str):
    """Logística e MLP passam por escala. Árvores recebem o 0/1 direto."""
    if slug == "logistica":
        clf = LogisticRegression(max_iter=800, solver="lbfgs", C=1.0)
        return Pipeline([("prep", StandardScaler()), ("clf", clf)])
    if slug == "floresta":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=10,
            random_state=SEMENTE,
            n_jobs=-1,
        )
    if slug == "boosting":
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEMENTE,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    if slug == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=SEMENTE,
        )
        return Pipeline([("prep", StandardScaler()), ("clf", clf)])
    raise KeyError(slug)


def fatiar(df: pd.DataFrame):
    classe = df["inadimplente"].to_numpy(dtype=int) + df["inadimplente_90d"].to_numpy(dtype=int)
    X = df[FEATURES]
    indice = np.arange(len(df))
    i_tr, i_te = train_test_split(
        indice,
        test_size=FRACAO_TESTE,
        stratify=classe,
        random_state=SEMENTE,
    )
    return X.iloc[i_tr], X.iloc[i_te], i_tr, i_te


FRACA_CALIBRACAO = 0.20


def _ajustar(slug: str, X: pd.DataFrame, y: np.ndarray):
    """Ajusta no grosso do treino e recalibra a chance numa fatia que o ajuste não viu."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X,
        y,
        test_size=FRACA_CALIBRACAO,
        stratify=y,
        random_state=SEMENTE,
    )
    modelo = construir(slug)
    if slug == "mlp":
        # MLPClassifier não tem class_weight. O peso por linha é o equivalente.
        pesos = compute_sample_weight(class_weight="balanced", y=y_fit)
        modelo.fit(X_fit, y_fit, clf__sample_weight=pesos)
    else:
        modelo.fit(X_fit, y_fit)
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")
    calibrado.fit(X_cal, y_cal)
    return calibrado


def treinar_todos(X_treino: pd.DataFrame, y_treino: np.ndarray) -> dict:
    matriz = X_treino[list(FEATURES)]
    return {item["slug"]: _ajustar(item["slug"], matriz, y_treino) for item in CATALOGO}
