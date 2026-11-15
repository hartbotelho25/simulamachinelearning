"""Quatro modelos no mesmo desenho de colunas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
            "É a peça que o comitê consegue auditar variável a variável."
        ),
    },
    {
        "slug": "floresta",
        "nome": "Random Forest",
        "papel": "Ensemble de árvores, estável diante de relação não linear.",
        "tecnica": (
            "Duzentas árvores. Cada uma vê uma amostra e um subconjunto de colunas. "
            "A chance é a média dos votos. A leitura dela é a curva, a matriz e a importância de cada gatilho."
        ),
    },
    {
        "slug": "boosting",
        "nome": "Gradient Boosting (XGBoost)",
        "papel": "Árvores em sequência, o padrão forte em tabela de crédito.",
        "tecnica": (
            "XGBoost com árvores rasas. A árvore seguinte corrige o erro da anterior. "
            "Captura não linearidade sem virar uma rede. A comparação com a logística mostra o que a não linearidade acrescenta."
        ),
    },
    {
        "slug": "mlp",
        "nome": "Rede neural MLP",
        "papel": "Interações entre os blocos, em camadas densas.",
        "tecnica": (
            "Perceptron de duas camadas ocultas, 64 e 32 neurônios, ativação ReLU. "
            "Os números entram padronizados. Dez por cento do treino servem só para parar cedo. "
            "A rede entra pela curva, pela matriz e pela importância aprendida em cada gatilho."
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


def treinar_todos(X_treino: pd.DataFrame, y_treino: np.ndarray) -> dict:
    matriz = X_treino[list(FEATURES)]
    modelos = {}
    for item in CATALOGO:
        modelo = construir(item["slug"])
        modelo.fit(matriz, y_treino)
        modelos[item["slug"]] = modelo
    return modelos
