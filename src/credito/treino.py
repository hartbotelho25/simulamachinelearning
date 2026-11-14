"""Quatro modelos no mesmo desenho de colunas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .dataset import CATEGORICAS, ESCOLARIDADES, FEATURES, NUMERICAS, OCUPACOES, PAGAMENTOS
from .politica import FRACAO_TESTE, SEMENTE

CATALOGO = [
    {
        "slug": "logistica",
        "nome": "Regressão logística",
        "papel": "Baseline e scorecard. O peso de cada variável fica visível.",
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
            "A chance é a média dos votos. A floresta não entrega um peso único por variável; "
            "a leitura dela, aqui, é a curva, a matriz e a importância por permutação."
        ),
    },
    {
        "slug": "boosting",
        "nome": "Gradient Boosting (XGBoost)",
        "papel": "Árvores em sequência, o padrão forte em tabela de crédito.",
        "tecnica": (
            "XGBoost com árvores rasas. A árvore seguinte corrige o erro da anterior. "
            "Captura não linearidade sem virar uma rede. Os ganhos internos não substituem "
            "a explicação do scorecard; a comparação com a logística mostra o que a não linearidade acrescenta."
        ),
    },
    {
        "slug": "mlp",
        "nome": "Rede neural MLP",
        "papel": "Interações entre os blocos, em camadas densas.",
        "tecnica": (
            "Perceptron de duas camadas ocultas, 64 e 32 neurônios, ativação ReLU. "
            "Os números entram padronizados. Dez por cento do treino servem só para parar cedo. "
            "Os pesos internos não se apresentam ao comitê; a rede entra pela curva, pela matriz e pela permutação."
        ),
    },
]


def _preparo() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS),
            (
                "cat",
                OneHotEncoder(
                    categories=[OCUPACOES, ESCOLARIDADES, PAGAMENTOS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAS,
            ),
        ]
    )


def construir(slug: str) -> Pipeline:
    if slug == "logistica":
        clf = LogisticRegression(max_iter=800, solver="lbfgs")
    elif slug == "floresta":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=20,
            random_state=SEMENTE,
            n_jobs=-1,
        )
    elif slug == "boosting":
        clf = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEMENTE,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    elif slug == "mlp":
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
    else:
        raise KeyError(slug)
    return Pipeline([("prep", _preparo()), ("clf", clf)])


def fatiar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X = df[FEATURES]
    y = df["inadimplente_90d"].to_numpy(dtype=int)
    return train_test_split(
        X,
        y,
        test_size=FRACAO_TESTE,
        stratify=y,
        random_state=SEMENTE,
    )


def treinar_todos(X_treino: pd.DataFrame, y_treino: np.ndarray) -> dict[str, Pipeline]:
    modelos = {}
    for item in CATALOGO:
        pipe = construir(item["slug"])
        pipe.fit(X_treino, y_treino)
        modelos[item["slug"]] = pipe
    return modelos
