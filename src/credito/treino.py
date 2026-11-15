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

from .dataset import FEATURES, aplicar_exclusivo
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


def matriz_exclusiva(amostra: pd.DataFrame) -> pd.DataFrame:
    """Zera a ocorrência única quando a reincidência está ligada. Fica fora do estimador."""
    colunas = list(FEATURES)
    return aplicar_exclusivo(amostra[colunas])[colunas]


def construir(slug: str) -> Pipeline:
    if slug == "logistica":
        clf = LogisticRegression(max_iter=800, solver="lbfgs", C=20.0)
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
    return Pipeline([("prep", StandardScaler()), ("clf", clf)])


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


def treinar_todos(X_treino: pd.DataFrame, y_treino: np.ndarray) -> dict[str, Pipeline]:
    matriz = matriz_exclusiva(X_treino)
    modelos = {}
    for item in CATALOGO:
        pipe = construir(item["slug"])
        pipe.fit(matriz, y_treino)
        modelos[item["slug"]] = pipe
    return modelos
