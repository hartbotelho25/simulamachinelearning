"""Importância por permutação, agregada nos sete blocos, e pesos da logística."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .dataset import BLOCO_NOME, BLOCOS, ROTULOS
from .politica import SEMENTE

AMOSTRA = 1600
REPETICOES = 3


def _amostra(X: pd.DataFrame, y: np.ndarray, n: int, rng: np.random.Generator):
    n = min(n, len(y))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), max(1, int(round(n * float(y.mean())))))
    n_neg = min(len(neg), n - n_pos)
    idx = np.concatenate(
        [
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n_neg, replace=False),
        ]
    )
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def permutar(modelo, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEMENTE)
    Xs, ys = _amostra(X, y, AMOSTRA, rng)
    bruto = permutation_importance(
        modelo,
        Xs,
        ys,
        n_repeats=REPETICOES,
        scoring="roc_auc",
        random_state=SEMENTE,
        n_jobs=-1,
    )
    return {coluna: float(valor) for coluna, valor in zip(Xs.columns, bruto.importances_mean)}


def ranking_itens(importancias: dict[str, float]) -> list[dict]:
    itens = []
    for coluna, valor in importancias.items():
        if valor <= 0:
            continue
        bloco = next(nome for nome, cols in BLOCOS.items() if coluna in cols)
        itens.append(
            {
                "variavel": coluna,
                "rotulo": ROTULOS.get(coluna, coluna),
                "bloco": bloco,
                "bloco_nome": BLOCO_NOME[bloco],
                "queda_auc": round(float(valor), 5),
            }
        )
    itens.sort(key=lambda item: (-item["queda_auc"], item["rotulo"]))
    total = sum(item["queda_auc"] for item in itens) or 1.0
    for item in itens:
        item["pct"] = round(100.0 * item["queda_auc"] / total, 1)
    if itens:
        folga = round(100.0 - sum(item["pct"] for item in itens), 1)
        itens[0]["pct"] = round(itens[0]["pct"] + folga, 1)
    return itens


def por_bloco(importancias: dict[str, float]) -> list[dict]:
    somas = {bloco: 0.0 for bloco in BLOCOS}
    for coluna, valor in importancias.items():
        for bloco, cols in BLOCOS.items():
            if coluna in cols:
                somas[bloco] += float(valor)
                break
    piso = {bloco: max(0.0, valor) for bloco, valor in somas.items()}
    total = sum(piso.values()) or 1.0
    ordem = sorted(somas, key=lambda b: piso[b], reverse=True)
    return [
        {
            "bloco": bloco,
            "nome": BLOCO_NOME[bloco],
            "queda_auc": round(somas[bloco], 5),
            "fatia": round(piso[bloco] / total, 4),
        }
        for bloco in ordem
    ]


def ranking_blocos(importancias: dict[str, float]) -> list[dict]:
    bruto = por_bloco(importancias)
    itens = [
        {"bloco": item["bloco"], "nome": item["nome"], "pct": round(100.0 * item["fatia"], 1)}
        for item in bruto
    ]
    itens.sort(key=lambda item: item["pct"], reverse=True)
    if itens:
        folga = round(100.0 - sum(item["pct"] for item in itens), 1)
        itens[0]["pct"] = round(itens[0]["pct"] + folga, 1)
    return itens


def explicar_logistica(pipe) -> list[dict]:
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    nomes = list(prep.get_feature_names_out())
    coefs = clf.coef_.ravel()
    numericos: list[tuple[str, float]] = []
    grupos: dict[str, list[tuple[str, float]]] = {}
    for nome, coef in zip(nomes, coefs):
        if nome.startswith("num__"):
            coluna = nome.split("__", 1)[1]
            numericos.append((coluna, float(coef)))
        elif nome.startswith("cat__"):
            resto = nome.split("__", 1)[1]
            coluna = None
            nivel = resto
            for candidata in ("tipo_ocupacao", "escolaridade", "pagamento_fatura_predominante"):
                prefixo = candidata + "_"
                if resto.startswith(prefixo):
                    coluna = candidata
                    nivel = resto[len(prefixo) :]
                    break
            if coluna:
                grupos.setdefault(coluna, []).append((nivel, float(coef)))
    itens = [
        {
            "variavel": coluna,
            "rotulo": ROTULOS.get(coluna, coluna),
            "detalhe": "efeito de +1 desvio-padrão",
            "efeito": round(efeito, 4),
            "bloco": next(b for b, cols in BLOCOS.items() if coluna in cols),
        }
        for coluna, efeito in numericos
    ]
    for coluna, niveis in grupos.items():
        media = float(np.mean([c for _, c in niveis]))
        for nivel, coef in niveis:
            itens.append(
                {
                    "variavel": f"{coluna}={nivel}",
                    "rotulo": f"{ROTULOS.get(coluna, coluna)} · {nivel}",
                    "detalhe": "contra a média da própria categoria",
                    "efeito": round(coef - media, 4),
                    "bloco": next(b for b, cols in BLOCOS.items() if coluna in cols),
                }
            )
    itens.sort(key=lambda item: abs(item["efeito"]), reverse=True)
    return itens[:12]


def pacote_importancia(modelos: dict, X_teste: pd.DataFrame, y_teste: np.ndarray) -> dict:
    medido = {}
    for slug, modelo in modelos.items():
        colunas = permutar(modelo, X_teste, y_teste)
        medido[slug] = {
            "colunas": {k: round(v, 5) for k, v in sorted(colunas.items(), key=lambda kv: -kv[1])},
            "blocos": por_bloco(colunas),
        }
    return {
        "amostra": AMOSTRA,
        "repeticoes": REPETICOES,
        "medido": medido,
    }


def spearman_blocos(plantado: list[dict], medido: list[dict]) -> float:
    ordem_p = {item["bloco"]: i for i, item in enumerate(plantado)}
    ordem_m = {item["bloco"]: i for i, item in enumerate(medido)}
    blocos = list(ordem_p)
    rp = np.array([ordem_p[b] for b in blocos], dtype=float)
    rm = np.array([ordem_m[b] for b in blocos], dtype=float)
    rp -= rp.mean()
    rm -= rm.mean()
    den = float(np.sqrt((rp**2).sum() * (rm**2).sum()))
    if den == 0:
        return 0.0
    return float((rp * rm).sum() / den)
