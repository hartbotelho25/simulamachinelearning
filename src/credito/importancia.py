"""Importância por permutação, agregada nos sete blocos, e pesos da logística."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .dataset import BLOCO_NOME, BLOCOS, FEATURES, ROTULOS
from .politica import SEMENTE


def _pelar(modelo):
    """Sai de CalibratedClassifierCV e de FrozenEstimator até o estimador treinado."""
    atual = modelo
    for _ in range(8):
        nome = type(atual).__name__
        if nome == "CalibratedClassifierCV":
            grupo = list(getattr(atual, "calibrated_classifiers_", None) or [])
            if not grupo:
                break
            atual = grupo[0].estimator
            continue
        if nome in {"FrozenEstimator", "_CalibratedClassifier"}:
            proximo = getattr(atual, "estimator", None)
            if proximo is None or proximo is atual:
                break
            atual = proximo
            continue
        break
    return atual


def estimadores_base(modelo) -> list:
    """Um estimador por fatia calibrada. A média deles é o peso do modelo gravado."""
    if type(modelo).__name__ == "CalibratedClassifierCV":
        grupo = list(getattr(modelo, "calibrated_classifiers_", None) or [])
        if grupo:
            return [_pelar(item.estimator) for item in grupo]
    return [_pelar(modelo)]


def _vetor(modelo) -> tuple[list[str], np.ndarray]:
    passos = getattr(modelo, "steps", None)
    if passos:
        mapa = dict(passos)
        prep = mapa.get("prep", passos[0][1])
        clf = mapa.get("clf", passos[-1][1])
        nomes = [str(nome) for nome in prep.feature_names_in_]
        escala = np.asarray(getattr(prep, "scale_", np.ones(len(nomes))), dtype=float)
        escala = np.where(escala == 0, 1.0, escala)
        coef = getattr(clf, "coef_", None)
        if coef is not None:
            bruto = np.abs(np.asarray(coef, dtype=float).ravel()) / escala
        else:
            bruto = np.abs(np.asarray(clf.coefs_[0], dtype=float)).sum(axis=1) / escala
        return nomes, bruto
    nomes = [str(nome) for nome in modelo.feature_names_in_]
    bruto = np.asarray(modelo.feature_importances_, dtype=float)
    return nomes, bruto


def pesos_do_modelo(modelo) -> dict[str, float]:
    """Peso aprendido de cada gatilho, já na escala do 0/1."""
    vetores = []
    nomes: list[str] | None = None
    for base in estimadores_base(modelo):
        nomes_base, bruto = _vetor(base)
        nomes = nomes_base
        vetores.append(bruto)
    medio = np.mean(np.vstack(vetores), axis=0)
    mapa = {nome: float(valor) for nome, valor in zip(nomes or [], medio)}
    return {coluna: mapa.get(coluna, 0.0) for coluna in FEATURES}


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


def _percentuais(valores: list[float]) -> list[float]:
    """Uma casa decimal, soma 100, e a ordem segue o valor exato."""
    total = sum(valores) or 1.0
    exatos = [100.0 * valor / total for valor in valores]
    base = [int(np.floor(valor * 10 + 1e-8)) for valor in exatos]
    sobra = int(round(1000 - sum(base)))
    resto = [exato * 10 - inteiro for exato, inteiro in zip(exatos, base)]
    if sobra > 0:
        ordem = sorted(range(len(base)), key=lambda i: resto[i], reverse=True)
        for indice in ordem[:sobra]:
            base[indice] += 1
    elif sobra < 0:
        ordem = sorted(range(len(base)), key=lambda i: resto[i])
        for indice in ordem[: -sobra]:
            base[indice] -= 1
    return [inteiro / 10.0 for inteiro in base]


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
    percentuais = _percentuais([item["queda_auc"] for item in itens])
    for item, pct in zip(itens, percentuais):
        item["pct"] = pct
    itens.sort(key=lambda item: (-item["pct"], -item["queda_auc"], item["rotulo"]))
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
    percentuais = _percentuais([item["fatia"] for item in bruto])
    itens = [
        {"bloco": item["bloco"], "nome": item["nome"], "pct": pct}
        for item, pct in zip(bruto, percentuais)
    ]
    itens.sort(key=lambda item: item["pct"], reverse=True)
    return itens


def explicar_logistica(pipe) -> list[dict]:
    pipe = estimadores_base(pipe)[0]
    passos = dict(getattr(pipe, "steps", []) or [])
    prep = passos["prep"]
    clf = passos["clf"]
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
