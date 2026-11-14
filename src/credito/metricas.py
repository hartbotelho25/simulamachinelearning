"""Métricas de teste, faixas de alerta e textos medidos."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .politica import AMARELO_MAX, CORTE_MATRIZ, FAIXAS, VERDE_MAX
from .treino import CATALOGO


def _corte(y: np.ndarray, proba: np.ndarray, corte: float) -> dict:
    pred = (proba >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "corte": corte,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acuracia": float(accuracy_score(y, pred)),
    }


def avaliar(slug: str, y: np.ndarray, proba: np.ndarray) -> dict:
    meta = next(item for item in CATALOGO if item["slug"] == slug)
    pontos = roc_curve(y, proba)
    # A curva fica leve para o gráfico, sem perder o desenho.
    fpr, tpr, _ = pontos
    passo = max(1, len(fpr) // 200)
    return {
        "slug": slug,
        "nome": meta["nome"],
        "papel": meta["papel"],
        "tecnica": meta["tecnica"],
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "matriz": _corte(y, proba, CORTE_MATRIZ),
        "roc": {
            "fpr": [float(v) for v in fpr[::passo]],
            "tpr": [float(v) for v in tpr[::passo]],
        },
    }


def faixas_alerta(y: np.ndarray, proba: np.ndarray) -> list[dict]:
    saida = []
    for faixa, acao, ini, fim in FAIXAS:
        mascara = (proba >= ini) & (proba < fim)
        n = int(mascara.sum())
        saida.append(
            {
                "faixa": faixa,
                "acao": acao,
                "de": float(ini),
                "ate": float(min(fim, 1.0)),
                "n": n,
                "fatia": float(n / len(y)) if len(y) else 0.0,
                "taxa_observada": float(y[mascara].mean()) if n else None,
                "prob_media": float(proba[mascara].mean()) if n else None,
            }
        )
    return saida


def modelo_referencia(modelos: list[dict]) -> str:
    return max(modelos, key=lambda m: (m["roc_auc"], m["slug"] == "logistica"))["slug"]


def leituras(modelos: list[dict], faixas: list[dict], referencia: str) -> dict:
    por_auc = max(modelos, key=lambda m: m["roc_auc"])
    por_recall = max(modelos, key=lambda m: m["matriz"]["recall"])
    por_precisao = max(modelos, key=lambda m: m["matriz"]["precisao"])
    taxas = [f["taxa_observada"] for f in faixas]
    separou = all(t is not None for t in taxas) and taxas[0] < taxas[1] < taxas[2]
    nome_ref = next(m["nome"] for m in modelos if m["slug"] == referencia)
    return {
        "referencia": referencia,
        "nome_referencia": nome_ref,
        "melhor_auc": {"slug": por_auc["slug"], "nome": por_auc["nome"], "roc_auc": por_auc["roc_auc"]},
        "melhor_recall": {
            "slug": por_recall["slug"],
            "nome": por_recall["nome"],
            "recall": por_recall["matriz"]["recall"],
        },
        "melhor_precisao": {
            "slug": por_precisao["slug"],
            "nome": por_precisao["nome"],
            "precisao": por_precisao["matriz"]["precisao"],
        },
        "faixas_sobem": bool(separou),
        "verde_max": VERDE_MAX,
        "amarelo_max": AMARELO_MAX,
    }
