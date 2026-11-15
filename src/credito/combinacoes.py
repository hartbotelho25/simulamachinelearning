"""Pares e trios medidos no teste, com a chance que os modelos deram."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import CATEGORICAS


def sinais_das_colunas(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Sentido em que o item acompanha o calote no teste."""
    alvo = pd.Series(np.asarray(y), index=X.index)
    base = float(alvo.mean())
    saida = {}
    for coluna in X.columns:
        if coluna in CATEGORICAS:
            taxas = alvo.groupby(X[coluna], observed=True).mean()
            nivel = str(taxas.idxmax())
            saida[coluna] = {
                "sentido": "categoria",
                "nivel": nivel,
                "taxa": round(float(taxas.max()), 4),
                "acima_da_base": bool(float(taxas.max()) > base * 1.15),
            }
            continue
        serie = X[coluna].to_numpy(dtype=float)
        if float(np.std(serie)) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(serie, alvo.to_numpy(dtype=float))[0, 1])
        saida[coluna] = {"sentido": "sobe" if corr >= 0 else "desce", "corr": round(corr, 4)}
    return saida


def _regras() -> list[dict]:
    return [
        {
            "id": "cheque_sem_fundo",
            "titulo": "Cheque devolvido e uso contínuo do especial",
            "sinais": [
                "Ao menos 1 cheque devolvido em 12 meses",
                "60 dias ou mais no cheque especial no ano",
            ],
            "mascara": lambda d: (d["qtd_cheques_devolvidos_12m"] >= 1)
            & (d["dias_utilizando_cheque_especial_12m"] >= 60),
        },
        {
            "id": "cartao_pago_com_especial",
            "titulo": "Cartão pago com cheque especial e atraso recente",
            "sinais": [
                "Pagou a fatura com cheque especial",
                "Ao menos 1 atraso de 30 dias no cartão",
            ],
            "mascara": lambda d: (d["flag_pagamento_cartao_com_cheque_especial"] == 1)
            & (d["qtd_vezes_atraso_cartao_30d_12m"] >= 1),
        },
        {
            "id": "fluxo_e_minimo",
            "titulo": "Parcelas pesadas no fluxo e fatura no mínimo",
            "sinais": [
                "Parcelas consomem 40% ou mais das entradas da conta",
                "3 ou mais pagamentos mínimos no ano",
            ],
            "mascara": lambda d: (d["comprometimento_fluxo_entradas"] >= 0.40)
            & (d["qtd_vezes_pagamento_minimo_12m"] >= 3),
        },
        {
            "id": "atraso_e_restricao",
            "titulo": "Atraso interno com restrição no mercado",
            "sinais": [
                "FPD ou ocorrência de 30 a 60 dias",
                "Protesto ou inadimplência em outra instituição",
            ],
            "mascara": lambda d: (
                (d["flag_historico_fpd"] == 1) | (d["qtd_ocorrencias_atraso_30_60d_12m"] >= 1)
            )
            & ((d["qtd_protestos_historico"] >= 1) | (d["valor_total_inadimplencia_outros_mercados"] > 0)),
        },
        {
            "id": "trio_cheque_cartao_atraso",
            "titulo": "Cheque sem fundo, cartão no especial e atraso de 30 a 60 dias",
            "sinais": [
                "Ao menos 1 cheque devolvido",
                "Pagou a fatura com cheque especial",
                "Ao menos 1 ocorrência de 30 a 60 dias",
            ],
            "mascara": lambda d: (d["qtd_cheques_devolvidos_12m"] >= 1)
            & (d["flag_pagamento_cartao_com_cheque_especial"] == 1)
            & (d["qtd_ocorrencias_atraso_30_60d_12m"] >= 1),
        },
        {
            "id": "especial_e_minimo",
            "titulo": "Especial estourado e mínimo recorrente",
            "sinais": [
                "Uso atual do cheque especial de 60% ou mais",
                "4 ou mais pagamentos mínimos no ano",
            ],
            "mascara": lambda d: (d["utilizacao_cheque_especial_atual"] >= 0.60)
            & (d["qtd_vezes_pagamento_minimo_12m"] >= 4),
        },
    ]


def medir_combinacoes(
    X: pd.DataFrame,
    y: np.ndarray,
    probas: dict[str, np.ndarray],
    referencia: str,
) -> list[dict]:
    alvo = np.asarray(y).astype(int)
    base = float(alvo.mean()) if len(alvo) else 0.0
    saida = []
    for regra in _regras():
        mascara = np.asarray(regra["mascara"](X), dtype=bool)
        n = int(mascara.sum())
        if n < 40:
            continue
        observada = float(alvo[mascara].mean())
        previstas = {
            slug: round(float(np.asarray(probas[slug])[mascara].mean()), 4)
            for slug in probas
        }
        piso = max(base * 1.8, 0.25)
        saida.append(
            {
                "id": regra["id"],
                "titulo": regra["titulo"],
                "sinais": regra["sinais"],
                "n": n,
                "taxa_base": round(base, 4),
                "taxa_observada": round(observada, 4),
                "previstas": previstas,
                "prevista_referencia": previstas[referencia],
                "critica": bool(observada >= piso),
            }
        )
    saida.sort(key=lambda item: item["taxa_observada"], reverse=True)
    return saida
