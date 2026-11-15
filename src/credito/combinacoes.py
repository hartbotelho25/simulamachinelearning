"""Combinações de gatilhos preventivos medidas no teste."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _regras() -> list[dict]:
    return [
        {
            "id": "especial_longo_e_rotativo",
            "titulo": "Especial por meses e rotativo frequente",
            "sinais": [
                "Cheque especial por mais de 3 meses",
                "Rotativo mais de 2 vezes",
            ],
            "mascara": lambda d: (d["uso_cheque_especial_mais_3m_em_6m"] == 1)
            & (d["rotativo_mais_2x_6m"] == 1),
        },
        {
            "id": "cheque_e_fatura",
            "titulo": "Mais de um cheque sem fundo e fatura paga com o especial",
            "sinais": [
                "Mais de 1 cheque sem fundo",
                "Pagou a fatura com o cheque especial",
            ],
            "mascara": lambda d: (d["mais_de_um_cheque_sem_fundo_12m"] == 1)
            & (d["pagou_fatura_com_cheque_especial"] == 1),
        },
        {
            "id": "protesto_e_reneg",
            "titulo": "Três protestos ou mais e renegociação ativa",
            "sinais": [
                "3 ou mais protestos",
                "Renegociação ativa",
            ],
            "mascara": lambda d: (d["protesto_3_ou_mais"] == 1) & (d["renegociacao_ativa"] == 1),
        },
        {
            "id": "reserva_e_vinculo",
            "titulo": "Resgate total da reserva e queda de vínculo",
            "sinais": [
                "Zerou a reserva no banco",
                "Queda do vínculo principal",
            ],
            "mascara": lambda d: (d["resgate_total_reserva"] == 1) & (d["queda_vinculo_principal"] == 1),
        },
        {
            "id": "teto_e_parcelamento",
            "titulo": "Cartão no teto repetido e fatura parcelada",
            "sinais": [
                "Cartão acima de 90% mais de 3 vezes",
                "Parcelou a fatura",
            ],
            "mascara": lambda d: (d["cartao_acima_90_mais_3x_6m"] == 1) & (d["parcelamento_fatura_6m"] == 1),
        },
        {
            "id": "emergencia_e_novo",
            "titulo": "Crédito emergencial e empréstimo novo",
            "sinais": [
                "Adiantamento emergencial",
                "Novo empréstimo em 6 meses",
            ],
            "mascara": lambda d: (d["adiantamento_emergencial"] == 1) & (d["novo_emprestimo_6m"] == 1),
        },
    ]


def medir_combinacoes(
    X: pd.DataFrame,
    y_inad: np.ndarray,
    y_90: np.ndarray,
    prob_inad: np.ndarray,
    prob_90: np.ndarray,
) -> list[dict]:
    inad = np.asarray(y_inad).astype(int)
    grave = np.asarray(y_90).astype(int)
    base_inad = float(inad.mean()) if len(inad) else 0.0
    base_90 = float(grave.mean()) if len(grave) else 0.0
    saida = []
    for regra in _regras():
        mascara = np.asarray(regra["mascara"](X), dtype=bool)
        n = int(mascara.sum())
        if n < 20:
            continue
        taxa_inad = float(inad[mascara].mean())
        taxa_90 = float(grave[mascara].mean())
        piso_inad = max(base_inad * 1.8, 0.22)
        piso_90 = max(base_90 * 1.8, 0.10)
        saida.append(
            {
                "id": regra["id"],
                "titulo": regra["titulo"],
                "sinais": regra["sinais"],
                "n": n,
                "base_inad": round(base_inad, 4),
                "base_90": round(base_90, 4),
                "taxa_inad": round(taxa_inad, 4),
                "taxa_90": round(taxa_90, 4),
                "prev_inad": round(float(np.asarray(prob_inad)[mascara].mean()), 4),
                "prev_90": round(float(np.asarray(prob_90)[mascara].mean()), 4),
                "critica": bool(taxa_inad >= piso_inad or taxa_90 >= piso_90),
            }
        )
    saida.sort(key=lambda item: item["taxa_inad"], reverse=True)
    return saida


# Degraus da carteira de teste. 0, 1 e 2 ficam no chão; o salto
# aparece a partir de 4 e segue aberto até 11 ou mais.
DEGRAUS = (
    ("0", 0, 0),
    ("1–2", 1, 2),
    ("3–4", 3, 4),
    ("5–6", 5, 6),
    ("7–8", 7, 8),
    ("9–10", 9, 10),
    ("11+", 11, 10**9),
)


def medir_escada(sinais, inad, grave) -> list[dict]:
    """Taxa observada no teste conforme os gatilhos ligados se acumulam."""
    contagem = np.asarray(sinais, dtype=int)
    y_inad = np.asarray(inad, dtype=int)
    y_90 = np.asarray(grave, dtype=int)
    base_inad = float(y_inad.mean()) if len(y_inad) else 0.0
    base_90 = float(y_90.mean()) if len(y_90) else 0.0
    saida = []
    for rotulo, inicio, fim in DEGRAUS:
        mascara = (contagem >= inicio) & (contagem <= fim)
        n = int(mascara.sum())
        taxa_inad = float(y_inad[mascara].mean()) if n else 0.0
        taxa_90 = float(y_90[mascara].mean()) if n else 0.0
        saida.append(
            {
                "rotulo": rotulo,
                "n": n,
                "taxa_inad": taxa_inad,
                "taxa_90": taxa_90,
                "base_inad": base_inad,
                "base_90": base_90,
            }
        )
    return saida
