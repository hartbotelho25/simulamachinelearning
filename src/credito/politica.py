"""Cortes de leitura desta tela.

São uma proposta de tela para o comitê, não uma política de banco.
Hart ainda define os cortes, a ação de cada faixa e o limiar de decisão.
"""

from __future__ import annotations

# Classificação lado a lado dos quatro modelos.
CORTE_MATRIZ = 0.50

# Faixa visual da própria probabilidade. O corte de SIM/NÃO continua em 50%.
VERDE_MAX = 0.20
AMARELO_MAX = 0.50

FAIXAS = (
    ("verde", "Aprovar", 0.0, VERDE_MAX),
    ("amarelo", "Atenção", VERDE_MAX, AMARELO_MAX),
    ("vermelho", "Recusar", AMARELO_MAX, 1.01),
)

TAXA_ALVO_GERADOR = 0.12
TAXA_ALVO_90D = 0.06
SEMENTE = 42
N_CLIENTES = 50_000
FRACAO_TESTE = 0.30


def nome_faixa(probabilidade: float) -> tuple[str, str]:
    """Verde abaixo de 20%, amarelo até 50%, vermelho a partir do corte de SIM."""
    for faixa, acao, ini, fim in FAIXAS:
        if ini <= probabilidade < fim:
            return faixa, acao
    return "vermelho", "Recusar"
