"""Cortes de leitura desta tela.

São uma proposta de tela para o comitê, não uma política de banco.
Hart ainda define os cortes, a ação de cada faixa e o limiar de decisão.
"""

from __future__ import annotations

# Classificação lado a lado dos quatro modelos.
CORTE_MATRIZ = 0.50

# Alerta antecipado. Abaixo do verde: aprovar. Até o amarelo: garantia. Acima: recusar.
VERDE_MAX = 0.08
AMARELO_MAX = 0.20

FAIXAS = (
    ("verde", "Aprovar", 0.0, VERDE_MAX),
    ("amarelo", "Exigir garantia", VERDE_MAX, AMARELO_MAX),
    ("vermelho", "Recusar", AMARELO_MAX, 1.01),
)

TAXA_ALVO_GERADOR = 0.12
SEMENTE = 42
N_CLIENTES = 50_000
FRACAO_TESTE = 0.30


def nome_faixa(probabilidade: float) -> tuple[str, str]:
    for faixa, acao, ini, fim in FAIXAS:
        if ini <= probabilidade < fim:
            return faixa, acao
    return "vermelho", "Recusar"
