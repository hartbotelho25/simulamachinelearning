"""Base sintética de crédito de varejo.

A renda declarada não entra: o banco quase nunca a vê com precisão.
O mercado externo fica em protesto e restrição em outras instituições.
O alvo nasce dos campos observados, com ruído. Um fator escondido só organiza
as correlações do livro; ele não entra na equação do alvo nem no treino.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .politica import TAXA_ALVO_GERADOR

COLUNAS = [
    "id_cliente",
    "idade",
    "tempo_relacionamento_meses",
    "tipo_ocupacao",
    "escolaridade",
    "possui_imovel_proprio",
    "fluxo_medio_entradas_3m",
    "saldo_medio_conta_12m",
    "saldo_medio_conta_3m",
    "limite_cheque_especial",
    "dias_utilizando_cheque_especial_30d",
    "dias_utilizando_cheque_especial_12m",
    "utilizacao_cheque_especial_atual",
    "qtd_cheques_devolvidos_12m",
    "limite_total_cartao",
    "fatura_media_3m",
    "fatura_media_12m",
    "taxa_crescimento_gastos_cartao",
    "pagamento_fatura_predominante",
    "qtd_vezes_pagamento_minimo_12m",
    "qtd_vezes_atraso_cartao_30d_12m",
    "valor_saque_cartao_credito_6m",
    "flag_pagamento_cartao_com_cheque_especial",
    "qtd_emprestimos_ativos",
    "qtd_novos_emprestimos_ultimos_6m",
    "parcela_total_comprometida",
    "comprometimento_fluxo_entradas",
    "max_atraso_emprestimos_historico",
    "taxa_juros_media_contratos_ativos",
    "qtd_ocorrencias_atraso_30_60d_12m",
    "flag_historico_fpd",
    "possui_renegociacao_ativa",
    "qtd_protestos_historico",
    "valor_total_inadimplencia_outros_mercados",
    "inadimplente_90d",
]

BLOCOS = {
    "cadastral": [
        "idade",
        "tempo_relacionamento_meses",
        "tipo_ocupacao",
        "escolaridade",
        "possui_imovel_proprio",
    ],
    "fluxo": [
        "fluxo_medio_entradas_3m",
        "saldo_medio_conta_12m",
        "saldo_medio_conta_3m",
    ],
    "cheque": [
        "limite_cheque_especial",
        "dias_utilizando_cheque_especial_30d",
        "dias_utilizando_cheque_especial_12m",
        "utilizacao_cheque_especial_atual",
        "qtd_cheques_devolvidos_12m",
    ],
    "cartao": [
        "limite_total_cartao",
        "fatura_media_3m",
        "fatura_media_12m",
        "taxa_crescimento_gastos_cartao",
        "pagamento_fatura_predominante",
        "qtd_vezes_pagamento_minimo_12m",
        "qtd_vezes_atraso_cartao_30d_12m",
        "valor_saque_cartao_credito_6m",
        "flag_pagamento_cartao_com_cheque_especial",
    ],
    "endividamento": [
        "qtd_emprestimos_ativos",
        "qtd_novos_emprestimos_ultimos_6m",
        "parcela_total_comprometida",
        "comprometimento_fluxo_entradas",
        "max_atraso_emprestimos_historico",
        "taxa_juros_media_contratos_ativos",
    ],
    "regua": [
        "qtd_ocorrencias_atraso_30_60d_12m",
        "flag_historico_fpd",
        "possui_renegociacao_ativa",
    ],
    "bureau": [
        "qtd_protestos_historico",
        "valor_total_inadimplencia_outros_mercados",
    ],
}

BLOCO_NOME = {
    "cadastral": "Cadastral e sociodemográfico",
    "fluxo": "Fluxo de caixa e liquidez",
    "cheque": "Conta corrente e cheque especial",
    "cartao": "Cartão de crédito e consumo",
    "endividamento": "Endividamento e empréstimos",
    "regua": "Régua interna de atrasos",
    "bureau": "Restrições no mercado",
}

ROTULOS = {
    "idade": "Idade",
    "tempo_relacionamento_meses": "Tempo de relacionamento (meses)",
    "tipo_ocupacao": "Ocupação",
    "escolaridade": "Escolaridade",
    "possui_imovel_proprio": "Imóvel próprio",
    "fluxo_medio_entradas_3m": "Fluxo médio de entradas, 3 meses",
    "saldo_medio_conta_12m": "Saldo médio da conta, 12 meses",
    "saldo_medio_conta_3m": "Saldo médio da conta, 3 meses",
    "limite_cheque_especial": "Limite do cheque especial",
    "dias_utilizando_cheque_especial_30d": "Dias no cheque especial, 30 dias",
    "dias_utilizando_cheque_especial_12m": "Dias no cheque especial, 12 meses",
    "utilizacao_cheque_especial_atual": "Uso atual do cheque especial",
    "qtd_cheques_devolvidos_12m": "Cheques devolvidos, 12 meses",
    "limite_total_cartao": "Limite total do cartão",
    "fatura_media_3m": "Fatura média, 3 meses",
    "fatura_media_12m": "Fatura média, 12 meses",
    "taxa_crescimento_gastos_cartao": "Crescimento dos gastos no cartão",
    "pagamento_fatura_predominante": "Pagamento predominante da fatura",
    "qtd_vezes_pagamento_minimo_12m": "Vezes no pagamento mínimo, 12 meses",
    "qtd_vezes_atraso_cartao_30d_12m": "Atrasos de 30 dias no cartão, 12 meses",
    "valor_saque_cartao_credito_6m": "Saque no cartão, 6 meses",
    "flag_pagamento_cartao_com_cheque_especial": "Pagou cartão com cheque especial",
    "qtd_emprestimos_ativos": "Empréstimos ativos",
    "qtd_novos_emprestimos_ultimos_6m": "Empréstimos novos, 6 meses",
    "parcela_total_comprometida": "Parcela total comprometida",
    "comprometimento_fluxo_entradas": "Parcelas sobre o fluxo de entradas",
    "max_atraso_emprestimos_historico": "Maior atraso em empréstimos (dias)",
    "taxa_juros_media_contratos_ativos": "Juros médio dos contratos ativos",
    "qtd_ocorrencias_atraso_30_60d_12m": "Ocorrências de 30 a 60 dias, 12 meses",
    "flag_historico_fpd": "Histórico de primeiro atraso",
    "possui_renegociacao_ativa": "Renegociação ativa",
    "qtd_protestos_historico": "Protestos",
    "valor_total_inadimplencia_outros_mercados": "Inadimplência em outros mercados",
    "inadimplente_90d": "Inadimplente 90 dias",
}

OCUPACOES = ["clt", "servidor", "autonomo", "empresario", "aposentado", "informal"]
PESO_OCUPACAO = [0.46, 0.12, 0.16, 0.08, 0.10, 0.08]
# Porte interno para nascer fluxo, limite e parcela juntos.
# Não é renda declarada e não entra no modelo.
PORTE_OCUPACAO = {
    "clt": 4500.0,
    "servidor": 7200.0,
    "autonomo": 3900.0,
    "empresario": 9800.0,
    "aposentado": 2800.0,
    "informal": 1900.0,
}
ESCOLARIDADES = ["fundamental", "medio", "superior", "pos"]
PESO_ESCOLARIDADE = [0.18, 0.42, 0.32, 0.08]
ESCOLARIDADE_ORD = {nome: i for i, nome in enumerate(ESCOLARIDADES)}
PAGAMENTOS = ["integral", "parcial", "minimo"]

# Efeito no logit, depois do padronizar. Positivo empurra para o calote.
# É o processo gerador, não uma descoberta de mercado.
SINAL_GERADOR = {
    "ocorrencias_30_60": 0.70,
    "fpd": 0.55,
    "renegociacao": 0.35,
    "inad_outros": 0.70,
    "protestos": 0.55,
    "comprometimento": 0.50,
    "atraso_emprestimo": 0.40,
    "novos_emprestimos": 0.22,
    "juros": 0.15,
    "atraso_cartao": 0.48,
    "pag_minimo": 0.32,
    "flag_pag_cheque": 0.22,
    "crescimento_cartao": 0.12,
    "saque_rel": 0.18,
    "uso_cheque": 0.38,
    "dias_cheque": 0.28,
    "cheques": 0.42,
    "caixa_apertado": 0.30,
    "queda_saldo": 0.18,
    "tempo_rel": -0.16,
    "imovel": -0.22,
    "escolaridade": -0.08,
    "informal": 0.12,
    "servidor": -0.08,
}

SINAL_BLOCO = {
    "ocorrencias_30_60": "regua",
    "fpd": "regua",
    "renegociacao": "regua",
    "inad_outros": "bureau",
    "protestos": "bureau",
    "comprometimento": "endividamento",
    "atraso_emprestimo": "endividamento",
    "novos_emprestimos": "endividamento",
    "juros": "endividamento",
    "atraso_cartao": "cartao",
    "pag_minimo": "cartao",
    "flag_pag_cheque": "cartao",
    "crescimento_cartao": "cartao",
    "saque_rel": "cartao",
    "uso_cheque": "cheque",
    "dias_cheque": "cheque",
    "cheques": "cheque",
    "caixa_apertado": "fluxo",
    "queda_saldo": "fluxo",
    "tempo_rel": "cadastral",
    "imovel": "cadastral",
    "escolaridade": "cadastral",
    "informal": "cadastral",
    "servidor": "cadastral",
}

# Ruído alto de propósito: os campos já carregam o fator escondido.
# Sem ele, o alvo fica fácil demais para um livro de crédito.
RUIDO_LOGIT = 5.0

FEATURES = [c for c in COLUNAS if c not in {"id_cliente", "inadimplente_90d"}]
NUMERICAS = [
    c
    for c in FEATURES
    if c not in {"tipo_ocupacao", "escolaridade", "pagamento_fatura_predominante"}
]
CATEGORICAS = ["tipo_ocupacao", "escolaridade", "pagamento_fatura_predominante"]

# Limites do gerador. O simulador usa os mesmos intervalos.
LIMITES = {
    "idade": (21, 78, 1),
    "tempo_relacionamento_meses": (1, 360, 1),
    "possui_imovel_proprio": (0, 1, 1),
    "fluxo_medio_entradas_3m": (200.0, 120000.0, 100.0),
    "saldo_medio_conta_12m": (-20000.0, 80000.0, 100.0),
    "saldo_medio_conta_3m": (-20000.0, 80000.0, 100.0),
    "limite_cheque_especial": (0.0, 40000.0, 100.0),
    "dias_utilizando_cheque_especial_30d": (0, 30, 1),
    "dias_utilizando_cheque_especial_12m": (0, 365, 1),
    "utilizacao_cheque_especial_atual": (0.0, 1.8, 0.01),
    "qtd_cheques_devolvidos_12m": (0, 12, 1),
    "limite_total_cartao": (0.0, 120000.0, 100.0),
    "fatura_media_3m": (0.0, 80000.0, 50.0),
    "fatura_media_12m": (0.0, 80000.0, 50.0),
    "taxa_crescimento_gastos_cartao": (-0.8, 1.5, 0.01),
    "qtd_vezes_pagamento_minimo_12m": (0, 12, 1),
    "qtd_vezes_atraso_cartao_30d_12m": (0, 12, 1),
    "valor_saque_cartao_credito_6m": (0.0, 30000.0, 50.0),
    "flag_pagamento_cartao_com_cheque_especial": (0, 1, 1),
    "qtd_emprestimos_ativos": (0, 8, 1),
    "qtd_novos_emprestimos_ultimos_6m": (0, 5, 1),
    "parcela_total_comprometida": (0.0, 20000.0, 50.0),
    "comprometimento_fluxo_entradas": (0.0, 1.5, 0.01),
    "max_atraso_emprestimos_historico": (0, 180, 1),
    "taxa_juros_media_contratos_ativos": (0.0, 12.0, 0.1),
    "qtd_ocorrencias_atraso_30_60d_12m": (0, 8, 1),
    "flag_historico_fpd": (0, 1, 1),
    "possui_renegociacao_ativa": (0, 1, 1),
    "qtd_protestos_historico": (0, 6, 1),
    "valor_total_inadimplencia_outros_mercados": (0.0, 50000.0, 100.0),
}


def conferir_esquema(df: pd.DataFrame) -> None:
    if list(df.columns) != COLUNAS:
        raise ValueError("A ordem das colunas não fecha com o esquema.")
    proibidas = {"renda_declarada", "score_bureau_atual", "qtd_consultas_bureau_3m"}
    if proibidas & set(COLUNAS):
        raise ValueError("Renda declarada e score de bureau ficaram de fora de propósito.")
    vistas = [c for cols in BLOCOS.values() for c in cols]
    if sorted(vistas) != sorted(FEATURES):
        raise ValueError("Os blocos não cobrem as colunas de treino.")
    if df.shape[1] != len(COLUNAS):
        raise ValueError("A base não tem o número de colunas do esquema.")
    if df.isna().any().any():
        raise ValueError("A base sintética saiu com vazio.")


def _clip(serie: np.ndarray, baixo, alto) -> np.ndarray:
    return np.clip(serie, baixo, alto)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def _poisson_clip(rng: np.random.Generator, lam: np.ndarray, alto: int) -> np.ndarray:
    lam = np.clip(lam, 0.02, None)
    return np.minimum(rng.poisson(lam), alto).astype(int)


def _drivers(df: pd.DataFrame) -> pd.DataFrame:
    fluxo = df["fluxo_medio_entradas_3m"].to_numpy(dtype=float)
    saldo12 = df["saldo_medio_conta_12m"].to_numpy(dtype=float)
    saldo3 = df["saldo_medio_conta_3m"].to_numpy(dtype=float)
    base = np.maximum(fluxo, 1.0)
    caixa = (saldo3 < 0).astype(float) + np.clip((0.25 * fluxo - saldo3) / base, 0, 3)
    return pd.DataFrame(
        {
            "ocorrencias_30_60": df["qtd_ocorrencias_atraso_30_60d_12m"].to_numpy(dtype=float),
            "fpd": df["flag_historico_fpd"].to_numpy(dtype=float),
            "renegociacao": df["possui_renegociacao_ativa"].to_numpy(dtype=float),
            "inad_outros": df["valor_total_inadimplencia_outros_mercados"].to_numpy(dtype=float) / base,
            "protestos": df["qtd_protestos_historico"].to_numpy(dtype=float),
            "comprometimento": df["comprometimento_fluxo_entradas"].to_numpy(dtype=float),
            "atraso_emprestimo": df["max_atraso_emprestimos_historico"].to_numpy(dtype=float) / 30.0,
            "novos_emprestimos": df["qtd_novos_emprestimos_ultimos_6m"].to_numpy(dtype=float),
            "juros": df["taxa_juros_media_contratos_ativos"].to_numpy(dtype=float),
            "atraso_cartao": df["qtd_vezes_atraso_cartao_30d_12m"].to_numpy(dtype=float),
            "pag_minimo": df["qtd_vezes_pagamento_minimo_12m"].to_numpy(dtype=float),
            "flag_pag_cheque": df["flag_pagamento_cartao_com_cheque_especial"].to_numpy(dtype=float),
            "crescimento_cartao": df["taxa_crescimento_gastos_cartao"].to_numpy(dtype=float),
            "saque_rel": df["valor_saque_cartao_credito_6m"].to_numpy(dtype=float) / base,
            "uso_cheque": df["utilizacao_cheque_especial_atual"].to_numpy(dtype=float),
            "dias_cheque": df["dias_utilizando_cheque_especial_12m"].to_numpy(dtype=float) / 30.0,
            "cheques": df["qtd_cheques_devolvidos_12m"].to_numpy(dtype=float),
            "caixa_apertado": caixa,
            "queda_saldo": (saldo12 - saldo3) / base,
            "tempo_rel": df["tempo_relacionamento_meses"].to_numpy(dtype=float) / 12.0,
            "imovel": df["possui_imovel_proprio"].to_numpy(dtype=float),
            "escolaridade": df["escolaridade"].map(ESCOLARIDADE_ORD).to_numpy(dtype=float),
            "informal": (df["tipo_ocupacao"] == "informal").to_numpy(dtype=float),
            "servidor": (df["tipo_ocupacao"] == "servidor").to_numpy(dtype=float),
        }
    )


def _betas() -> np.ndarray:
    return np.array([SINAL_GERADOR[c] for c in SINAL_GERADOR], dtype=float)


def _intercepto(eta: np.ndarray, taxa: float, ruido: float, rng: np.random.Generator) -> float:
    amostra = rng.normal(0.0, ruido, size=len(eta))
    lo, hi = -14.0, 6.0
    for _ in range(52):
        meio = (lo + hi) / 2.0
        media = _sigmoid(eta + amostra + meio).mean()
        if media > taxa:
            hi = meio
        else:
            lo = meio
    return float((lo + hi) / 2.0)


def _parametros(drivers: pd.DataFrame, rng: np.random.Generator) -> dict:
    colunas = list(SINAL_GERADOR)
    matriz = drivers[colunas].to_numpy(dtype=float)
    media = matriz.mean(axis=0)
    desvio = matriz.std(axis=0)
    desvio = np.where(desvio < 1e-6, 1.0, desvio)
    z = (matriz - media) / desvio
    eta = z @ _betas()
    return {
        "colunas": colunas,
        "media": media.tolist(),
        "desvio": desvio.tolist(),
        "betas": _betas().tolist(),
        "intercepto": _intercepto(eta, TAXA_ALVO_GERADOR, RUIDO_LOGIT, rng),
        "ruido": RUIDO_LOGIT,
        "taxa_alvo": TAXA_ALVO_GERADOR,
    }


def _rotular(drivers: pd.DataFrame, parametros: dict, rng: np.random.Generator) -> np.ndarray:
    colunas = parametros["colunas"]
    matriz = drivers[colunas].to_numpy(dtype=float)
    media = np.array(parametros["media"], dtype=float)
    desvio = np.array(parametros["desvio"], dtype=float)
    betas = np.array(parametros["betas"], dtype=float)
    z = (matriz - media) / desvio
    eta = z @ betas + float(parametros["intercepto"])
    eta = eta + rng.normal(0.0, float(parametros["ruido"]), size=len(drivers))
    prob = _sigmoid(eta)
    return rng.binomial(1, prob).astype(int)


def sinal_plantado_por_bloco() -> list[dict]:
    bruto = {bloco: 0.0 for bloco in BLOCOS}
    for nome, beta in SINAL_GERADOR.items():
        bruto[SINAL_BLOCO[nome]] += abs(float(beta))
    total = sum(bruto.values()) or 1.0
    ordem = sorted(bruto, key=bruto.get, reverse=True)
    return [
        {
            "bloco": bloco,
            "nome": BLOCO_NOME[bloco],
            "peso_absoluto": round(bruto[bloco], 4),
            "fatia": round(bruto[bloco] / total, 4),
        }
        for bloco in ordem
    ]


def gerar_base(n: int, seed: int, parametros: dict | None = None, prefixo_id: str = "C") -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    stress = rng.normal(0.0, 1.0, n)
    ocupacao = rng.choice(OCUPACOES, size=n, p=PESO_OCUPACAO)
    escolaridade = rng.choice(ESCOLARIDADES, size=n, p=PESO_ESCOLARIDADE)
    idade = _clip(rng.normal(42, 12, n), 21, 78)
    tempo = _clip(rng.gamma(2.2, 28, n) + np.maximum(idade - 30, 0) * 0.4, 1, 360)
    esc_ord = np.array([ESCOLARIDADE_ORD[e] for e in escolaridade])
    p_imovel = _sigmoid(-0.4 + 0.35 * esc_ord + 0.02 * (idade - 40) - 0.15 * stress)
    imovel = rng.binomial(1, np.clip(p_imovel, 0.05, 0.9))

    porte_base = np.array([PORTE_OCUPACAO[o] for o in ocupacao])
    porte = _clip(porte_base * rng.lognormal(0.0, 0.36, n), 700, 80000)
    mult_fluxo = np.ones(n)
    mult_fluxo = np.where(ocupacao == "informal", 0.70, mult_fluxo)
    mult_fluxo = np.where(ocupacao == "autonomo", 0.86, mult_fluxo)
    fluxo = _clip(porte * mult_fluxo * rng.lognormal(-0.04 - 0.08 * stress, 0.28, n), 200, 120000)
    saldo12 = _clip(porte * rng.normal(0.85 - 0.35 * stress, 0.65, n), -20000, 80000)
    saldo3 = _clip(0.62 * saldo12 + porte * rng.normal(-0.05 * stress, 0.28, n), -20000, 80000)

    limite_cheque = _clip(porte * rng.uniform(0.05, 0.85, n) * (rng.random(n) > 0.12), 0, 40000)
    uso = _sigmoid(-0.6 + 0.85 * stress + rng.normal(0, 0.35, n))
    uso = np.where(limite_cheque <= 0, 0.0, _clip(uso * 1.15, 0, 1.8))
    dias30 = _poisson_clip(rng, uso * 14 + np.maximum(stress, 0) * 3, 30)
    dias12 = _clip(dias30 * rng.uniform(5.5, 13.5, n) + np.maximum(stress, 0) * 18, 0, 365).astype(int)
    cheques = _poisson_clip(rng, 0.15 + uso * 1.4 + np.maximum(stress, 0) * 0.8, 12)

    limite_cartao = _clip(porte * rng.uniform(0.5, 3.2, n), 0, 120000)
    fatura12 = _clip(limite_cartao * _clip(rng.normal(0.38 + 0.12 * stress, 0.16, n), 0.05, 0.95), 0, 80000)
    fatura3 = _clip(fatura12 * rng.lognormal(0.04 * stress, 0.18, n), 0, 80000)
    crescimento = _clip(np.where(fatura12 > 1, fatura3 / fatura12 - 1.0, 0.0), -0.8, 1.5)
    p_min = _sigmoid(-1.3 + 1.1 * stress)
    p_parc = _sigmoid(-0.2 + 0.4 * stress) * (1 - p_min)
    u = rng.random(n)
    pagamento = np.where(u < p_min, "minimo", np.where(u < p_min + p_parc, "parcial", "integral"))
    qtd_min = _poisson_clip(rng, p_min * 7 + np.maximum(stress, 0), 12)
    atraso_cartao = _poisson_clip(rng, 0.2 + np.maximum(stress, 0) * 1.3 + (pagamento == "minimo") * 0.8, 12)
    saque = _clip(porte * _clip(rng.normal(0.02 + 0.06 * np.maximum(stress, 0), 0.05, n), 0, None), 0, 30000)
    flag_cheque = rng.binomial(1, _sigmoid(-1.6 + 1.3 * uso + 0.5 * (pagamento == "minimo")))

    ativos = _poisson_clip(rng, 0.55 + 0.35 * np.maximum(stress, 0), 8)
    novos = _poisson_clip(rng, 0.18 + 0.28 * np.maximum(stress, 0), 5)
    parcela = _clip(porte * _clip(rng.normal(0.08 + 0.07 * ativos + 0.04 * stress, 0.06, n), 0, None), 0, 20000)
    comprometimento = _clip(parcela / np.maximum(fluxo, 200.0), 0, 1.5)
    atraso_emp = _poisson_clip(rng, 4 + np.maximum(stress, 0) * 22, 180)
    juros = _clip(rng.normal(2.4 + 0.9 * stress + 0.15 * ativos, 0.7, n), 0.8, 12.0)

    ocorrencias = _poisson_clip(rng, 0.12 + np.maximum(stress, 0) * 0.85 + atraso_cartao * 0.08, 8)
    p_fpd = _sigmoid(-2.0 + 0.7 * stress + 0.35 * (novos > 0) + 0.25 * (tempo < 12))
    fpd = rng.binomial(1, p_fpd)
    p_reneg = _sigmoid(-2.3 + 0.8 * stress + 0.4 * (ocorrencias > 0))
    reneg = rng.binomial(1, p_reneg)

    protestos = _poisson_clip(rng, 0.05 + np.maximum(stress, 0) * 0.28, 6)
    inad_outros = _clip(
        rng.lognormal(6.5 + 0.7 * stress, 0.8, n) * (rng.random(n) < _sigmoid(-1.4 + 0.9 * stress)),
        0,
        50000,
    )

    df = pd.DataFrame(
        {
            "idade": idade.round().astype(int),
            "tempo_relacionamento_meses": tempo.round().astype(int),
            "tipo_ocupacao": ocupacao,
            "escolaridade": escolaridade,
            "possui_imovel_proprio": imovel.astype(int),
            "fluxo_medio_entradas_3m": fluxo.round(2),
            "saldo_medio_conta_12m": saldo12.round(2),
            "saldo_medio_conta_3m": saldo3.round(2),
            "limite_cheque_especial": limite_cheque.round(2),
            "dias_utilizando_cheque_especial_30d": dias30.astype(int),
            "dias_utilizando_cheque_especial_12m": dias12.astype(int),
            "utilizacao_cheque_especial_atual": np.round(uso, 4),
            "qtd_cheques_devolvidos_12m": cheques.astype(int),
            "limite_total_cartao": limite_cartao.round(2),
            "fatura_media_3m": fatura3.round(2),
            "fatura_media_12m": fatura12.round(2),
            "taxa_crescimento_gastos_cartao": np.round(crescimento, 4),
            "pagamento_fatura_predominante": pagamento,
            "qtd_vezes_pagamento_minimo_12m": qtd_min.astype(int),
            "qtd_vezes_atraso_cartao_30d_12m": atraso_cartao.astype(int),
            "valor_saque_cartao_credito_6m": saque.round(2),
            "flag_pagamento_cartao_com_cheque_especial": flag_cheque.astype(int),
            "qtd_emprestimos_ativos": ativos.astype(int),
            "qtd_novos_emprestimos_ultimos_6m": novos.astype(int),
            "parcela_total_comprometida": parcela.round(2),
            "comprometimento_fluxo_entradas": np.round(comprometimento, 4),
            "max_atraso_emprestimos_historico": atraso_emp.astype(int),
            "taxa_juros_media_contratos_ativos": np.round(juros, 2),
            "qtd_ocorrencias_atraso_30_60d_12m": ocorrencias.astype(int),
            "flag_historico_fpd": fpd.astype(int),
            "possui_renegociacao_ativa": reneg.astype(int),
            "qtd_protestos_historico": protestos.astype(int),
            "valor_total_inadimplencia_outros_mercados": inad_outros.round(2),
        }
    )
    drivers = _drivers(df)
    if parametros is None:
        parametros = _parametros(drivers, rng)
    df["inadimplente_90d"] = _rotular(drivers, parametros, rng)
    largura = 6 if prefixo_id == "C" else 8
    df.insert(0, "id_cliente", [f"{prefixo_id}{i + 1:0{largura}d}" for i in range(n)])
    df = df[COLUNAS]
    conferir_esquema(df)
    return df, parametros


def gerar_cliente(seed: int, parametros: dict) -> pd.Series:
    df, _ = gerar_base(1, seed, parametros=parametros, prefixo_id="N")
    return df.iloc[0]


def perfil_controles(df: pd.DataFrame) -> dict:
    """Mediana, moda e amplitude medidas, para o simulador abrir num cliente típico."""
    perfil = {}
    for coluna in FEATURES:
        if coluna in CATEGORICAS:
            moda = str(df[coluna].mode(dropna=False).iloc[0])
            perfil[coluna] = {"tipo": "categoria", "moda": moda, "opcoes": sorted(df[coluna].astype(str).unique())}
        else:
            serie = df[coluna]
            perfil[coluna] = {
                "tipo": "numero",
                "mediana": float(serie.median()),
                "min": float(serie.min()),
                "max": float(serie.max()),
            }
    return perfil


def salvar_parametros(path: Path, parametros: dict) -> None:
    import json

    path.write_text(json.dumps(parametros), encoding="utf-8")


def ler_parametros(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
