"""Perguntas Sim/Não do simulador e o vetor que cada uma liga nos modelos."""

from __future__ import annotations

import pandas as pd

from .dataset import CATEGORICAS, FEATURES

INTEIROS = {
    "idade",
    "tempo_relacionamento_meses",
    "possui_imovel_proprio",
    "dias_utilizando_cheque_especial_30d",
    "dias_utilizando_cheque_especial_12m",
    "qtd_cheques_devolvidos_12m",
    "qtd_vezes_pagamento_minimo_12m",
    "qtd_vezes_atraso_cartao_30d_12m",
    "flag_pagamento_cartao_com_cheque_especial",
    "qtd_emprestimos_ativos",
    "qtd_novos_emprestimos_ultimos_6m",
    "max_atraso_emprestimos_historico",
    "qtd_ocorrencias_atraso_30_60d_12m",
    "flag_historico_fpd",
    "possui_renegociacao_ativa",
    "qtd_protestos_historico",
}

ZERAR = {
    "dias_utilizando_cheque_especial_30d",
    "dias_utilizando_cheque_especial_12m",
    "qtd_cheques_devolvidos_12m",
    "qtd_vezes_pagamento_minimo_12m",
    "qtd_vezes_atraso_cartao_30d_12m",
    "valor_saque_cartao_credito_6m",
    "flag_pagamento_cartao_com_cheque_especial",
    "qtd_novos_emprestimos_ultimos_6m",
    "max_atraso_emprestimos_historico",
    "qtd_ocorrencias_atraso_30_60d_12m",
    "flag_historico_fpd",
    "possui_renegociacao_ativa",
    "qtd_protestos_historico",
    "valor_total_inadimplencia_outros_mercados",
}


def _numero(coluna: str, valor) -> int | float | str:
    if coluna in CATEGORICAS:
        return str(valor)
    if coluna in INTEIROS:
        return int(round(float(valor)))
    return round(float(valor), 4)


def _mediana(df: pd.DataFrame, mascara: pd.Series, coluna: str):
    serie = df.loc[mascara, coluna]
    if serie.empty:
        serie = df[coluna]
    if coluna in CATEGORICAS:
        return str(serie.mode(dropna=False).iloc[0])
    return _numero(coluna, serie.median())


def perfil_em_dia(df: pd.DataFrame) -> dict:
    linha = {}
    for coluna in FEATURES:
        if coluna in CATEGORICAS:
            linha[coluna] = str(df[coluna].mode(dropna=False).iloc[0])
        else:
            linha[coluna] = _numero(coluna, df[coluna].median())
    for coluna in ZERAR:
        linha[coluna] = 0
    linha["pagamento_fatura_predominante"] = "integral"
    linha["possui_imovel_proprio"] = 1
    linha["tempo_relacionamento_meses"] = max(int(linha["tempo_relacionamento_meses"]), 24)
    linha["comprometimento_fluxo_entradas"] = round(min(float(linha["comprometimento_fluxo_entradas"]), 0.12), 4)
    linha["utilizacao_cheque_especial_atual"] = round(min(float(linha["utilizacao_cheque_especial_atual"]), 0.15), 4)
    linha["taxa_crescimento_gastos_cartao"] = round(min(float(linha["taxa_crescimento_gastos_cartao"]), 0.0), 4)
    return linha


def _pacote(df: pd.DataFrame, mascara: pd.Series, colunas: list[str]) -> dict:
    return {coluna: _mediana(df, mascara, coluna) for coluna in colunas}


def patamares_sim(df: pd.DataFrame) -> dict[str, dict]:
    atraso = df["qtd_ocorrencias_atraso_30_60d_12m"] >= 1
    fpd = df["flag_historico_fpd"] == 1
    reneg = df["possui_renegociacao_ativa"] == 1
    protesto = df["qtd_protestos_historico"] >= 1
    restricao = df["valor_total_inadimplencia_outros_mercados"] > 0
    cheque = df["qtd_cheques_devolvidos_12m"] >= 1
    continuo = (df["dias_utilizando_cheque_especial_12m"] >= 60) | (df["utilizacao_cheque_especial_atual"] >= 0.5)
    caixa = (df["saldo_medio_conta_3m"] < 0) | (
        df["saldo_medio_conta_3m"] < 0.15 * df["fluxo_medio_entradas_3m"]
    )
    minimo = (df["pagamento_fatura_predominante"] == "minimo") | (df["qtd_vezes_pagamento_minimo_12m"] >= 3)
    pagou = df["flag_pagamento_cartao_com_cheque_especial"] == 1
    cartao_ruim = (df["qtd_vezes_atraso_cartao_30d_12m"] >= 1) | (df["valor_saque_cartao_credito_6m"] > 0)
    parcelas = df["comprometimento_fluxo_entradas"] >= 0.40
    contrato = (df["qtd_novos_emprestimos_ultimos_6m"] >= 1) | (df["max_atraso_emprestimos_historico"] >= 30)
    recente = df["tempo_relacionamento_meses"] < 12
    return {
        "atraso_30_60": _pacote(df, atraso, ["qtd_ocorrencias_atraso_30_60d_12m"]),
        "fpd": {"flag_historico_fpd": 1},
        "renegociacao": {"possui_renegociacao_ativa": 1},
        "protesto": _pacote(df, protesto, ["qtd_protestos_historico"]),
        "restricao": _pacote(df, restricao, ["valor_total_inadimplencia_outros_mercados"]),
        "cheque_devolvido": _pacote(df, cheque, ["qtd_cheques_devolvidos_12m"]),
        "cheque_continuo": _pacote(
            df,
            continuo,
            [
                "dias_utilizando_cheque_especial_30d",
                "dias_utilizando_cheque_especial_12m",
                "utilizacao_cheque_especial_atual",
            ],
        ),
        "caixa_fraco": _pacote(
            df,
            caixa,
            ["fluxo_medio_entradas_3m", "saldo_medio_conta_12m", "saldo_medio_conta_3m"],
        ),
        "fatura_minimo": {
            "pagamento_fatura_predominante": "minimo",
            "qtd_vezes_pagamento_minimo_12m": _mediana(df, minimo, "qtd_vezes_pagamento_minimo_12m"),
        },
        "pagou_com_especial": {"flag_pagamento_cartao_com_cheque_especial": 1},
        "atraso_ou_saque": _pacote(
            df,
            cartao_ruim,
            ["qtd_vezes_atraso_cartao_30d_12m", "valor_saque_cartao_credito_6m"],
        ),
        "parcelas_altas": _pacote(df, parcelas, ["comprometimento_fluxo_entradas", "parcela_total_comprometida"]),
        "contrato_novo_ou_atraso": _pacote(
            df,
            contrato,
            ["qtd_novos_emprestimos_ultimos_6m", "max_atraso_emprestimos_historico"],
        ),
        "cliente_recente": _pacote(df, recente, ["tempo_relacionamento_meses"]),
    }


PERGUNTAS = [
    {"id": "atraso_30_60", "grupo": "Atrasos internos", "texto": "Já atrasou entre 30 e 60 dias?"},
    {"id": "fpd", "grupo": "Atrasos internos", "texto": "Atrasou logo na 1ª parcela?"},
    {"id": "renegociacao", "grupo": "Atrasos internos", "texto": "Possui renegociação ativa?"},
    {"id": "protesto", "grupo": "Restrições no mercado", "texto": "Possui protesto em cartório?"},
    {"id": "restricao", "grupo": "Restrições no mercado", "texto": "Possui restrição em outra instituição?"},
    {"id": "cheque_devolvido", "grupo": "Cheque especial e conta", "texto": "Teve cheque devolvido?"},
    {"id": "cheque_continuo", "grupo": "Cheque especial e conta", "texto": "Usa o cheque especial de forma contínua?"},
    {"id": "caixa_fraco", "grupo": "Cheque especial e conta", "texto": "Fluxo caiu ou o saldo ficou baixo?"},
    {"id": "fatura_minimo", "grupo": "Cartão de crédito", "texto": "Paga a fatura no mínimo com frequência?"},
    {"id": "pagou_com_especial", "grupo": "Cartão de crédito", "texto": "Pagou a fatura com o cheque especial?"},
    {"id": "atraso_ou_saque", "grupo": "Cartão de crédito", "texto": "Atrasou a fatura ou fez saque no crédito?"},
    {"id": "parcelas_altas", "grupo": "Empréstimos", "texto": "As parcelas consomem fatia alta das entradas?"},
    {"id": "contrato_novo_ou_atraso", "grupo": "Empréstimos", "texto": "Contratou empréstimo recente ou já atrasou contrato?"},
    {"id": "cliente_recente", "grupo": "Relacionamento", "texto": "Cliente recente no banco?"},
    {"id": "imovel", "grupo": "Relacionamento", "texto": "Possui imóvel próprio?"},
]


def respostas_em_dia() -> dict[str, bool]:
    return {pergunta["id"]: pergunta["id"] == "imovel" for pergunta in PERGUNTAS}


def respostas_da_linha(row: pd.Series) -> dict[str, bool]:
    atraso = int(row["qtd_ocorrencias_atraso_30_60d_12m"]) >= 1
    caixa = float(row["saldo_medio_conta_3m"]) < 0 or float(row["saldo_medio_conta_3m"]) < 0.15 * float(
        row["fluxo_medio_entradas_3m"]
    )
    return {
        "atraso_30_60": atraso,
        "fpd": int(row["flag_historico_fpd"]) == 1,
        "renegociacao": int(row["possui_renegociacao_ativa"]) == 1,
        "protesto": int(row["qtd_protestos_historico"]) >= 1,
        "restricao": float(row["valor_total_inadimplencia_outros_mercados"]) > 0,
        "cheque_devolvido": int(row["qtd_cheques_devolvidos_12m"]) >= 1,
        "cheque_continuo": int(row["dias_utilizando_cheque_especial_12m"]) >= 60
        or float(row["utilizacao_cheque_especial_atual"]) >= 0.5,
        "caixa_fraco": caixa,
        "fatura_minimo": str(row["pagamento_fatura_predominante"]) == "minimo"
        or int(row["qtd_vezes_pagamento_minimo_12m"]) >= 3,
        "pagou_com_especial": int(row["flag_pagamento_cartao_com_cheque_especial"]) == 1,
        "atraso_ou_saque": int(row["qtd_vezes_atraso_cartao_30d_12m"]) >= 1
        or float(row["valor_saque_cartao_credito_6m"]) > 0,
        "parcelas_altas": float(row["comprometimento_fluxo_entradas"]) >= 0.40,
        "contrato_novo_ou_atraso": int(row["qtd_novos_emprestimos_ultimos_6m"]) >= 1
        or int(row["max_atraso_emprestimos_historico"]) >= 30,
        "cliente_recente": int(row["tempo_relacionamento_meses"]) < 12,
        "imovel": int(row["possui_imovel_proprio"]) == 1,
    }


def aplicar(em_dia: dict, sims: dict, respostas: dict[str, bool]) -> dict:
    linha = dict(em_dia)
    for pergunta in PERGUNTAS:
        ident = pergunta["id"]
        if ident == "imovel":
            linha["possui_imovel_proprio"] = 1 if respostas.get(ident) else 0
            continue
        if respostas.get(ident):
            linha.update(sims[ident])
    return linha


def montar_chaves(df: pd.DataFrame) -> dict:
    return {
        "em_dia": perfil_em_dia(df),
        "sim": patamares_sim(df),
        "perguntas": [{"id": p["id"], "grupo": p["grupo"], "texto": p["texto"]} for p in PERGUNTAS],
    }
