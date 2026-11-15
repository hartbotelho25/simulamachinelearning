"""Gatilhos preventivos do simulador. Tudo é Sim ou Não.

Uma reincidência ligada exige o gatilho base. O vetor do modelo
respeita essa hierarquia mesmo que a tela chegue inconsistente.
"""

from __future__ import annotations

GATILHOS = [
    {
        "id": "usou_cheque_especial_6m",
        "grupo_id": "cheque",
        "grupo": "Cheque especial e cheques",
        "texto": "Usou cheque especial nos últimos 6 meses?",
        "rotulo": "Usou cheque especial, 6 meses",
        "pai": None,
    },
    {
        "id": "uso_cheque_especial_mais_3m_em_6m",
        "grupo_id": "cheque",
        "grupo": "Cheque especial e cheques",
        "texto": "Uso do cheque especial por mais de 3 meses nos últimos 6 meses?",
        "rotulo": "Especial por mais de 3 meses",
        "pai": "usou_cheque_especial_6m",
    },
    {
        "id": "cheque_sem_fundo_12m",
        "grupo_id": "cheque",
        "grupo": "Cheque especial e cheques",
        "texto": "Cheque sem fundo devolvido nos últimos 12 meses?",
        "rotulo": "Cheque sem fundo, 12 meses",
        "pai": None,
    },
    {
        "id": "mais_de_um_cheque_sem_fundo_12m",
        "grupo_id": "cheque",
        "grupo": "Cheque especial e cheques",
        "texto": "Mais de 1 cheque sem fundo nos últimos 12 meses?",
        "rotulo": "Mais de 1 cheque sem fundo",
        "pai": "cheque_sem_fundo_12m",
    },
    {
        "id": "cartao_acima_90_ou_compra_negada",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Cartão acima de 90% do limite ou compra negada por teto?",
        "rotulo": "Cartão acima de 90% ou compra negada",
        "pai": None,
    },
    {
        "id": "cartao_acima_90_mais_3x_6m",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Uso acima de 90% do cartão mais de 3 vezes nos últimos 6 meses?",
        "rotulo": "Cartão acima de 90% mais de 3 vezes",
        "pai": "cartao_acima_90_ou_compra_negada",
    },
    {
        "id": "pagamento_minimo_rotativo",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Pagamento mínimo ou rotativo da fatura?",
        "rotulo": "Pagamento mínimo ou rotativo",
        "pai": None,
    },
    {
        "id": "rotativo_mais_2x_6m",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Pagamento mínimo ou rotativo mais de 2 vezes nos últimos 6 meses?",
        "rotulo": "Rotativo mais de 2 vezes",
        "pai": "pagamento_minimo_rotativo",
    },
    {
        "id": "parcelamento_fatura_6m",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Parcelamento da fatura do cartão nos últimos 6 meses?",
        "rotulo": "Parcelou a fatura do cartão",
        "pai": None,
    },
    {
        "id": "pagou_fatura_com_cheque_especial",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Pagou a fatura do cartão usando o cheque especial?",
        "rotulo": "Pagou a fatura com o especial",
        "pai": None,
    },
    {
        "id": "saque_cartao_credito",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Saque em dinheiro no cartão de crédito?",
        "rotulo": "Saque no cartão de crédito",
        "pai": None,
    },
    {
        "id": "salto_gastos_cartao",
        "grupo_id": "cartao",
        "grupo": "Cartão de crédito e fatura",
        "texto": "Salto repentino de gastos no cartão frente ao padrão habitual?",
        "rotulo": "Salto de gastos no cartão",
        "pai": None,
    },
    {
        "id": "novo_emprestimo_6m",
        "grupo_id": "emprestimo",
        "grupo": "Empréstimos, emergência e reserva",
        "texto": "Contratou novo empréstimo nos últimos 6 meses?",
        "rotulo": "Novo empréstimo, 6 meses",
        "pai": None,
    },
    {
        "id": "acumulo_emprestimos",
        "grupo_id": "emprestimo",
        "grupo": "Empréstimos, emergência e reserva",
        "texto": "Acúmulo de empréstimos ativos, mais de 2 contratos ou taxa elevada?",
        "rotulo": "Acúmulo de empréstimos",
        "pai": None,
    },
    {
        "id": "adiantamento_emergencial",
        "grupo_id": "emprestimo",
        "grupo": "Empréstimos, emergência e reserva",
        "texto": "Adiantamento emergencial ou crédito caro?",
        "rotulo": "Adiantamento emergencial",
        "pai": None,
    },
    {
        "id": "renegociacao_ativa",
        "grupo_id": "emprestimo",
        "grupo": "Empréstimos, emergência e reserva",
        "texto": "Renegociação ou refinanciamento de dívida ativo?",
        "rotulo": "Renegociação ativa",
        "pai": None,
    },
    {
        "id": "resgate_total_reserva",
        "grupo_id": "emprestimo",
        "grupo": "Empréstimos, emergência e reserva",
        "texto": "Resgate total da reserva, zerou poupança ou investimentos no banco?",
        "rotulo": "Resgate total da reserva",
        "pai": None,
    },
    {
        "id": "protesto_cartorio",
        "grupo_id": "mercado",
        "grupo": "Cartório e mercado",
        "texto": "Protesto em cartório?",
        "rotulo": "Protesto em cartório",
        "pai": None,
    },
    {
        "id": "protesto_3_ou_mais",
        "grupo_id": "mercado",
        "grupo": "Cartório e mercado",
        "texto": "Protesto +3 em cartório, 3 ou mais protestos?",
        "rotulo": "3 ou mais protestos",
        "pai": "protesto_cartorio",
    },
    {
        "id": "restricao_outras_instituicoes",
        "grupo_id": "mercado",
        "grupo": "Cartório e mercado",
        "texto": "Restrição ativa em outras instituições financeiras?",
        "rotulo": "Restrição em outra instituição",
        "pai": None,
    },
    {
        "id": "queda_vinculo_principal",
        "grupo_id": "vinculo",
        "grupo": "Vínculo e perfil",
        "texto": "Queda de vínculo principal, cancelou salário ou parou de usar a conta?",
        "rotulo": "Queda do vínculo principal",
        "pai": None,
    },
    {
        "id": "relacionamento_recente",
        "grupo_id": "vinculo",
        "grupo": "Vínculo e perfil",
        "texto": "Relacionamento recente, conta com menos de 1 ano?",
        "rotulo": "Relacionamento recente",
        "pai": None,
    },
    {
        "id": "sem_imovel_proprio",
        "grupo_id": "vinculo",
        "grupo": "Vínculo e perfil",
        "texto": "Sem imóvel próprio?",
        "rotulo": "Sem imóvel próprio",
        "pai": None,
    },
]

IDS = [item["id"] for item in GATILHOS]
POR_ID = {item["id"]: item for item in GATILHOS}
FILHOS: dict[str, list[str]] = {}
for _item in GATILHOS:
    if _item["pai"]:
        FILHOS.setdefault(_item["pai"], []).append(_item["id"])

PARES = [(item["id"], item["pai"]) for item in GATILHOS if item["pai"]]

# Pontuação interna do comitê. A tela não mostra esses números.
# No par, a reincidência substitui a ocorrência única.
PONTOS = {
    "restricao_outras_instituicoes": 35,
    "protesto_3_ou_mais": 30,
    "protesto_cartorio": 18,
    "mais_de_um_cheque_sem_fundo_12m": 25,
    "cheque_sem_fundo_12m": 15,
    "uso_cheque_especial_mais_3m_em_6m": 12,
    "usou_cheque_especial_6m": 5,
    "saque_cartao_credito": 20,
    "rotativo_mais_2x_6m": 18,
    "pagou_fatura_com_cheque_especial": 18,
    "parcelamento_fatura_6m": 15,
    "cartao_acima_90_mais_3x_6m": 12,
    "salto_gastos_cartao": 10,
    "pagamento_minimo_rotativo": 8,
    "cartao_acima_90_ou_compra_negada": 8,
    "renegociacao_ativa": 22,
    "resgate_total_reserva": 18,
    "adiantamento_emergencial": 12,
    "acumulo_emprestimos": 10,
    "novo_emprestimo_6m": 5,
    "queda_vinculo_principal": 15,
    "relacionamento_recente": 4,
    "sem_imovel_proprio": 2,
}

PRESETS = [
    {"id": "limpar", "texto": "Limpar tudo (tudo Não)", "ligar": []},
    {
        "id": "aperto",
        "texto": "Aperto no cartão e no especial",
        "ligar": [
            "uso_cheque_especial_mais_3m_em_6m",
            "rotativo_mais_2x_6m",
            "cartao_acima_90_mais_3x_6m",
            "parcelamento_fatura_6m",
        ],
    },
    {
        "id": "grave",
        "texto": "Reincidência grave",
        "ligar": [
            "mais_de_um_cheque_sem_fundo_12m",
            "protesto_3_ou_mais",
            "renegociacao_ativa",
            "resgate_total_reserva",
            "queda_vinculo_principal",
            "pagou_fatura_com_cheque_especial",
            "adiantamento_emergencial",
        ],
    },
]


def respostas_vazias() -> dict[str, bool]:
    return {item["id"]: False for item in GATILHOS}


def coerente(respostas: dict[str, bool]) -> dict[str, bool]:
    saida = {item["id"]: bool(respostas.get(item["id"], False)) for item in GATILHOS}
    for filho, pai in PARES:
        if saida[filho]:
            saida[pai] = True
    return saida


def vetor(respostas: dict[str, bool]) -> dict[str, int]:
    limpo = coerente(respostas)
    return {item["id"]: int(limpo[item["id"]]) for item in GATILHOS}


def codificar(linha: dict) -> dict[str, int]:
    """Entrada do modelo: se a reincidência está ligada, a ocorrência única entra como zero."""
    saida = {item["id"]: int(linha.get(item["id"], 0)) for item in GATILHOS}
    for filho, pai in PARES:
        if saida[filho]:
            saida[pai] = 0
    return saida


def pontuar(linha: dict) -> int:
    exclusivo = codificar(linha)
    return int(sum(PONTOS[chave] * exclusivo[chave] for chave in exclusivo))


def principais(linha: dict, limite: int = 3) -> list[str]:
    exclusivo = codificar(linha)
    ativos = [item for item in GATILHOS if exclusivo[item["id"]]]
    ativos.sort(key=lambda item: PONTOS[item["id"]], reverse=True)
    return [item["rotulo"] for item in ativos[:limite]]


def grupos() -> list[tuple[str, list[dict]]]:
    ordem: list[str] = []
    mapa: dict[str, list[dict]] = {}
    for item in GATILHOS:
        if item["grupo"] not in mapa:
            ordem.append(item["grupo"])
            mapa[item["grupo"]] = []
        mapa[item["grupo"]].append(item)
    return [(nome, mapa[nome]) for nome in ordem]


def montar_chaves() -> dict:
    return {
        "gatilhos": [
            {
                "id": item["id"],
                "grupo": item["grupo"],
                "texto": item["texto"],
                "rotulo": item["rotulo"],
                "pai": item["pai"],
            }
            for item in GATILHOS
        ],
        "presets": PRESETS,
    }
