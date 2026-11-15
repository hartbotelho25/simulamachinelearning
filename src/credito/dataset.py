"""Base sintética de gatilhos preventivos.

Dois alvos: ficar inadimplente e passar de 90 dias.
Quem passa de 90 dias já está inadimplente. Não há renda, saldo,
fluxo, score de bureau nem atraso que já se instalou.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chaves import GATILHOS, IDS, PARES, PONTOS
from .politica import N_CLIENTES

ALVOS = ("inadimplente", "inadimplente_90d")

COLUNAS = ["id_cliente", *IDS, *ALVOS]

PROIBIDAS = {
    "renda_declarada",
    "fluxo_medio_entradas_3m",
    "saldo_medio_conta_12m",
    "saldo_medio_conta_3m",
    "comprometimento_fluxo_entradas",
    "score_bureau_atual",
    "qtd_consultas_bureau_3m",
    "qtd_ocorrencias_atraso_30_60d_12m",
    "qtd_vezes_atraso_cartao_30d_12m",
    "max_atraso_emprestimos_historico",
    "flag_historico_fpd",
    "pix_boleto_parcelado",
    "debito_automatico_devolvido",
    "qtd_consultas_mercado",
}

BLOCOS: dict[str, list[str]] = {}
BLOCO_NOME: dict[str, str] = {}
ROTULOS: dict[str, str] = {}
for _item in GATILHOS:
    BLOCOS.setdefault(_item["grupo_id"], []).append(_item["id"])
    BLOCO_NOME[_item["grupo_id"]] = _item["grupo"]
    ROTULOS[_item["id"]] = _item["rotulo"]

FEATURES = list(IDS)
CATEGORICAS: list[str] = []
NUMERICAS = list(FEATURES)

# A curva cruza 50% na passagem do amarelo para o vermelho.
CENTRO_INAD = 25.5
INCLINACAO_INAD = 0.152
CENTRO_90 = 32.0
INCLINACAO_90 = 0.168
RUIDO = 0.15

# Casos em que o gatilho grave aparece sozinho, com a casa limpa.
ISOLADOS = (
    ("restricao_outras_instituicoes", None),
    ("protesto_3_ou_mais", "protesto_cartorio"),
    ("protesto_cartorio", None),
    ("mais_de_um_cheque_sem_fundo_12m", "cheque_sem_fundo_12m"),
    ("cheque_sem_fundo_12m", None),
    ("renegociacao_ativa", None),
    ("saque_cartao_credito", None),
    ("pagou_fatura_com_cheque_especial", None),
    ("resgate_total_reserva", None),
    ("queda_vinculo_principal", None),
    ("rotativo_mais_2x_6m", "pagamento_minimo_rotativo"),
    ("uso_cheque_especial_mais_3m_em_6m", "usou_cheque_especial_6m"),
)


def conferir_esquema(df: pd.DataFrame, checar_taxas: bool = False) -> None:
    if list(df.columns) != COLUNAS:
        raise ValueError("A ordem das colunas não fecha com os gatilhos.")
    if PROIBIDAS & set(df.columns):
        raise ValueError("Entrou coluna que o estudo deixou de fora.")
    if df.isna().any().any():
        raise ValueError("A base sintética saiu com vazio.")
    for coluna in IDS:
        valores = set(df[coluna].unique())
        if not valores <= {0, 1}:
            raise ValueError(f"{coluna} não é Sim/Não.")
    for filho, pai in PARES:
        if ((df[filho] == 1) & (df[pai] == 0)).any():
            raise ValueError(f"{filho} ficou ligado sem {pai}.")
    if (df["inadimplente_90d"] > df["inadimplente"]).any():
        raise ValueError("Acima de 90 dias ficou fora da inadimplência.")
    if not checar_taxas:
        return
    taxa = float(df["inadimplente"].mean())
    taxa_90 = float(df["inadimplente_90d"].mean())
    if not 0.11 <= taxa <= 0.15:
        raise ValueError(f"Inadimplência saiu em {taxa:.3%}, fora de 11% a 15%.")
    if not 0.05 <= taxa_90 <= 0.08:
        raise ValueError(f"Acima de 90 dias saiu em {taxa_90:.3%}, fora de 5% a 8%.")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def _flag(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.random(logit.shape[0]) < _sigmoid(logit)).astype(int)


def aplicar_exclusivo(df: pd.DataFrame) -> pd.DataFrame:
    """Zera a ocorrência única quando a reincidência do mesmo par está ligada."""
    saida = df.copy()
    for filho, pai in PARES:
        saida.loc[saida[filho].to_numpy() == 1, pai] = 0
    return saida


def _marcador(sinais: pd.DataFrame) -> np.ndarray:
    exclusivo = aplicar_exclusivo(sinais[FEATURES])
    bruto = np.zeros(len(exclusivo), dtype=float)
    for coluna in FEATURES:
        bruto += exclusivo[coluna].to_numpy(dtype=float) * PONTOS[coluna]
    ligados = exclusivo.to_numpy(dtype=float).sum(axis=1)
    bruto += np.where(ligados >= 3, 2.0, np.where(ligados >= 2, 1.0, 0.0))
    return bruto


def _isolar(rng: np.random.Generator, sinais: pd.DataFrame) -> None:
    n = len(sinais)
    cota = max(30, n // 140)
    bloco = cota * len(ISOLADOS)
    if bloco >= n:
        cota = max(1, n // (len(ISOLADOS) + 1))
        bloco = cota * len(ISOLADOS)
    escolhidos = rng.choice(n, size=bloco, replace=False)
    for i, (grave, pai) in enumerate(ISOLADOS):
        linhas = escolhidos[i * cota : (i + 1) * cota]
        sinais.loc[linhas, FEATURES] = 0
        sinais.loc[linhas, grave] = 1
        if pai:
            sinais.loc[linhas, pai] = 1


def _sinais(rng: np.random.Generator, n: int) -> pd.DataFrame:
    stress = rng.normal(0.0, 1.0, n)

    def par(logit_leve: float, logit_grave: float) -> tuple[np.ndarray, np.ndarray]:
        grave = _flag(rng, logit_grave + 0.4 * stress)
        leve = _flag(rng, logit_leve + 0.35 * stress)
        return np.maximum(leve, grave), grave

    usou, mais_3m = par(-3.15, -4.15)
    cheque, mais_um = par(-3.85, -4.65)
    teto, teto_3x = par(-3.45, -4.25)
    minimo, rotativo = par(-3.35, -4.15)
    parcela = _flag(rng, -3.75 + 0.45 * stress)
    pagou = _flag(rng, -4.15 + 0.5 * stress)
    saque = _flag(rng, -4.35 + 0.5 * stress)
    salto = _flag(rng, -3.55 + 0.4 * stress)
    novo = _flag(rng, -3.35 + 0.35 * stress)
    acumulo = _flag(rng, -3.85 + 0.45 * stress)
    adiant = _flag(rng, -4.25 + 0.45 * stress)
    reneg = _flag(rng, -4.45 + 0.5 * stress)
    resgate = _flag(rng, -4.55 + 0.5 * stress)
    protesto, protesto3 = par(-4.05, -4.75)
    restricao = _flag(rng, -4.65 + 0.45 * stress)
    queda = _flag(rng, -4.25 + 0.45 * stress)
    recente = _flag(rng, -2.3 + 0.15 * stress)
    sem_imovel = _flag(rng, -0.85 + 0.1 * stress)
    sinais = pd.DataFrame(
        {
            "usou_cheque_especial_6m": usou,
            "uso_cheque_especial_mais_3m_em_6m": mais_3m,
            "cheque_sem_fundo_12m": cheque,
            "mais_de_um_cheque_sem_fundo_12m": mais_um,
            "cartao_acima_90_ou_compra_negada": teto,
            "cartao_acima_90_mais_3x_6m": teto_3x,
            "pagamento_minimo_rotativo": minimo,
            "rotativo_mais_2x_6m": rotativo,
            "parcelamento_fatura_6m": parcela,
            "pagou_fatura_com_cheque_especial": pagou,
            "saque_cartao_credito": saque,
            "salto_gastos_cartao": salto,
            "novo_emprestimo_6m": novo,
            "acumulo_emprestimos": acumulo,
            "adiantamento_emergencial": adiant,
            "renegociacao_ativa": reneg,
            "resgate_total_reserva": resgate,
            "protesto_cartorio": protesto,
            "protesto_3_ou_mais": protesto3,
            "restricao_outras_instituicoes": restricao,
            "queda_vinculo_principal": queda,
            "relacionamento_recente": recente,
            "sem_imovel_proprio": sem_imovel,
        }
    )
    _isolar(rng, sinais)
    _plantar_pares(rng, sinais)
    return sinais


def _plantar_pares(rng: np.random.Generator, sinais: pd.DataFrame) -> None:
    """Garante as combinações críticas com amostra no teste, sem dominar a carteira."""
    pares = (
        ("uso_cheque_especial_mais_3m_em_6m", "usou_cheque_especial_6m", "rotativo_mais_2x_6m", "pagamento_minimo_rotativo"),
        ("mais_de_um_cheque_sem_fundo_12m", "cheque_sem_fundo_12m", "pagou_fatura_com_cheque_especial"),
        ("protesto_3_ou_mais", "protesto_cartorio", "renegociacao_ativa"),
        ("resgate_total_reserva", "queda_vinculo_principal"),
        ("cartao_acima_90_mais_3x_6m", "cartao_acima_90_ou_compra_negada", "parcelamento_fatura_6m"),
        ("adiantamento_emergencial", "novo_emprestimo_6m"),
    )
    n = len(sinais)
    cota = max(8, n // 620)
    livres = np.arange(n)
    rng.shuffle(livres)
    for i, grupo in enumerate(pares):
        linhas = livres[i * cota : (i + 1) * cota]
        sinais.loc[linhas, FEATURES] = 0
        for coluna in grupo:
            sinais.loc[linhas, coluna] = 1


def gerar_base(n: int, seed: int, parametros: dict | None = None, prefixo_id: str = "C") -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    sinais = _sinais(rng, n)
    marcador = _marcador(sinais)
    ruido = float(parametros["ruido"]) if parametros else RUIDO
    ruido_inad = rng.normal(0.0, ruido, n)
    ruido_90 = rng.normal(0.0, ruido, n)
    p_inad = _sigmoid((marcador - CENTRO_INAD) * INCLINACAO_INAD + ruido_inad)
    p_grave = _sigmoid((marcador - CENTRO_90) * INCLINACAO_90 + ruido_90)
    sorteio = rng.random(n)
    inad = (sorteio < p_inad).astype(int)
    acima = (sorteio < np.minimum(p_grave, p_inad)).astype(int)
    if parametros is None:
        parametros = {
            "ruido": RUIDO,
            "centro_inad": CENTRO_INAD,
            "inclinacao_inad": INCLINACAO_INAD,
            "centro_90": CENTRO_90,
            "inclinacao_90": INCLINACAO_90,
        }
    df = sinais.copy()
    df["inadimplente"] = inad
    df["inadimplente_90d"] = acima
    largura = 6 if prefixo_id == "C" else 8
    df.insert(0, "id_cliente", [f"{prefixo_id}{i + 1:0{largura}d}" for i in range(n)])
    df = df[COLUNAS]
    conferir_esquema(df, checar_taxas=n >= N_CLIENTES)
    return df, parametros


def salvar_parametros(caminho: Path, parametros: dict) -> None:
    caminho.write_text(json.dumps(parametros, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ler_parametros(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))
