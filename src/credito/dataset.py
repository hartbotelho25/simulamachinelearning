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

from .chaves import GATILHOS, IDS, PARES
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

# Carga latente do processo. Não é pontuação e não entra na tela.
PESO_LATENTE = {
    "usou_cheque_especial_6m": 0.85,
    "uso_cheque_especial_mais_3m_em_6m": 0.75,
    "cheque_sem_fundo_12m": 1.15,
    "mais_de_um_cheque_sem_fundo_12m": 2.40,
    "cartao_acima_90_ou_compra_negada": 0.80,
    "cartao_acima_90_mais_3x_6m": 0.70,
    "pagamento_minimo_rotativo": 0.90,
    "rotativo_mais_2x_6m": 0.75,
    "parcelamento_fatura_6m": 1.05,
    "pagou_fatura_com_cheque_especial": 3.40,
    "saque_cartao_credito": 1.35,
    "salto_gastos_cartao": 0.95,
    "novo_emprestimo_6m": 0.80,
    "acumulo_emprestimos": 1.15,
    "adiantamento_emergencial": 1.25,
    "renegociacao_ativa": 3.50,
    "resgate_total_reserva": 1.70,
    "protesto_cartorio": 1.20,
    "protesto_3_ou_mais": 2.60,
    "restricao_outras_instituicoes": 3.80,
    "queda_vinculo_principal": 1.45,
    "relacionamento_recente": 0.45,
    "sem_imovel_proprio": 0.28,
}
BASE_INAD = -3.84
INCLINACAO = 1.619
BASE_90 = -4.70
INCLINACAO_90 = 1.62
RUIDO = 0.18

# Perfis em que um sinal forte aparece com o resto da casa limpa.
ISOLADOS = (
    ("restricao_outras_instituicoes",),
    ("protesto_3_ou_mais", "protesto_cartorio"),
    ("mais_de_um_cheque_sem_fundo_12m", "cheque_sem_fundo_12m"),
    ("pagou_fatura_com_cheque_especial",),
    ("renegociacao_ativa",),
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


def _escore(sinais: pd.DataFrame) -> np.ndarray:
    pesos = np.array([PESO_LATENTE[coluna] for coluna in FEATURES], dtype=float)
    return sinais[FEATURES].to_numpy(dtype=float) @ pesos


def _plantar(rng: np.random.Generator, sinais: pd.DataFrame, grupos: tuple[tuple[str, ...], ...], cota: int) -> None:
    n = len(sinais)
    if cota <= 0 or cota * len(grupos) >= n:
        return
    livres = np.arange(n)
    rng.shuffle(livres)
    for i, grupo in enumerate(grupos):
        linhas = livres[i * cota : (i + 1) * cota]
        sinais.loc[linhas, FEATURES] = 0
        for coluna in grupo:
            sinais.loc[linhas, coluna] = 1


def _sinais(rng: np.random.Generator, n: int) -> pd.DataFrame:
    stress = rng.normal(0.0, 1.0, n)

    def par(logit_leve: float, logit_grave: float) -> tuple[np.ndarray, np.ndarray]:
        grave = _flag(rng, logit_grave + 0.45 * stress)
        leve = _flag(rng, logit_leve + 0.40 * stress)
        return np.maximum(leve, grave), grave

    usou, mais_3m = par(-1.55, -2.70)
    cheque, mais_um = par(-2.85, -3.70)
    teto, teto_3x = par(-1.95, -3.05)
    minimo, rotativo = par(-1.90, -3.00)
    parcela = _flag(rng, -2.45 + 0.40 * stress)
    pagou = _flag(rng, -3.55 + 0.45 * stress)
    saque = _flag(rng, -3.35 + 0.40 * stress)
    salto = _flag(rng, -2.25 + 0.35 * stress)
    novo = _flag(rng, -2.05 + 0.35 * stress)
    acumulo = _flag(rng, -2.75 + 0.40 * stress)
    adiant = _flag(rng, -3.15 + 0.40 * stress)
    reneg = _flag(rng, -3.55 + 0.45 * stress)
    resgate = _flag(rng, -3.45 + 0.40 * stress)
    protesto, protesto3 = par(-3.05, -3.85)
    restricao = _flag(rng, -3.60 + 0.40 * stress)
    queda = _flag(rng, -3.05 + 0.40 * stress)
    recente = _flag(rng, -1.65 + 0.20 * stress)
    sem_imovel = _flag(rng, -0.90 + 0.12 * stress)
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
    _plantar(rng, sinais, ((),), max(20, n // 14))
    _plantar(rng, sinais, ISOLADOS, max(12, n // 180))
    _plantar(
        rng,
        sinais,
        (
            (
                "restricao_outras_instituicoes",
                "protesto_cartorio",
                "protesto_3_ou_mais",
                "renegociacao_ativa",
            ),
            (
                "restricao_outras_instituicoes",
                "protesto_cartorio",
                "protesto_3_ou_mais",
                "pagou_fatura_com_cheque_especial",
                "renegociacao_ativa",
            ),
            (
                "restricao_outras_instituicoes",
                "protesto_cartorio",
                "protesto_3_ou_mais",
                "cheque_sem_fundo_12m",
                "mais_de_um_cheque_sem_fundo_12m",
                "pagou_fatura_com_cheque_especial",
            ),
        ),
        max(8, n // 220),
    )
    _plantar(
        rng,
        sinais,
        (
            ("uso_cheque_especial_mais_3m_em_6m", "usou_cheque_especial_6m", "rotativo_mais_2x_6m", "pagamento_minimo_rotativo"),
            ("mais_de_um_cheque_sem_fundo_12m", "cheque_sem_fundo_12m", "pagou_fatura_com_cheque_especial"),
            ("protesto_3_ou_mais", "protesto_cartorio", "renegociacao_ativa"),
            ("resgate_total_reserva", "queda_vinculo_principal"),
            ("cartao_acima_90_mais_3x_6m", "cartao_acima_90_ou_compra_negada", "parcelamento_fatura_6m"),
            ("adiantamento_emergencial", "novo_emprestimo_6m"),
        ),
        max(8, n // 700),
    )
    return sinais


def gerar_base(n: int, seed: int, parametros: dict | None = None, prefixo_id: str = "C") -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    sinais = _sinais(rng, n)
    escore = _escore(sinais)
    ruido = float(parametros["ruido"]) if parametros else RUIDO
    curva = np.log1p(escore)
    p_inad = _sigmoid(BASE_INAD + INCLINACAO * curva + rng.normal(0.0, ruido, n))
    p_grave = _sigmoid(BASE_90 + INCLINACAO_90 * curva + rng.normal(0.0, ruido, n))
    sorteio = rng.random(n)
    inad = (sorteio < p_inad).astype(int)
    acima = (sorteio < np.minimum(p_grave, p_inad)).astype(int)
    if parametros is None:
        parametros = {
            "ruido": RUIDO,
            "base_inad": BASE_INAD,
            "inclinacao": INCLINACAO,
            "base_90": BASE_90,
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
