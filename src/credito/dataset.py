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
from .politica import N_CLIENTES, SEMENTE, TAXA_ALVO_90D, TAXA_ALVO_GERADOR

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

# Peso no logit. O alvo de 90 dias carrega mais as reincidências graves.
PESO_INAD = {coluna: 0.48 for coluna in FEATURES}
PESO_INAD.update(
    {
        "usou_cheque_especial_6m": 0.42,
        "uso_cheque_especial_mais_3m_em_6m": 0.72,
        "cheque_sem_fundo_12m": 0.78,
        "mais_de_um_cheque_sem_fundo_12m": 0.95,
        "cartao_acima_90_ou_compra_negada": 0.5,
        "cartao_acima_90_mais_3x_6m": 0.62,
        "pagamento_minimo_rotativo": 0.55,
        "rotativo_mais_2x_6m": 0.7,
        "parcelamento_fatura_6m": 0.46,
        "pagou_fatura_com_cheque_especial": 0.82,
        "saque_cartao_credito": 0.5,
        "salto_gastos_cartao": 0.4,
        "novo_emprestimo_6m": 0.38,
        "acumulo_emprestimos": 0.58,
        "adiantamento_emergencial": 0.64,
        "renegociacao_ativa": 0.86,
        "resgate_total_reserva": 0.84,
        "protesto_cartorio": 0.6,
        "protesto_3_ou_mais": 0.98,
        "restricao_outras_instituicoes": 0.66,
        "queda_vinculo_principal": 0.74,
        "relacionamento_recente": 0.22,
        "sem_imovel_proprio": 0.16,
    }
)
PESO_90 = {coluna: 0.22 for coluna in FEATURES}
PESO_90.update(
    {
        "uso_cheque_especial_mais_3m_em_6m": 0.55,
        "cheque_sem_fundo_12m": 1.05,
        "mais_de_um_cheque_sem_fundo_12m": 1.45,
        "cartao_acima_90_mais_3x_6m": 0.48,
        "rotativo_mais_2x_6m": 0.5,
        "pagou_fatura_com_cheque_especial": 1.2,
        "adiantamento_emergencial": 0.55,
        "renegociacao_ativa": 1.35,
        "resgate_total_reserva": 1.25,
        "protesto_3_ou_mais": 1.55,
        "restricao_outras_instituicoes": 0.45,
        "queda_vinculo_principal": 1.15,
        "relacionamento_recente": 0.08,
        "sem_imovel_proprio": 0.06,
    }
)
RUIDO = 1.05


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
    if not 0.11 <= taxa <= 0.13:
        raise ValueError(f"Inadimplência saiu em {taxa:.3%}, fora de 11% a 13%.")
    if not 0.05 <= taxa_90 <= 0.07:
        raise ValueError(f"Acima de 90 dias saiu em {taxa_90:.3%}, fora de 5% a 7%.")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12, 12)))


def _flag(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.random(logit.shape[0]) < _sigmoid(logit)).astype(int)


def _sinais(rng: np.random.Generator, n: int) -> pd.DataFrame:
    stress = rng.normal(0.0, 1.0, n)
    usou = _flag(rng, -0.35 + 0.95 * stress)
    mais_3m = usou * _flag(rng, -0.25 + 0.7 * stress)
    cheque = _flag(rng, -1.85 + 1.15 * stress)
    mais_um = cheque * _flag(rng, -0.45 + 0.65 * stress)
    teto = _flag(rng, -1.05 + 0.95 * stress)
    teto_3x = teto * _flag(rng, -0.15 + 0.55 * stress)
    minimo = _flag(rng, -0.95 + 0.9 * stress)
    rotativo = minimo * _flag(rng, -0.05 + 0.6 * stress)
    parcela = _flag(rng, -1.55 + 0.85 * stress)
    pagou = _flag(rng, -2.15 + 1.2 * stress + 1.35 * usou)
    saque = _flag(rng, -2.25 + 1.05 * stress)
    salto = _flag(rng, -1.25 + 0.75 * stress)
    novo = _flag(rng, -1.15 + 0.8 * stress)
    acumulo = _flag(rng, -1.75 + 0.95 * stress)
    adiant = _flag(rng, -2.35 + 1.05 * stress)
    reneg = _flag(rng, -2.55 + 1.2 * stress + 0.7 * np.maximum(minimo, acumulo))
    resgate = _flag(rng, -2.7 + 1.15 * stress)
    protesto = _flag(rng, -2.25 + 1.05 * stress)
    protesto3 = protesto * _flag(rng, -0.95 + 0.75 * stress)
    restricao = _flag(rng, -2.05 + 1.1 * stress)
    queda = _flag(rng, -2.15 + 1.0 * stress)
    recente = _flag(rng, -1.45 + 0.25 * stress)
    sem_imovel = _flag(rng, -0.05 + 0.2 * stress)
    return pd.DataFrame(
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


def _intercepto(base: np.ndarray, ruido: np.ndarray, alvo: float) -> float:
    baixo, alto = -12.0, 8.0
    for _ in range(32):
        meio = (baixo + alto) / 2
        taxa = float(_sigmoid(base + meio + ruido).mean())
        if taxa > alvo:
            alto = meio
        else:
            baixo = meio
    return (baixo + alto) / 2


def _intercepto_90(p_inad: np.ndarray, base: np.ndarray, ruido: np.ndarray, alvo: float) -> float:
    baixo, alto = -12.0, 8.0
    for _ in range(32):
        meio = (baixo + alto) / 2
        p_grave = _sigmoid(base + meio + ruido)
        taxa = float((p_inad * p_grave).mean())
        if taxa > alvo:
            alto = meio
        else:
            baixo = meio
    return (baixo + alto) / 2


def _pesos(mapa: dict[str, float]) -> np.ndarray:
    return np.array([mapa[coluna] for coluna in FEATURES], dtype=float)


def gerar_base(n: int, seed: int, parametros: dict | None = None, prefixo_id: str = "C") -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    sinais = _sinais(rng, n)
    matriz = sinais[FEATURES].to_numpy(dtype=float)
    if parametros is None:
        ruido_inad = rng.normal(0.0, RUIDO, n)
        ruido_90 = rng.normal(0.0, RUIDO, n)
        base_inad = matriz @ _pesos(PESO_INAD)
        base_90 = matriz @ _pesos(PESO_90)
        b_inad = _intercepto(base_inad, ruido_inad, TAXA_ALVO_GERADOR)
        p_inad = _sigmoid(base_inad + b_inad + ruido_inad)
        b_90 = _intercepto_90(p_inad, base_90, ruido_90, TAXA_ALVO_90D)
        p_grave = _sigmoid(base_90 + b_90 + ruido_90)
        parametros = {
            "b_inad": float(b_inad),
            "b_90": float(b_90),
            "ruido": RUIDO,
            "peso_inad": PESO_INAD,
            "peso_90": PESO_90,
        }
        inad = (rng.random(n) < p_inad).astype(int)
        acima = ((rng.random(n) < p_grave) & (inad == 1)).astype(int)
    else:
        ruido_inad = rng.normal(0.0, parametros["ruido"], n)
        ruido_90 = rng.normal(0.0, parametros["ruido"], n)
        p_inad = _sigmoid(matriz @ _pesos(parametros["peso_inad"]) + parametros["b_inad"] + ruido_inad)
        p_grave = _sigmoid(matriz @ _pesos(parametros["peso_90"]) + parametros["b_90"] + ruido_90)
        inad = (rng.random(n) < p_inad).astype(int)
        acima = ((rng.random(n) < p_grave) & (inad == 1)).astype(int)
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
